# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""vqperf command line interface."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

from . import report
from .config import ConfigError, parse_configs
from .metrics import DEFAULT_THRESHOLD_PCT
from .runner import Runner, RunState
from .util import VqperfError, setup_logging

logger = logging.getLogger("vqperf")

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_PROVISION = 3
EXIT_CELL_FAILED = 4
EXIT_LOCKED = 5
EXIT_INTERRUPTED = 130

PROFILE_CHOICES = ("off", "torch", "lrt", "both")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vqperf",
        description=(
            "A/B performance comparison for vllm-qaic commits using upstream "
            "vLLM's serving benchmark. See tools/vqperf/README.md."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="provision, benchmark and report")
    _add_common(run_parser)
    _add_run_args(run_parser)

    report_parser = sub.add_parser(
        "report", help="regenerate reports from an existing output directory"
    )
    report_parser.add_argument(
        "--out", required=True, help="output directory of a previous 'vqperf run'"
    )
    _add_report_args(report_parser)
    _add_verbose(report_parser)

    validate_parser = sub.add_parser(
        "validate", help="lint the config CSV and print resolved commands"
    )
    _add_configs(validate_parser)
    _add_mode(validate_parser)
    _add_profile_mode(validate_parser)
    _add_verbose(validate_parser)

    clean_parser = sub.add_parser("clean", help="remove cached clones and envs")
    _add_workdir(clean_parser)
    clean_parser.add_argument(
        "--sha",
        help="remove the checkout and envs for one SHA only (first 12 hex chars "
        "are matched, so pass at least 12)",
    )
    clean_parser.add_argument(
        "--all", action="store_true", help="remove the entire --workdir"
    )
    _add_verbose(clean_parser)

    return parser


def _add_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )


def _add_configs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--configs",
        required=True,
        help="configuration CSV, one benchmark cell per row "
        "(see tools/vqperf/configs/example.csv)",
    )


def _add_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=("aot", "pyt"),
        default="aot",
        help="vllm-qaic execution mode; selects 'scripts/install.sh {aot|pyt}' "
        "(default: %(default)s)",
    )


def _add_workdir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workdir",
        default="./.vqperf",
        help="cache for the git mirror, per-SHA checkouts, conda envs, the "
        "shared HF/QEfficient caches and device locks (default: %(default)s)",
    )


def _add_profile_mode(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="off",
        help="default profile_mode for rows that do not set one; requires "
        "--mode pyt (default: %(default)s)",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="force profiling off, overriding the CSV",
    )


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=DEFAULT_THRESHOLD_PCT,
        help="minimum relative change, in percent, before a metric can be a "
        "REGRESSION or IMPROVEMENT (default: %(default)s)",
    )
    parser.add_argument(
        "--profile-top-ops",
        type=int,
        default=30,
        help="number of ops listed in the profile op diff (default: %(default)s)",
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-sha",
        required=True,
        help="baseline vllm-qaic commit: SHA, branch or tag in --fork",
    )
    parser.add_argument(
        "--new-sha",
        required=True,
        help="candidate vllm-qaic commit: SHA, branch or tag in --fork",
    )
    parser.add_argument(
        "--fork",
        default=None,
        help="owner/repo, git URL or local path (default: qualcomm/vllm-qaic)",
    )
    _add_configs(parser)
    _add_mode(parser)
    parser.add_argument(
        "--python",
        default="3.12",
        help="Python version for each conda env (default: %(default)s)",
    )
    _add_workdir(parser)
    parser.add_argument(
        "--out",
        default=None,
        help="output directory for logs, results and reports "
        "(default: ./results/<timestamp>-<mode>)",
    )
    _add_verbose(parser)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="measured runs per (config, commit); use 3 or more to get a noise "
        "estimate (default: %(default)s)",
    )
    parser.add_argument(
        "--order",
        choices=("interleave", "grouped"),
        default="interleave",
        help="'interleave' alternates base/new per config and repeat so drift "
        "hits both equally; 'grouped' runs all base cells first "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="unmeasured requests sent to each server before the measured "
        "benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--reuse-env",
        action="store_true",
        help="reuse a previously built conda env whose cache key (SHA, mode, "
        "Python, install script hashes) still matches; otherwise rebuild",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="skip cells that already have a successful result for the same "
        "SHA in --out (resume an interrupted run)",
    )
    parser.add_argument(
        "--devices",
        default="0",
        help="comma-separated QAIC device IDs, exported as QAIC_VISIBLE_DEVICES "
        "and locked for the whole run (default: %(default)s)",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=2700.0,
        help="seconds to wait for 'vllm serve' to become healthy, including "
        "compilation (default: %(default)s)",
    )
    parser.add_argument(
        "--bench-timeout",
        type=float,
        default=3600.0,
        help="seconds allowed for one 'vllm bench serve' run (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the CSV and print the server/client commands, then exit",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue past failed cells instead of stopping at the first one",
    )
    _add_report_args(parser)
    _add_profile_mode(parser)
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="run only the profiling passes and skip the measured cells",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "clean":
            return _cmd_clean(args)
    except ConfigError as exc:
        logger.error("%s", exc)
        return EXIT_CONFIG
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return EXIT_INTERRUPTED
    except VqperfError as exc:
        message = str(exc)
        logger.error("%s", message)
        if "device lock" in message:
            return EXIT_LOCKED
        return EXIT_PROVISION
    return EXIT_CONFIG


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def _run_id(args: argparse.Namespace) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{args.mode}"


def _cmd_run(args: argparse.Namespace) -> int:
    if args.repeats < 1:
        raise ConfigError("--repeats must be >= 1")
    if args.profile_only and args.no_profile:
        raise ConfigError("--profile-only and --no-profile are mutually exclusive")

    configs = parse_configs(
        Path(args.configs).expanduser(),
        mode=args.mode,
        default_profile_mode=args.profile,
        force_profile_off=args.no_profile,
    )
    if args.profile_only and not any(c.profile.enabled for c in configs):
        raise ConfigError(
            "--profile-only was given but no configuration enables profiling "
            "(set --profile torch|lrt|both or the profile_mode column)"
        )

    run_id = _run_id(args)
    out_dir = Path(args.out or f"./results/{run_id}").expanduser().resolve()
    args.out = str(out_dir)

    for config in configs:
        for warning in config.warnings:
            logger.warning("%s: %s", config.config_id, warning)

    if args.dry_run:
        _print_plan(args, configs)
        return EXIT_OK

    state = RunState(
        run_id=run_id,
        out_dir=out_dir,
        args=vars(args).copy(),
        configs=configs,
    )
    runner = Runner(args)

    exit_code = EXIT_OK
    try:
        runner.run(state)
    except KeyboardInterrupt:
        logger.warning("interrupted; writing partial reports")
        exit_code = EXIT_INTERRUPTED
    except VqperfError as exc:
        logger.error("%s", exc)
        exit_code = EXIT_LOCKED if "device lock" in str(exc) else EXIT_PROVISION
        if not state.cells:
            raise

    written = report.write_all(
        state, threshold_pct=args.threshold_pct, top_ops=args.profile_top_ops
    )
    logger.info("wrote %d report file(s) to %s", len(written), out_dir)
    for path in written:
        logger.debug("  %s", path)

    failures = [cell for cell in state.cells if not cell.ok]
    if failures and exit_code == EXIT_OK:
        logger.error("%d cell(s) failed; see summary.md", len(failures))
        exit_code = EXIT_CELL_FAILED
    return exit_code


def _print_plan(args: argparse.Namespace, configs) -> None:
    print(f"# plan: {len(configs)} config(s) x 2 commits x {args.repeats} repeat(s)")
    print(f"# base={args.base_sha} new={args.new_sha} mode={args.mode}")
    for config in configs:
        print(f"\n## {config.config_id}")
        print("server: vllm serve " + shlex.join(config.server_args()))
        print("client: vllm bench serve " + shlex.join(config.client_args()))
        if config.profile.enabled:
            from .profiler import torch_profiler_flags

            flags = torch_profiler_flags(
                config.profile, Path("/abs/profile/dir").resolve()
            )
            print(f"profile ({config.profile.mode}): " + shlex.join(flags))


def _cmd_report(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"no manifest.json in {out_dir}; nothing to report on")
    state = _state_from_manifest(out_dir, manifest_path)
    written = report.write_all(
        state, threshold_pct=args.threshold_pct, top_ops=args.profile_top_ops
    )
    logger.info("regenerated %d report file(s) in %s", len(written), out_dir)
    return EXIT_OK


def _state_from_manifest(out_dir: Path, manifest_path: Path) -> RunState:
    """Rebuild enough state to re-render reports without touching hardware."""
    from .config import BenchConfig, ProfileSpec
    from .runner import CellResult, ProfileResult

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    configs: list[BenchConfig] = []
    for entry in payload.get("configurations", []):
        config = BenchConfig(
            config_id=entry["config_id"],
            model=entry["model"],
            max_model_len=int(entry["max_model_len"]),
            num_prompts=int(entry["num_prompts"]),
        )
        config.max_num_seqs = int(entry.get("max_num_seqs", 8))
        config.quantization = entry.get("quantization") or None
        config.kv_cache_dtype = entry.get("kv_cache_dtype") or None
        config.dataset_name = entry.get("dataset_name", "random")
        config.random_input_len = int(entry.get("random_input_len", 1024))
        config.random_output_len = int(entry.get("random_output_len", 128))
        config.request_rate = str(entry.get("request_rate", "inf"))
        max_concurrency = entry.get("max_concurrency")
        config.max_concurrency = (
            int(max_concurrency) if str(max_concurrency).strip() else None
        )
        config.warnings = list(entry.get("warnings", []))
        config.profile = ProfileSpec(
            mode=entry.get("profile", {}).get("profile_mode", "off")
        )
        config.labels = {
            key: value for key, value in entry.items() if key.startswith("x_")
        }
        configs.append(config)

    state = RunState(
        run_id=payload.get("run_id", out_dir.name),
        out_dir=out_dir,
        args=payload.get("args", {}),
        configs=configs,
    )
    state.started_at = payload.get("started_at", "")
    state.finished_at = payload.get("finished_at", "")
    state.host = payload.get("host", {})
    state.warnings = list(payload.get("warnings", []))
    state.cells = [CellResult(**cell) for cell in payload.get("cells", [])]
    state.profiles = [ProfileResult(**item) for item in payload.get("profiles", [])]

    # Environments are not reconstructed (they hold live Checkout objects); the
    # summary header degrades gracefully when they are absent.
    from . import profiler

    for result in state.profiles:
        if result.status == "ok" and result.torch_dir:
            rows = profiler.load_op_table(Path(result.torch_dir))
            if rows:
                state.op_tables.setdefault(result.config_id, {})[result.side] = rows
    return state


def _cmd_validate(args: argparse.Namespace) -> int:
    configs = parse_configs(
        Path(args.configs).expanduser(),
        mode=args.mode,
        default_profile_mode=args.profile,
        force_profile_off=args.no_profile,
    )
    print(f"OK: {len(configs)} configuration(s) in {args.configs} (mode={args.mode})")
    for config in configs:
        print(f"\n## {config.config_id}")
        print("  server: vllm serve " + shlex.join(config.server_args()))
        print("  client: vllm bench serve " + shlex.join(config.client_args()))
        if config.profile.enabled:
            from .profiler import torch_profiler_flags

            print(f"  profile mode: {config.profile.mode}")
            for flag in torch_profiler_flags(config.profile, Path("/tmp/vqperf-trace")):
                print(f"    {flag}")
        for warning in config.warnings:
            print(f"  WARNING: {warning}")
    return EXIT_OK


def _cmd_clean(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).expanduser().resolve()
    if not workdir.exists():
        print(f"nothing to clean: {workdir} does not exist")
        return EXIT_OK

    if args.all:
        targets = [workdir]
    elif args.sha:
        short = args.sha[:12]
        targets = [workdir / "src" / short] + sorted(
            path for path in (workdir / "envs").glob(f"*-{short}") if path.is_dir()
        )
    else:
        raise ConfigError("clean needs --all or --sha <sha>")

    for target in targets:
        if target.exists():
            print(f"removing {target}")
            shutil.rmtree(target)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
