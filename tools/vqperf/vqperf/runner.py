# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Orchestration: the cell plan, provisioning, and serialized execution."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path

from . import hostinfo, metrics, profiler
from .bench import run_benchmark
from .config import BenchConfig
from .env import Environment, EnvManager
from .repo import RepoManager, resolve_fork_url
from .server import VllmServer
from .util import DeviceLock, VqperfError, short_sha, tail_file

logger = logging.getLogger("vqperf.runner")

SIDES = ("base", "new")


@dataclass
class CellResult:
    """One measured pass of one (side, config, repeat)."""

    side: str
    sha: str
    config_id: str
    repeat: int
    status: str
    reason: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    result_path: str = ""
    server_ready_s: float | None = None
    bench_duration_s: float = 0.0
    started_at: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "partial")


@dataclass
class ProfileResult:
    side: str
    sha: str
    config_id: str
    mode: str
    status: str  # ok | failed | unsupported
    reason: str = ""
    torch_dir: str = ""
    lrt_dir: str = ""
    lrt_trace_count: int = 0
    artifacts: list[dict] = field(default_factory=list)
    op_rows: int = 0


@dataclass
class RunState:
    run_id: str
    out_dir: Path
    args: dict
    configs: list[BenchConfig]
    envs: dict[str, Environment] = field(default_factory=dict)
    cells: list[CellResult] = field(default_factory=list)
    profiles: list[ProfileResult] = field(default_factory=list)
    op_tables: dict[str, dict[str, list[profiler.OpRow]]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    host: dict = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""

    def env_for(self, side: str) -> Environment:
        return self.envs[side]


class Runner:
    def __init__(self, options) -> None:  # options: argparse.Namespace
        self.opt = options
        self.workdir = Path(options.workdir).expanduser().resolve()
        self.out_dir = Path(options.out).expanduser().resolve()
        self.cache_dir = self.workdir / "cache"
        self.fork_url = resolve_fork_url(options.fork)

    # -- provisioning ------------------------------------------------------

    def provision(self, state: RunState) -> None:
        repo = RepoManager(self.workdir, self.fork_url)
        repo.ensure_mirror()

        refs = {"base": self.opt.base_sha, "new": self.opt.new_sha}
        checkouts = {}
        for side, ref in refs.items():
            sha, remote = repo.resolve(ref)
            checkouts[side] = repo.checkout(sha, remote)
            logger.info(
                "%s = %s (%s) from %s",
                side,
                short_sha(sha),
                checkouts[side].subject,
                remote,
            )

        if checkouts["base"].sha == checkouts["new"].sha:
            state.warnings.append(
                f"base and new resolve to the same commit "
                f"{short_sha(checkouts['base'].sha)}; the comparison measures "
                "only noise"
            )
            logger.warning("%s", state.warnings[-1])

        for side, checkout in checkouts.items():
            (self.out_dir / "provision" / checkout.short).mkdir(
                parents=True, exist_ok=True
            )
            (self.out_dir / "provision" / checkout.short / "git-info.txt").write_text(
                repo.git_info_text(checkout), encoding="utf-8"
            )

        manager = EnvManager(
            self.workdir,
            mode=self.opt.mode,
            python_version=self.opt.python,
            out_dir=self.out_dir,
        )

        # Builds need no hardware, so run them concurrently and report both
        # failures together instead of one at a time.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                side: pool.submit(manager.provision, checkout, reuse=self.opt.reuse_env)
                for side, checkout in checkouts.items()
            }
            errors = []
            for side, future in futures.items():
                try:
                    state.envs[side] = future.result()
                except VqperfError as exc:
                    errors.append(f"{side} ({checkouts[side].short}): {exc}")
            if errors:
                raise VqperfError(
                    "environment provisioning failed:\n  " + "\n  ".join(errors)
                )

        base_vllm = state.envs["base"].vllm_version
        new_vllm = state.envs["new"].vllm_version
        if base_vllm != new_vllm:
            state.warnings.append(
                f"client_version_mismatch: base pins vllm {base_vllm}, new pins "
                f"{new_vllm}. The benchmark client measuring each side differs; "
                f"this is expected when the comparison *is* the pin bump."
            )
            logger.warning("%s", state.warnings[-1])

        styles = {side: env.profiler_style for side, env in state.envs.items()}
        if len(set(styles.values())) > 1 and any(
            config.profile.enabled for config in state.configs
        ):
            state.warnings.append(
                f"profiler_style_mismatch: {styles}. Op tables remain comparable "
                f"(same key_averages format) but flag handling differs."
            )
            logger.warning("%s", state.warnings[-1])

    # -- plan --------------------------------------------------------------

    def plan(self, state: RunState) -> list[tuple[str, BenchConfig, int]]:
        cells: list[tuple[str, BenchConfig, int]] = []
        if self.opt.profile_only:
            return cells
        repeats = range(self.opt.repeats)
        if self.opt.order == "grouped":
            for side in SIDES:
                for config in state.configs:
                    for repeat in repeats:
                        cells.append((side, config, repeat))
        else:
            # Interleaved: slow drift (thermals, host load) hits both commits
            # roughly equally instead of penalizing whichever ran second.
            for repeat in repeats:
                for config in state.configs:
                    for side in SIDES:
                        cells.append((side, config, repeat))
        return cells

    # -- execution ---------------------------------------------------------

    def execute(self, state: RunState) -> None:
        cells = self.plan(state)
        total = len(cells)
        for index, (side, config, repeat) in enumerate(cells, start=1):
            logger.info(
                "[%d/%d] %s %s repeat %d",
                index,
                total,
                side,
                config.config_id,
                repeat,
            )
            result = self._run_cell(state, side, config, repeat)
            state.cells.append(result)
            if not result.ok and not self.opt.keep_going:
                raise VqperfError(
                    f"cell failed ({side} {config.config_id} r{repeat}): "
                    f"{result.reason}\nRe-run with --keep-going to continue past "
                    f"failures."
                )

        self._run_profiles(state)

    def _cell_dir(self, side: str, config: BenchConfig, repeat: int, sha: str) -> Path:
        return (
            self.out_dir
            / "raw"
            / config.config_id
            / f"{side}-{short_sha(sha)}"
            / f"r{repeat}"
        )

    def _run_cell(
        self, state: RunState, side: str, config: BenchConfig, repeat: int
    ) -> CellResult:
        env = state.env_for(side)
        cell_dir = self._cell_dir(side, config, repeat, env.sha)
        started = datetime.now(UTC).isoformat(timespec="seconds")

        cached = self._reuse_cell(side, env.sha, config, repeat, cell_dir)
        if cached is not None:
            logger.info("reusing existing result for %s r%d", config.config_id, repeat)
            return cached

        attempt = CellResult(
            side=side,
            sha=env.sha,
            config_id=config.config_id,
            repeat=repeat,
            status="failed",
            started_at=started,
        )
        try:
            with VllmServer(
                env,
                config,
                log_path=cell_dir / "server.log",
                devices=self.opt.devices,
                cache_dir=self.cache_dir,
                timeout_s=self.opt.server_timeout,
            ) as server:
                attempt.server_ready_s = server.ready_after_s

                if self.opt.warmup_requests > 0:
                    logger.info("warmup: %d requests", self.opt.warmup_requests)
                    run_benchmark(
                        env,
                        config,
                        server,
                        cell_dir=cell_dir / "warmup",
                        sha=env.sha,
                        repeat=repeat,
                        num_prompts=self.opt.warmup_requests,
                        timeout_s=self.opt.bench_timeout,
                        label="warmup",
                    )

                outcome = run_benchmark(
                    env,
                    config,
                    server,
                    cell_dir=cell_dir,
                    sha=env.sha,
                    repeat=repeat,
                    timeout_s=self.opt.bench_timeout,
                )
        except VqperfError as exc:
            attempt.reason = _short_reason(exc, cell_dir / "server.log")
            logger.error("cell failed: %s", attempt.reason)
            self._write_cell(cell_dir, attempt)
            return attempt

        attempt.status = outcome.status
        attempt.reason = outcome.reason
        attempt.bench_duration_s = outcome.duration_s
        if outcome.ok:
            attempt.metrics = metrics.extract(outcome.result)
            attempt.result_path = str(outcome.result_path or "")
        self._write_cell(cell_dir, attempt)
        return attempt

    def _reuse_cell(
        self, side: str, sha: str, config: BenchConfig, repeat: int, cell_dir: Path
    ) -> CellResult | None:
        if not self.opt.reuse_results:
            return None
        record = cell_dir / "vqperf-cell.json"
        if not record.exists():
            return None
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("sha") != sha or payload.get("status") not in ("ok", "partial"):
            return None
        return CellResult(**payload)

    @staticmethod
    def _write_cell(cell_dir: Path, result: CellResult) -> None:
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "vqperf-cell.json").write_text(
            json.dumps(asdict(result), indent=2), encoding="utf-8"
        )

    # -- profiling ---------------------------------------------------------

    def _run_profiles(self, state: RunState) -> None:
        wanted = [config for config in state.configs if config.profile.enabled]
        if not wanted:
            return
        logger.info("profiling pass for %d configuration(s)", len(wanted))
        for config in wanted:
            for side in SIDES:
                result = self._run_profile_cell(state, side, config)
                state.profiles.append(result)
                if result.status == "ok":
                    rows = (
                        profiler.load_op_table(Path(result.torch_dir))
                        if (result.torch_dir)
                        else []
                    )
                    if rows:
                        state.op_tables.setdefault(config.config_id, {})[side] = rows
                        result.op_rows = len(rows)

    def _profile_dir(self, side: str, config: BenchConfig, sha: str) -> Path:
        return self.out_dir / "profile" / config.config_id / f"{side}-{short_sha(sha)}"

    def _run_profile_cell(
        self, state: RunState, side: str, config: BenchConfig
    ) -> ProfileResult:
        env = state.env_for(side)
        spec = config.profile
        base_dir = self._profile_dir(side, config, env.sha)
        torch_dir = (base_dir / "torch").resolve()
        lrt_dir = (base_dir / "lrt").resolve()

        result = ProfileResult(
            side=side,
            sha=env.sha,
            config_id=config.config_id,
            mode=spec.mode,
            status="failed",
            torch_dir=str(torch_dir) if spec.wants_torch else "",
            lrt_dir=str(lrt_dir) if spec.wants_lrt else "",
        )

        if spec.wants_torch and env.profiler_style != "profiler_config":
            result.status = "unsupported"
            result.reason = (
                f"vllm {env.vllm_version} does not accept --profiler-config; "
                f"host (torch) profiling is unavailable in this environment"
            )
            return result

        extra_args: list[str] = []
        extra_env: dict[str, str] = {}
        if spec.wants_torch:
            torch_dir.mkdir(parents=True, exist_ok=True)
            extra_args += profiler.torch_profiler_flags(spec, torch_dir)
        if spec.wants_lrt:
            lrt_dir.mkdir(parents=True, exist_ok=True)
            extra_env.update(profiler.lrt_env(spec, lrt_dir))

        logger.info("profiling %s on %s (%s)", config.config_id, side, spec.mode)
        try:
            with VllmServer(
                env,
                config,
                log_path=base_dir / "server.log",
                devices=self.opt.devices,
                cache_dir=self.cache_dir,
                extra_args=extra_args,
                extra_env=extra_env,
                timeout_s=self.opt.server_timeout,
            ) as server:
                outcome = run_benchmark(
                    env,
                    config,
                    server,
                    cell_dir=base_dir,
                    sha=env.sha,
                    repeat=0,
                    num_prompts=config.effective_profile_prompts,
                    profile=spec.wants_torch,
                    timeout_s=self.opt.bench_timeout,
                    label="profile",
                )
        except VqperfError as exc:
            result.reason = _short_reason(exc, base_dir / "server.log")
            return result

        if not outcome.ok:
            result.reason = outcome.reason
            return result

        # A 200 from /start_profile with no trace means the profiler never
        # started; that is a failure, not a silent pass.
        if spec.wants_torch and profiler.torch_trace_is_empty(torch_dir):
            result.status = "failed"
            result.reason = (
                f"profile_empty: no torch trace under {torch_dir}. The benchmark "
                f"reported success, so /start_profile was accepted but produced "
                f"nothing."
            )
            logger.error("%s", result.reason)
            return result

        artifacts = profiler.index_torch_traces(torch_dir) if spec.wants_torch else []
        lrt = profiler.index_lrt_traces(lrt_dir) if spec.wants_lrt else []
        result.lrt_trace_count = len(lrt)
        result.artifacts = [
            {
                "kind": item.kind,
                "path": str(item.path),
                "size_bytes": item.size_bytes,
                "note": item.note,
            }
            for item in (*artifacts, *lrt)
        ]
        if spec.wants_lrt and not lrt:
            result.status = "failed"
            result.reason = (
                f"profile_empty: no lrt_trace_iter*_pid* directories under {lrt_dir}"
            )
            return result

        result.status = "ok"
        return result

    # -- top level ---------------------------------------------------------

    def run(self, state: RunState) -> None:
        state.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        state.host = hostinfo.collect(self.opt.devices)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.provision(state)

        # Hardware is the serializing resource; hold the lock for the whole run.
        with DeviceLock(self.workdir, self.opt.devices):
            started = time.monotonic()
            self.execute(state)
            logger.info("execution finished in %.0fs", time.monotonic() - started)

        state.finished_at = datetime.now(UTC).isoformat(timespec="seconds")


def _short_reason(exc: Exception, log_path: Path) -> str:
    text = str(exc).strip().splitlines()
    head = text[0] if text else exc.__class__.__name__
    tail = tail_file(log_path, 8)
    return f"{head}" + (f" | log tail: {tail.splitlines()[-1]}" if tail else "")


def aggregate_state(
    state: RunState,
) -> dict[str, dict[str, dict[str, metrics.Aggregate]]]:
    """``{config_id: {side: {metric: Aggregate}}}`` over successful cells."""
    grouped: dict[str, dict[str, list[dict[str, float]]]] = {}
    for cell in state.cells:
        if not cell.ok:
            continue
        grouped.setdefault(cell.config_id, {}).setdefault(cell.side, []).append(
            cell.metrics
        )
    return {
        config_id: {side: metrics.aggregate(runs) for side, runs in sides.items()}
        for config_id, sides in grouped.items()
    }
