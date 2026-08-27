#!/usr/bin/env python3
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
run_tests.py — Auto-detect AOT vs. eager environment and run the matching
vllm-qaic test tier(s).

Detection uses the exact same signal QaicPlatform itself uses to decide
is_aot (vllm_qaic/platform_base.py:64-65): whether the `torch_qaic` package
is importable. This is deliberately NOT re-derived from conda env names or
any other heuristic, so detection here can never drift from what the source
actually branches on.

Two tiers, two different mechanisms (see the design note in
unit/generic/test_platform.py::TestEagerModeBranches and
device_unit_and_e2e/test_unit_qaic_eager_mode.py for the full rationale):

  unit    — pure-Python tests under unit/. These always run in full
            regardless of detected mode: the eager-branch tests in this
            tier drive QaicPlatform.check_and_update_config()'s eager
            branches via monkeypatch.setattr(QaicPlatform, "is_aot", False),
            not via a real torch_qaic install, so they exercise eager logic
            in ANY environment. Wraps unit/run_all_tests.sh unchanged.

  device  — on-device tests under device_unit_and_e2e/. These build real
            vllm_runner(...)/LLM(...) instances, so whichever mode the
            launching environment's torch_qaic installation resolves to is
            the mode actually exercised — no driver-side branching needed
            for that. Detected mode here is used only for the banner and
            for `--mode` mismatch warnings; the same `pytest
            device_unit_and_e2e/` invocation runs unchanged in either case.

Usage:
    python3 run_tests.py                        # auto-detect, run everything
    python3 run_tests.py --tier unit            # pure-Python tier only
    python3 run_tests.py --tier device          # on-device tier only
    python3 run_tests.py --mode eager           # force-check eager, warn on mismatch
    python3 run_tests.py -- --model-name ... --device-id 0   # passthrough args

Any arguments after a literal `--`, or any unrecognized arguments, are
forwarded verbatim to the underlying unit/run_all_tests.sh and/or
`pytest device_unit_and_e2e/` invocations.
"""

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def detect_mode() -> str:
    """Mirror QaicPlatform.is_aot's own detection signal exactly
    (vllm_qaic/platform_base.py:64): torch_qaic importable => eager-capable."""
    return "eager" if importlib.util.find_spec("torch_qaic") is not None else "aot"


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def run_unit_tier(passthrough: list[str]) -> tuple[int, float]:
    script = SCRIPT_DIR / "unit" / "run_all_tests.sh"
    print(f"\n>>> [unit] bash {script} {' '.join(passthrough)}")
    start = time.monotonic()
    rc = subprocess.run(["bash", str(script), *passthrough]).returncode
    return rc, time.monotonic() - start


def run_device_tier(passthrough: list[str]) -> tuple[int, float]:
    target = SCRIPT_DIR / "device_unit_and_e2e"
    print(f"\n>>> [device] python3 -m pytest {target} {' '.join(passthrough)}")
    start = time.monotonic()
    rc = subprocess.run(
        [sys.executable, "-m", "pytest", str(target), *passthrough]
    ).returncode
    return rc, time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["auto", "aot", "eager"], default="auto",
        help="Environment mode to report/target. 'auto' (default) detects "
             "via torch_qaic importability, matching QaicPlatform.is_aot. "
             "'aot'/'eager' force a mode for the banner and warn if it "
             "doesn't match what's actually installed — this does not skip "
             "or select tests, since both tiers already run unchanged "
             "regardless of mode (see module docstring).",
    )
    parser.add_argument(
        "--tier", choices=["all", "unit", "device"], default="all",
        help="Which test tier(s) to run.",
    )
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    actual_mode = detect_mode()
    mode = actual_mode if args.mode == "auto" else args.mode
    if args.mode != "auto" and args.mode != actual_mode:
        print(
            f"WARNING: --mode {args.mode} requested but this environment "
            f"looks like {actual_mode} (torch_qaic "
            f"{'found' if actual_mode == 'eager' else 'not found'}). "
            f"Proceeding with the requested banner, but the actual test "
            f"behavior below is driven by the real environment, not by "
            f"this flag."
        )

    print(f"Detected mode: {actual_mode}" + (
        f"  (reporting as: {mode}, per --mode override)" if mode != actual_mode else ""
    ))

    results: dict[str, tuple[int, float]] = {}
    if args.tier in ("all", "unit"):
        results["unit"] = run_unit_tier(passthrough)
    if args.tier in ("all", "device"):
        results["device"] = run_device_tier(passthrough)

    print("\n" + "=" * 60)
    print("run_tests.py SUMMARY")
    print("=" * 60)
    for tier, (rc, elapsed) in results.items():
        print(f"  {tier:8s} exit code: {rc}   elapsed: {_fmt_elapsed(elapsed)}")
    if len(results) > 1:
        print(f"  {'total':8s}{'':17s}elapsed: "
              f"{_fmt_elapsed(sum(e for _, e in results.values()))}")
    print(f"  mode: {mode}")

    return 1 if any(rc != 0 for rc, _ in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
