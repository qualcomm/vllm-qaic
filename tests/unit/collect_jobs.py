# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Discover parallel-run jobs for tests/unit/, one job per test_*.py file.

Mirrors the collection step of ../ci_scripts/collect_jobs.py (used for
device_unit_and_e2e/), but simplified: unit/ tests are pure-Python and never
contend for a real QAIC device pool (confirmed by grepping unit/ for device
acquisition — no test under this tree requests a real device slot; the one
file that used to require hardware, custom_ops/test_grouped_topk_qaic.py, was
a duplicate of device_unit_and_e2e/test_unit_qaic_grouped_topk.py and has been
moved to tests/_deprecated/). So there is no num_devices / export-cold /
compile-cold gating here, unlike the e2e version — a job here is simply "one
test file", the same atomic unit run_all_tests.sh already uses for its own
sequential loop.

Each job's extra pytest args are resolved here, using the exact same routing
rule run_all_tests.sh's main dispatch loop uses (embedding/* -> embed flags,
lora/* -> lora flags, else -> device flags), so scheduler.py doesn't need to
know anything about device/embed/lora semantics — it just runs each job's
already-resolved command.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def discover_test_files() -> list:
    """Same discovery rule as run_all_tests.sh's discover_test_files():
    every test_*.py under this directory, sorted by path relative to it."""
    return sorted(
        str(p.relative_to(SCRIPT_DIR))
        for p in SCRIPT_DIR.rglob("test_*.py")
    )


def feature_matches(rel_path: str, feature: str) -> bool:
    if not feature:
        return True
    return rel_path.startswith(f"{feature}/")


def extra_args_for(rel_path: str, device_flags: list, embed_flags: list, lora_flags: list) -> list:
    """Same routing rule as run_all_tests.sh's main dispatch loop."""
    if rel_path.startswith("embedding/") and embed_flags:
        return embed_flags
    if rel_path.startswith("lora/") and lora_flags:
        return lora_flags
    return device_flags


def collect_jobs(
    feature: str = "",
    device_flags: list = None,
    embed_flags: list = None,
    lora_flags: list = None,
) -> list:
    device_flags = device_flags or []
    embed_flags = embed_flags or []
    lora_flags = lora_flags or []

    jobs = []
    for job_id, rel_path in enumerate(discover_test_files()):
        if not feature_matches(rel_path, feature):
            continue
        jobs.append(
            {
                "job_id": job_id,
                "name": rel_path[: -len(".py")],
                "rel_path": rel_path,
                "extra_args": extra_args_for(rel_path, device_flags, embed_flags, lora_flags),
            }
        )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="path to write the jobs JSON file to"
    )
    parser.add_argument(
        "--feature", default="", help="only collect jobs under this feature subdirectory"
    )
    parser.add_argument(
        "--device-flags", default="[]", help="JSON list of extra pytest args for device-flagged jobs"
    )
    parser.add_argument(
        "--embed-flags", default="[]", help="JSON list of extra pytest args for embedding/* jobs"
    )
    parser.add_argument(
        "--lora-flags", default="[]", help="JSON list of extra pytest args for lora/* jobs"
    )
    args = parser.parse_args()

    jobs = collect_jobs(
        feature=args.feature,
        device_flags=json.loads(args.device_flags),
        embed_flags=json.loads(args.embed_flags),
        lora_flags=json.loads(args.lora_flags),
    )
    Path(args.output).write_text(json.dumps(jobs, indent=2))
    print(f"Collected {len(jobs)} job(s) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
