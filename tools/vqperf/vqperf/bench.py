# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Running the upstream serving benchmark against a live server."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import BenchConfig
from .env import Environment
from .server import VllmServer
from .util import RunResult, run, tail_file

logger = logging.getLogger("vqperf.bench")

RESULT_FILENAME = "result.json"


@dataclass
class BenchOutcome:
    status: str  # ok | partial | failed
    reason: str = ""
    result: dict = field(default_factory=dict)
    result_path: Path | None = None
    log_path: Path | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "partial")


def run_benchmark(
    env: Environment,
    config: BenchConfig,
    server: VllmServer,
    *,
    cell_dir: Path,
    sha: str,
    repeat: int,
    num_prompts: int | None = None,
    profile: bool = False,
    timeout_s: float = 3600.0,
    label: str = "measure",
) -> BenchOutcome:
    """Run one benchmark pass and load its result JSON."""
    assert env.bench is not None, "environment has no resolved benchmark invocation"
    cell_dir.mkdir(parents=True, exist_ok=True)
    log_path = cell_dir / "bench.log"

    args = [
        "--backend",
        "openai-chat",
        "--base-url",
        server.base_url,
        *config.client_args(num_prompts=num_prompts),
        # --save-result plus an explicit filename pins the path; otherwise the
        # name embeds a timestamp and the report cannot find it.
        "--save-result",
        "--result-dir",
        str(cell_dir),
        "--result-filename",
        RESULT_FILENAME,
        "--metadata",
        f"vqperf_sha={sha}",
        f"vqperf_config={config.config_id}",
        f"vqperf_repeat={repeat}",
        f"vqperf_pass={label}",
    ]
    if profile:
        args.append("--profile")

    result_path = cell_dir / RESULT_FILENAME
    if result_path.exists():
        result_path.unlink()

    outcome = run(
        env.bench.with_args(args),
        log_path=log_path,
        timeout=timeout_s,
    )
    return _interpret(outcome, config, result_path, log_path, label=label)


def _interpret(
    outcome: RunResult,
    config: BenchConfig,
    result_path: Path,
    log_path: Path,
    *,
    label: str,
) -> BenchOutcome:
    if outcome.timed_out:
        return BenchOutcome(
            "failed",
            f"bench_timeout after {outcome.duration_s:.0f}s",
            log_path=log_path,
            duration_s=outcome.duration_s,
        )
    if outcome.returncode != 0:
        return BenchOutcome(
            "failed",
            f"bench_error rc={outcome.returncode}: {tail_file(log_path, 15)}",
            log_path=log_path,
            duration_s=outcome.duration_s,
        )
    if not result_path.exists():
        return BenchOutcome(
            "failed",
            "bench_error: no result JSON was written",
            log_path=log_path,
            duration_s=outcome.duration_s,
        )

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BenchOutcome(
            "failed",
            f"bench_error: unreadable result JSON ({exc})",
            log_path=log_path,
            duration_s=outcome.duration_s,
        )

    completed = payload.get("completed")
    expected = payload.get("num_prompts", config.num_prompts)
    status, reason = "ok", ""
    both_known = isinstance(completed, int) and isinstance(expected, int)
    if both_known and completed < expected:
        # Partial completion inflates throughput; keep the numbers but say so.
        status = "partial"
        reason = f"only {completed}/{expected} requests completed"
        logger.warning("%s pass %s: %s", config.config_id, label, reason)

    return BenchOutcome(
        status,
        reason,
        result=payload,
        result_path=result_path,
        log_path=log_path,
        duration_s=outcome.duration_s,
    )
