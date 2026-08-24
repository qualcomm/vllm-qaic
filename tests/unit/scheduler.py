# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Custom in-process scheduler for running vllm-qaic unit test jobs (see
collect_jobs.py) concurrently, mirroring ../ci_scripts/scheduler.py's
job-dispatch + atomic per-job log printing pattern used for
device_unit_and_e2e/.

Unlike the e2e scheduler, there is no device pool, no export/compile
cold-start gating, and no device acquire/release/cooldown logic here — unit/
tests are pure-Python and don't contend for a scarce QAIC device pool (see
collect_jobs.py's docstring). Concurrency here is bounded purely by a worker
count, via a plain ThreadPoolExecutor.
"""

import argparse
import concurrent.futures
import contextlib
import enum
import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

_DEFAULT_WORKERS = 8
_DEFAULT_TIMEOUT_S = 1800.0


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Job:
    job_id: int
    name: str
    rel_path: str
    extra_args: list
    status: JobStatus = JobStatus.PENDING
    start_time: float = None
    end_time: float = None
    return_code: int = None
    log_path: str = None


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


class Scheduler:
    def __init__(self, jobs: list, script_dir: Path, output_dir: Path, workers: int, timeout_s: float):
        self.jobs = jobs
        self.script_dir = script_dir
        self.output_dir = output_dir
        self.workers = workers
        self.timeout_s = timeout_s
        self.print_lock = threading.Lock()

    def run(self) -> dict:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self._execute_job, job) for job in self.jobs]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        return self._summary()

    def _execute_job(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.start_time = time.monotonic()
        job_dir = self.output_dir / f"job_{job.job_id}_{_slugify(job.name)}"
        log_path = job_dir / "output.log"
        job.log_path = str(log_path)

        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            full_path = self.script_dir / job.rel_path
            cmd = [
                "python3", "-m", "pytest", str(full_path),
                "--tb=short", "-ra",
                *job.extra_args,
            ]
            with open(log_path, "wb") as log_file:
                process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
                try:
                    job.return_code = process.wait(timeout=self.timeout_s)
                    # rc=5 means pytest collected 0 tests — treat as pass,
                    # matching run_all_tests.sh's run_suite() convention.
                    job.status = (
                        JobStatus.PASSED if job.return_code in (0, 5) else JobStatus.FAILED
                    )
                except subprocess.TimeoutExpired:
                    process.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=5)
                    job.status = JobStatus.TIMEOUT
                    job.return_code = -1
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.return_code = -1
            print(f"[scheduler] ERROR: job {job.job_id} ({job.name}) failed to launch/execute: {exc!r}", file=sys.stderr)
        finally:
            job.end_time = time.monotonic()

        self._print_job_log(job)

    def _print_job_log(self, job: Job) -> None:
        """Print each job's full captured output as one contiguous block as
        soon as it finishes, so console output never interleaves two
        concurrently-running jobs' lines — reads like a sequential run even
        though execution underneath is fully parallel."""
        duration = (job.end_time or 0) - (job.start_time or 0)
        try:
            body = Path(job.log_path).read_text(errors="replace")
        except OSError as e:
            body = f"<failed to read log: {e}>"

        with self.print_lock:
            print(f"===== BEGIN {job.name} =====")
            print(body, end="" if body.endswith("\n") else "\n")
            print(f"===== END {job.name}: {job.status.upper()} (rc={job.return_code}, {duration:.1f}s) =====\n")
            sys.stdout.flush()

    def _summary(self) -> dict:
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.status,
                    "return_code": job.return_code,
                    "duration_s": (job.end_time or 0) - (job.start_time or 0) if job.start_time else None,
                    "log_path": job.log_path,
                }
                for job in self.jobs
            ]
        }


def _load_jobs(jobs_path: Path) -> list:
    raw = json.loads(jobs_path.read_text())
    return [
        Job(
            job_id=item["job_id"],
            name=item["name"],
            rel_path=item["rel_path"],
            extra_args=item.get("extra_args", []),
        )
        for item in raw
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_file", help="path to jobs JSON produced by collect_jobs.py")
    parser.add_argument("--script-dir", required=True, help="tests/unit/ directory (job rel_paths are relative to this)")
    parser.add_argument("--output-dir", required=True, help="directory for per-job logs")
    parser.add_argument("--workers", type=int, default=_DEFAULT_WORKERS, help=f"max concurrent jobs (default: {_DEFAULT_WORKERS})")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_S, help=f"per-job wall-clock limit in seconds (default: {_DEFAULT_TIMEOUT_S})")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _load_jobs(Path(args.jobs_file))
    if not jobs:
        print("[scheduler] no jobs to run")
        return 0

    scheduler = Scheduler(jobs, Path(args.script_dir), output_dir, workers=args.workers, timeout_s=args.timeout)
    print(f"[scheduler] running {len(jobs)} job(s) with up to {args.workers} concurrent worker(s)")
    run_start = time.monotonic()
    summary = scheduler.run()
    total_duration_s = time.monotonic() - run_start

    failed = [j for j in summary["jobs"] if j["status"] != "passed"]
    passed_count = len(summary["jobs"]) - len(failed)
    print(f"[scheduler] {passed_count}/{len(summary['jobs'])} job(s) passed")
    print(f"[scheduler] total execution time: {total_duration_s:.1f}s ({timedelta(seconds=round(total_duration_s))})")
    if failed:
        print("[scheduler] failed job(s):")
        for j in failed:
            print(f"  - {j['name']} ({j['status']}, rc={j['return_code']})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
