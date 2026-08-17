# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Custom in-process scheduler for running vllm-qaic e2e test jobs (see
`collect_jobs.py`) across a pool of QAIC devices in parallel, replacing
`pytest -n <N> --dist=loadscope`.
"""

import argparse
import concurrent.futures
import contextlib
import enum
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import psutil
import regex as re

try:
    from qaicrt import QDevStatus, QStatus
    from qaicrt import Util as _qaic_util
except ImportError:
    import platform

    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    from qaicrt import QDevStatus, QStatus
    from qaicrt import Util as _qaic_util

_DEFAULT_TIMEOUT_S = 1800.0
# Devices need a brief settling window after release before reuse
_DEFAULT_COOLDOWN_S = 5.0
_WAKE_FALLBACK_S = 5.0
_DEVICE_READY_CHECK_TIMEOUT_S = 30.0


def _is_device_ready(qid: int) -> bool:
    """Checks if device is ready and has at least one NSP free."""

    def _check() -> bool:
        api_status, device_info = _qaic_util().getDeviceInfo(qid)
        return (
            api_status == QStatus.QS_SUCCESS
            and device_info.devStatus == QDevStatus.QDS_READY
            and device_info.devData.resourceInfo.nspFree > 0
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_check)
        try:
            return future.result(timeout=_DEVICE_READY_CHECK_TIMEOUT_S)
        except concurrent.futures.TimeoutError:
            print(
                f"[scheduler] device readiness check for qid={qid} timed out "
                f"after {_DEVICE_READY_CHECK_TIMEOUT_S}s",
                file=sys.stderr,
            )
            return False
        except Exception as exc:
            print(
                f"[scheduler] device readiness check for qid={qid} raised: {exc!r}",
                file=sys.stderr,
            )
            return False


def _slugify_scope_id(scope_id: str) -> str:
    """Turn a pytest scope_id (e.g. 'tests/e2e/test_x.py::TestX') into a
    short, filesystem-safe fragment for job directory names - drop the
    directory part, keep the test file/class."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.basename(scope_id)).strip("_")


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
    scope_id: str
    nodeids: list
    model_name: str
    config_key: tuple
    num_devices: int
    base_args: list
    status: JobStatus = JobStatus.PENDING
    device_ids: list = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    return_code: int | None = None
    log_path: str | None = None
    is_export_cold: bool = False
    is_compile_cold: bool = False


class DevicePool:
    """In-memory device pool. The scheduler is the sole authority over
    devices for this run - no cross-process locking needed."""

    def __init__(self, device_ids: list, cooldown_s: float = _DEFAULT_COOLDOWN_S):
        self._free = list(device_ids)
        self._pending: set[int] = set()
        self._cooldown_s = cooldown_s
        self._cooldown_until: dict[int, float] = {}
        self._lock = threading.Lock()

    def try_acquire(self, count: int) -> list | None:
        with self._lock:
            now = time.monotonic()
            available = [d for d in self._free if self._cooldown_until.get(d, 0) <= now]
            if len(available) < count:
                return None
            chosen = available[:count]
            for d in chosen:
                self._free.remove(d)
            return chosen

    def release(self, device_ids: list) -> None:
        # _is_device_ready can block up to _DEVICE_READY_CHECK_TIMEOUT_S per
        # device, so run all checks outside the lock - otherwise a slow or
        # wedged device would stall every other job's dispatch/release.
        with self._lock:
            recheck = list(self._pending)

        candidates = recheck + list(device_ids)
        ready = {d for d in candidates if _is_device_ready(d)}
        not_ready = [d for d in candidates if d not in ready]

        now = time.monotonic()
        with self._lock:
            for d in ready:
                self._pending.discard(d)
                self._cooldown_until[d] = now + self._cooldown_s
                self._free.append(d)
            for d in not_ready:
                self._pending.add(d)

        if not_ready:
            print(
                f"[scheduler] WARNING: device(s) {not_ready} failed readiness check "
                f"on release; quarantined until a future release() re-validates them",
                file=sys.stderr,
            )


class Scheduler:
    def __init__(
        self,
        jobs: list,
        device_ids: list,
        output_dir: Path,
        timeout_s: float,
        cooldown_s: float,
        dry_run: bool = False,
        set_qaic_visible_devices: bool = False,
    ):
        self.jobs = jobs
        self.device_pool = DevicePool(device_ids, cooldown_s=cooldown_s)
        self.output_dir = output_dir
        self.timeout_s = timeout_s
        self.dry_run = dry_run
        self.set_qaic_visible_devices = set_qaic_visible_devices

        self.cond = threading.Condition()
        self.print_lock = threading.Lock()
        self.pending: list[Job] = list(jobs)
        self.running: set[int] = set()

        self.export_gate: dict[str, threading.Event] = {}
        self.compile_gate: dict[tuple, threading.Event] = {}
        self.export_owner: set[str] = set()
        self.compile_owner: set[tuple] = set()
        for job in self.jobs:
            if job.model_name is not None and job.model_name not in self.export_gate:
                self.export_gate[job.model_name] = threading.Event()
            config_key = tuple(job.config_key)
            job.config_key = config_key
            if config_key not in self.compile_gate:
                self.compile_gate[config_key] = threading.Event()

    def _claim_gates(self, job: Job) -> None:
        """First job for a model claims the export gate; first job for an
        exact config claims the compile gate. Both are decided once, at
        queue-build time - "first job seen wins" for each key."""
        if job.model_name is None:
            job.is_export_cold = True
        elif job.model_name not in self.export_owner:
            self.export_owner.add(job.model_name)
            job.is_export_cold = True
        if job.config_key not in self.compile_owner:
            self.compile_owner.add(job.config_key)
            job.is_compile_cold = True

    def _ready(self, job: Job) -> bool:
        if (
            not job.is_export_cold
            and job.model_name is not None
            and not self.export_gate[job.model_name].is_set()
        ):
            return False
        return job.is_compile_cold or self.compile_gate[job.config_key].is_set()

    def _release_gates(self, job: Job) -> None:
        if job.is_compile_cold:
            self.compile_gate[job.config_key].set()
        if job.is_export_cold and job.model_name is not None:
            self.export_gate[job.model_name].set()
        if job.is_export_cold and job.return_code != 0:
            print(
                f"[scheduler] WARNING: export-cold job for model "
                f"'{job.model_name}' ({job.scope_id}) failed (rc={job.return_code}); "
                f"dependent jobs for this model are unblocked and will likely fail "
                f"too.",
                file=sys.stderr,
            )

    def run(self) -> dict:
        for job in self.jobs:
            self._claim_gates(job)

        if self.dry_run:
            for job in self.jobs:
                print(
                    f"[dry-run] job={job.job_id} scope={job.scope_id} "
                    f"model={job.model_name} config={job.config_key} "
                    f"export_cold={job.is_export_cold} "
                    f"compile_cold={job.is_compile_cold} "
                    f"num_devices={job.num_devices}"
                )
            return {"dry_run": True, "jobs": [j.scope_id for j in self.jobs]}

        threads = []
        while True:
            with self.cond:
                if not self.pending and not self.running:
                    break

                still_pending = []
                for job in self.pending:
                    if not self._ready(job):
                        still_pending.append(job)
                        continue
                    device_ids = self.device_pool.try_acquire(job.num_devices)
                    if device_ids is None:
                        still_pending.append(job)
                        continue
                    job.device_ids = device_ids
                    job.status = JobStatus.RUNNING
                    self.running.add(job.job_id)
                    print(
                        f"[scheduler] dispatching job {job.job_id}: {job.scope_id} "
                        f"(devices {job.device_ids})"
                    )
                    t = threading.Thread(
                        target=self._execute_job, args=(job,), daemon=True
                    )
                    threads.append(t)
                    t.start()
                self.pending = still_pending

                if self.pending or self.running:
                    self.cond.wait(timeout=_WAKE_FALLBACK_S)

        for t in threads:
            t.join()

        return self._summary()

    def _execute_job(self, job: Job) -> None:
        # Runs on a daemon thread: everything from here through job.status
        # being set MUST stay inside this try/finally.
        job.start_time = time.monotonic()
        job_dir = (
            self.output_dir / f"job_{job.job_id}_{_slugify_scope_id(job.scope_id)}"
        )
        log_path = job_dir / "output.log"
        job.log_path = str(log_path)

        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python3",
                "-m",
                "pytest",
                "--disable-warnings",
                "-v",
                "-s",
                "-p",
                "no:logging",
                *job.nodeids,
                "--device-id",
                ",".join(str(d) for d in job.device_ids),
                *job.base_args,
            ]
            job_env = None
            if self.set_qaic_visible_devices:
                job_env = os.environ.copy()
                job_env["QAIC_VISIBLE_DEVICES"] = ",".join(
                    str(device_id) for device_id in job.device_ids
                )

            with open(log_path, "wb") as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=job_env,
                )
                try:
                    job.return_code = process.wait(timeout=self.timeout_s)
                    job.status = (
                        JobStatus.PASSED if job.return_code == 0 else JobStatus.FAILED
                    )
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(process.pid)
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=5)
                    job.status = JobStatus.TIMEOUT
                    job.return_code = -1
        except Exception as exc:
            job.status = JobStatus.ERROR
            job.return_code = -1
            print(
                f"[scheduler] ERROR: job {job.job_id} ({job.scope_id}) failed to "
                f"launch/execute: {exc!r}",
                file=sys.stderr,
            )
        finally:
            job.end_time = time.monotonic()

        self._print_job_log(job)

        with self.cond:
            self._release_gates(job)
            self.device_pool.release(job.device_ids)
            self.running.discard(job.job_id)
            self.cond.notify_all()

    @staticmethod
    def _kill_process_tree(pid: int) -> None:
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for proc in children + [parent]:
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=5)
        for proc in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                proc.kill()

    def _print_job_log(self, job: Job) -> None:
        """Print each job's full captured output as one contiguous block as
        soon as it finishes, so console output never interleaves two
        concurrently-running jobs' lines - reads like a sequential run even
        though execution underneath is fully parallel."""
        duration = (job.end_time or 0) - (job.start_time or 0)
        try:
            body = Path(job.log_path).read_text(errors="replace")
        except OSError as e:
            body = f"<failed to read log: {e}>"

        with self.print_lock:
            print(f"===== BEGIN {job.scope_id} (device {job.device_ids}) =====")
            print(body, end="" if body.endswith("\n") else "\n")
            print(
                f"===== END {job.scope_id}: {job.status.upper()} "
                f"(rc={job.return_code}, {duration:.1f}s) =====\n"
            )
            sys.stdout.flush()

    def _summary(self) -> dict:
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "scope_id": job.scope_id,
                    "model_name": job.model_name,
                    "status": job.status,
                    "return_code": job.return_code,
                    "device_ids": job.device_ids,
                    "duration_s": (job.end_time or 0) - (job.start_time or 0)
                    if job.start_time
                    else None,
                    "log_path": job.log_path,
                }
                for job in self.jobs
            ]
        }


def _load_jobs(jobs_path: Path) -> list:
    raw = json.loads(jobs_path.read_text())
    return [
        Job(
            job_id=idx,
            scope_id=item["scope_id"],
            nodeids=item["nodeids"],
            model_name=item["model_name"],
            config_key=tuple(item["config_key"]),
            num_devices=item["num_devices"],
            base_args=item.get("base_args", []),
        )
        for idx, item in enumerate(raw)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "jobs_file", help="path to jobs JSON produced by collect_jobs.py"
    )
    parser.add_argument(
        "--device-ids", required=True, help="comma-separated device id pool"
    )
    parser.add_argument(
        "--output-dir", required=True, help="directory for per-job logs"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT_S,
        help=(
            "per-job wall-clock limit in seconds before it's killed "
            f"(default: {_DEFAULT_TIMEOUT_S})"
        ),
    )
    parser.add_argument(
        "--set-qaic-visible-devices",
        action="store_true",
        help=(
            "set QAIC_VISIBLE_DEVICES to each job's assigned physical IDs "
            "before its pytest subprocess starts"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device_ids = [int(v) for v in args.device_ids.split(",") if v.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _load_jobs(Path(args.jobs_file))
    scheduler = Scheduler(
        jobs,
        device_ids,
        output_dir,
        timeout_s=args.timeout,
        cooldown_s=_DEFAULT_COOLDOWN_S,
        dry_run=args.dry_run,
        set_qaic_visible_devices=args.set_qaic_visible_devices,
    )
    run_start = time.monotonic()
    summary = scheduler.run()
    total_duration_s = time.monotonic() - run_start

    if args.dry_run:
        return 0

    failed = [j for j in summary["jobs"] if j["status"] != "passed"]
    passed_count = len(summary["jobs"]) - len(failed)
    print(f"[scheduler] {passed_count}/{len(summary['jobs'])} job(s) passed")
    print(
        f"[scheduler] total execution time: {total_duration_s:.1f}s "
        f"({timedelta(seconds=round(total_duration_s))})"
    )
    if failed:
        print("[scheduler] failed test(s):")
        for j in failed:
            print(f"  - {j['scope_id']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
