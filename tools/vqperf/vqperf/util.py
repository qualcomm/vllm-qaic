# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared helpers: logging, subprocess execution, locking, ports."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import logging
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("vqperf")


class VqperfError(Exception):
    """Base class for all expected, user-facing failures."""


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@dataclass
class RunResult:
    argv: list[str]
    returncode: int
    duration_s: float
    log_path: Path | None
    timed_out: bool = False
    stdout: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    log_path: Path | None = None,
    capture: bool = False,
    check: bool = False,
) -> RunResult:
    """Run ``argv`` with no shell, teeing output to ``log_path``.

    Every subprocess in vqperf goes through here so that the argv, cwd and
    timeout handling are uniform and each invocation leaves a ``cmd.txt``
    breadcrumb next to its log.
    """
    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        (log_path.parent / f"{log_path.stem}.cmd.txt").write_text(
            format_argv(argv, cwd=cwd, env=env) + "\n", encoding="utf-8"
        )

    logger.debug("exec: %s", shlex.join(argv))
    start = time.monotonic()
    stdout_text = ""
    timed_out = False

    if capture:
        try:
            proc = subprocess.run(  # noqa: S603
                argv,
                cwd=str(cwd) if cwd else None,
                env=full_env,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout_text = (proc.stdout or "") + (proc.stderr or "")
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode, timed_out = -1, True
        if log_path is not None and stdout_text:
            log_path.write_text(stdout_text, encoding="utf-8")
    else:
        # Not a context manager: the handle is the child's stdout for the whole
        # life of the process and is closed in the finally below.
        sink = open(log_path, "wb") if log_path else subprocess.DEVNULL  # noqa: SIM115
        try:
            proc_popen = subprocess.Popen(  # noqa: S603
                argv,
                cwd=str(cwd) if cwd else None,
                env=full_env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = proc_popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_tree(proc_popen)
                returncode = -1
        finally:
            if hasattr(sink, "close"):
                sink.close()

    result = RunResult(
        argv=argv,
        returncode=returncode,
        duration_s=time.monotonic() - start,
        log_path=log_path,
        timed_out=timed_out,
        stdout=stdout_text,
    )
    if check and not result.ok:
        raise VqperfError(
            f"command failed (rc={returncode}, timed_out={timed_out}): "
            f"{shlex.join(argv)}"
            + (f"\nlog: {log_path}" if log_path else "")
            + (f"\n{tail(stdout_text)}" if stdout_text else "")
        )
    return result


def format_argv(
    argv: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    parts = []
    if cwd:
        parts.append(f"# cwd: {cwd}")
    if env:
        for key in sorted(env):
            parts.append(f"{key}={shlex.quote(env[key])} \\")
    parts.append(shlex.join(argv))
    return "\n".join(parts)


def terminate_tree(proc: subprocess.Popen, grace_s: float = 60.0) -> None:
    """SIGTERM the process group, then SIGKILL anything that survives."""
    import signal

    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    logger.warning(
        "pid %s ignored SIGTERM after %.0fs; sending SIGKILL", proc.pid, grace_s
    )
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.error("pid %s survived SIGKILL", proc.pid)


def tail(text: str, lines: int = 40) -> str:
    rows = text.rstrip().splitlines()
    return "\n".join(rows[-lines:])


def tail_file(path: Path | None, lines: int = 40) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return tail(path.read_text(encoding="utf-8", errors="replace"), lines)
    except OSError:
        return ""


def free_port() -> int:
    """Bind-and-release an ephemeral port.

    Inherently racy; callers retry once on EADDRINUSE rather than pretending
    this is a reservation.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) != 0


def wait_for_port_release(port: int, timeout_s: float = 120.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if port_is_free(port):
            return True
        time.sleep(1.0)
    return False


class DeviceLock:
    """Advisory ``flock`` over a device set.

    QAIC devices are exclusive; two concurrent benchmark runs on one device
    produce numbers nobody can trust. Failing loudly beats contending quietly.
    """

    def __init__(self, workdir: Path, devices: str) -> None:
        self.paths = [
            workdir / f"device-{dev.strip()}.lock"
            for dev in devices.split(",")
            if dev.strip()
        ] or [workdir / "device-none.lock"]
        self._handles: list = []

    def __enter__(self) -> DeviceLock:
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "a+")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                handle.seek(0)
                holder = handle.read().strip() or "unknown"
                handle.close()
                self.__exit__(None, None, None)
                raise VqperfError(
                    f"device lock {path} is held by {holder}. Another vqperf run "
                    f"(or another user) is using these devices."
                ) from exc
            handle.seek(0)
            handle.truncate()
            handle.write(f"pid={os.getpid()} host={socket.gethostname()}")
            handle.flush()
            self._handles.append(handle)
        return self

    def __exit__(self, *_exc: object) -> None:
        while self._handles:
            handle = self._handles.pop()
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def short_sha(sha: str) -> str:
    return sha[:12]
