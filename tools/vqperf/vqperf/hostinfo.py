# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Host, SDK and driver facts, recorded so a mid-run platform change is
detectable after the fact.

vqperf never installs or touches qaic-fw / qaic-kmd / qaic-rt: they are
host-level, affect a shared multi-device machine, and are out of scope. It only
reads their versions.
"""

from __future__ import annotations

import os
import platform
import re
import socket
from pathlib import Path

from .util import run, which

QAIC_PACKAGES = ("qaic-fw", "qaic-kmd", "qaic-rt", "qaic-platform-sdk", "qaic-apps-sdk")


def collect(devices: str) -> dict[str, object]:
    info: dict[str, object] = {
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "qaic_visible_devices": devices,
        "qaic_packages": _qaic_packages(),
        "qaic_util_version": _qaic_util_version(),
        "sdk_version": _sdk_version(),
    }
    return info


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kib = float(line.split()[1])
                return round(kib / (1024 * 1024), 1)
    except (OSError, IndexError, ValueError):
        pass
    return None


def _qaic_packages() -> dict[str, str]:
    """Read installed QAIC package versions via dpkg, if available."""
    out: dict[str, str] = {}
    if not which("dpkg-query"):
        return out
    for name in QAIC_PACKAGES:
        result = run(
            ["dpkg-query", "-W", "-f=${Version}", name],
            capture=True,
            timeout=30,
        )
        version = result.stdout.strip()
        if result.ok and version:
            out[name] = version
    return out


def _qaic_util_version() -> str:
    if not which("qaic-util"):
        return ""
    result = run(["qaic-util", "--version"], capture=True, timeout=30)
    if not result.ok:
        return ""
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def _sdk_version() -> str:
    for candidate in (
        Path("/opt/qti-aic/versions/platform.xml"),
        Path("/opt/qti-aic/versions/apps.xml"),
    ):
        if not candidate.exists():
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"<base_version>([^<]+)</base_version>", text)
        if match:
            return f"{candidate.stem}:{match.group(1).strip()}"
    return ""
