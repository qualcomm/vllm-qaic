# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Rejection-sampler Numba≡Triton parity tiers.

* Tier A (``@pytest.mark.triton_parity``; ``test_triton_parity_*.py``):
  mandatory differential tests against the real upstream triton-cpu kernels.
  Skipped (with a loud summary banner) when triton-cpu is unavailable, or an
  error when ``VLLM_QAIC_REQUIRE_TRITON_PARITY=1``.
* Tier C (``test_upstream_drift_guard.py``): source hashes of the upstream
  kernels and call sites; no triton needed.

    VLLM_QAIC_REQUIRE_TRITON_PARITY=1 TRITON_CPU_BACKEND=1 .venv_aot/bin/python \
        -m pytest -s tests/unit/spec_decode/rejection_parity
"""

import os

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import pytest  # noqa: E402
from _triton_probe import triton_cpu_status  # noqa: E402

REQUIRE_ENV = "VLLM_QAIC_REQUIRE_TRITON_PARITY"
MARKER = "triton_parity"
SKIP_REASON = (
    "triton-cpu not installed; Numba≡Triton parity NOT verified "
    f"(set {REQUIRE_ENV}=1 to make this an error)"
)

HAS_TRITON_CPU, TRITON_CPU_REASON = triton_cpu_status()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        f"{MARKER}: differential Numba-vs-upstream-triton-cpu rejection-sampler "
        "test (skipped without triton-cpu)",
    )


def pytest_collection_modifyitems(config, items):
    parity = [it for it in items if it.get_closest_marker(MARKER) is not None]
    if not parity or HAS_TRITON_CPU:
        return
    if os.environ.get(REQUIRE_ENV, "0") not in ("", "0"):
        raise pytest.UsageError(
            f"{REQUIRE_ENV}=1 but triton-cpu is unavailable ({TRITON_CPU_REASON}); "
            f"{len(parity)} Numba≡Triton parity tests cannot run."
        )
    skip = pytest.mark.skip(reason=f"{SKIP_REASON} [{TRITON_CPU_REASON}]")
    for item in parity:
        item.add_marker(skip)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    n = sum(
        1
        for rep in terminalreporter.stats.get("skipped", [])
        if MARKER in getattr(rep, "keywords", {}) and SKIP_REASON in str(rep.longrepr)
    )
    if not n:
        return
    bar = "!" * 78
    tr = terminalreporter
    tr.write_line("")
    tr.write_line(bar, red=True, bold=True)
    tr.write_line(
        f"!!  {n} Numba≡Triton parity test(s) SKIPPED: triton-cpu unavailable",
        red=True,
        bold=True,
    )
    tr.write_line(f"!!  ({TRITON_CPU_REASON})", red=True, bold=True)
    tr.write_line(
        "!!  Only the drift-guard (tier C) checks ran; tier A was not verified.",
        red=True,
        bold=True,
    )
    tr.write_line(
        f"!!  Run mandatory tier A in a triton-cpu venv with {REQUIRE_ENV}=1.",
        red=True,
        bold=True,
    )
    tr.write_line(bar, red=True, bold=True)
