# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Side-effect-free probe: are the upstream rejection kernels real triton-cpu?

``has_triton_cpu()`` is True only when all of the following hold:

* ``triton`` is importable (``importlib.util.find_spec``),
* the triton ``cpu`` backend exists and its driver reports active, and
* ``vllm.v1.sample.rejection_sampler.expand_kernel`` (or, if the QAIC Numba
  backend is already installed, the original it replaced) is a
  ``triton.runtime.jit.JITFunction`` -- i.e. vLLM did not fall back to its
  ``TritonPlaceholder``.

No exception escapes; any failure means "no triton-cpu".  Set
``TRITON_CPU_BACKEND=1`` before the first ``triton`` import (the
``rejection_parity`` conftest does this).

Import path from a sibling test under ``tests/unit/spec_decode`` (pytest
prepend import mode puts that directory on ``sys.path``)::

    from rejection_parity._triton_probe import has_triton_cpu
"""

from __future__ import annotations

import importlib.util
import os

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

_CACHE: dict[str, tuple[bool, str]] = {}


def _upstream_expand_kernel():
    import vllm.v1.sample.rejection_sampler as rs

    kernel = rs.expand_kernel
    try:
        from vllm_qaic.v1.sample import rejection_sampler_numba as rsn

        if rsn._installed and "expand_kernel" in rsn._originals:
            kernel = rsn._originals["expand_kernel"]
    except Exception:  # noqa: BLE001 - the probe must never raise
        pass
    return kernel


def _probe() -> tuple[bool, str]:
    try:
        if importlib.util.find_spec("triton") is None:
            return False, "triton is not installed"
    except Exception as exc:  # noqa: BLE001
        return False, f"find_spec('triton') failed: {exc!r}"
    try:
        from triton.backends import backends

        cpu = backends.get("cpu")
        if cpu is None:
            return False, f"triton has no cpu backend (backends={sorted(backends)})"
        if not (cpu.driver and cpu.driver.is_active()):
            return False, "triton cpu driver is not active"
    except Exception as exc:  # noqa: BLE001
        return False, f"triton backend probe failed: {exc!r}"
    try:
        from triton.runtime.jit import JITFunction

        kernel = _upstream_expand_kernel()
        if not isinstance(kernel, JITFunction):
            return False, (
                "vllm rejection_sampler.expand_kernel is "
                f"{type(kernel).__module__}.{type(kernel).__name__}, not a "
                "triton JITFunction (vLLM TritonPlaceholder in use?)"
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"upstream kernel probe failed: {exc!r}"
    return True, "triton-cpu active; upstream kernels are JITFunctions"


def triton_cpu_status() -> tuple[bool, str]:
    """``(available, human-readable reason)``; cached per process."""
    if "status" not in _CACHE:
        _CACHE["status"] = _probe()
    return _CACHE["status"]


def has_triton_cpu() -> bool:
    return triton_cpu_status()[0]
