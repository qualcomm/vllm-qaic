# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""
This patch is required for enabling Triton with the QAIC backend.
It imports vLLM's Triton utilities and re-enables ``HAS_TRITON``. It also
imports the optional ``triton_kernels`` package during QAIC startup and reports
its actual availability, without waiting for GPT-OSS model construction.
"""

# What's Patched and how it works:
# --------------------------------
#   1. vllm.triton_utils.HAS_TRITON / triton / tl / tldevice
#    Why:
#       Triton is required to run kernels on the QAIC Triton Backend, but
#       upstream's HAS_TRITON check can end up False even when a usable
#       Triton install is present.
#    How:
#       Re-import triton and check for QAIC Triton Backend in
#       triton.backends.backends; if found, re-enable HAS_TRITON and rebind
#       triton/tl/tldevice on vllm.triton_utils to the real modules.
#    Note:
#       Must run before any vllm module does
#       `from vllm.triton_utils import ...`, since that binds triton/tl/
#       HAS_TRITON into the importing module's own namespace at that
#       moment.
import sys

from vllm import triton_utils
from vllm.utils import import_utils

QAIC_TRITON_BACKEND_KEY = "qcom_hexagon_backend"


def _qaic_has_triton_kernels() -> bool:
    """Import Triton kernels and report whether the package is available.

    ``import_triton_kernels`` also aliases vLLM's vendored package as the
    top-level ``triton_kernels`` module, which is how the GPT-OSS kernels
    import it.
    """

    import_utils.import_triton_kernels()
    return "triton_kernels" in sys.modules


def enable_qaic_triton_kernels() -> None:
    """Require Triton kernels already initialized during QAIC startup."""
    if not _qaic_has_triton_kernels():
        raise ImportError(
            "QAIC GPT-OSS requires a compatible triton_kernels package, but "
            "the package could not be imported."
        )


try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    if QAIC_TRITON_BACKEND_KEY in triton.backends.backends:
        triton_utils.HAS_TRITON = True
        triton_utils.triton = triton
        triton_utils.tl = tl
        triton_utils.tldevice = tldevice
except Exception:
    triton_utils.HAS_TRITON = False

# Import Triton kernels if the QAIC Triton backend is available.
# Set opt flags if triton_kernels is present.
if triton_utils.HAS_TRITON and QAIC_TRITON_BACKEND_KEY in triton.backends.backends:
    # Retry imports cached before QAIC's Triton bindings were restored. Load
    # the package now, independently of which model will be constructed.
    import_utils._has_module.cache_clear()
    import_utils.import_triton_kernels.cache_clear()
    import_utils.has_triton_kernels = _qaic_has_triton_kernels
    if _qaic_has_triton_kernels():
        import vllm_qaic.patch.patch_triton_kernels_opt_flags  # noqa: F401
