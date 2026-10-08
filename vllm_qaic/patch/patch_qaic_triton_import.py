# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""
This patch is required for enabling Triton with the QAIC backend.
It imports vLLM's Triton utilities, re-enables ``HAS_TRITON``, and makes
``has_triton_kernels()`` recognize the Triton kernels bundled with vLLM when
the Qualcomm backend is registered.
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
    """Expose Triton kernels after QAIC's Triton backend is registered.

    ``import_triton_kernels`` also aliases vLLM's vendored package as the
    top-level ``triton_kernels`` module, which is how the GPT-OSS kernels
    import it.
    """
    if "triton_kernels" not in sys.modules:
        # Retry if vLLM cached an import attempt made before QAIC registered
        # its Triton backend.
        import_utils.import_triton_kernels.cache_clear()
        import_utils.import_triton_kernels()
    return True


try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    if QAIC_TRITON_BACKEND_KEY in triton.backends.backends:
        triton_utils.HAS_TRITON = True
        triton_utils.triton = triton
        triton_utils.tl = tl
        triton_utils.tldevice = tldevice
        import_utils.has_triton_kernels = _qaic_has_triton_kernels
        import vllm_qaic.patch.patch_triton_kernels_opt_flags  # noqa: F401
        import vllm_qaic.patch.patch_triton_kernels_layout  # noqa: F401
except Exception:
    triton_utils.HAS_TRITON = False
