# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""
This patch is required for enabling triton with QAIC backend.
It imports vllm.triton_utils and re-enables HAS_TRITON.
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

from vllm import triton_utils

QAIC_TRITON_BACKEND_KEY = "qcom_hexagon_backend"

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
