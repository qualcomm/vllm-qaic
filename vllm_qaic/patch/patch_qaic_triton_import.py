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
