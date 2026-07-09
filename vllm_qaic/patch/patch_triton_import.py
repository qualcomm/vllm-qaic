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

from importlib.metadata import PackageNotFoundError, distributions
from vllm import triton_utils
from vllm.triton_utils import importing as triton_importing

QAIC_TRITON_BACKEND_KEY = "qcom_hexagon_backend"

try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice
    if QAIC_TRITON_BACKEND_KEY in triton.backends.backends.keys():
        triton_utils.HAS_TRITON = True
        triton_utils.triton = triton
        triton_utils.tl = tl
        triton_utils.tldevice = tldevice
except:
    triton_utils.HAS_TRITON = False