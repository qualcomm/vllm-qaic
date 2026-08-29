# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

QAIC_QUANTIZATION_METHOD = "mxfp6"
QAIC_KV_CACHE_DTYPE = "mxint8"

QAIC_QUANTIZATION_LIST = [
    QAIC_QUANTIZATION_METHOD,
    "awq",
    "gptq",
    "fp8",
    "compressed-tensors",
    "mxfp4",
]
