# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/logger.py
from vllm.logger import _VllmLogger
from vllm.logger import init_logger as _init_vllm_logger


def init_logger(name: str) -> _VllmLogger:
    # Prepend "vllm." so vllm_qaic loggers inherit vllm's configured handlers
    # through propagation.
    return _init_vllm_logger("vllm." + name)
