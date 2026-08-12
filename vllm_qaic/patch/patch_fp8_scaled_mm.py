# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Register the QAIC fp8 scaled-mm kernels for the OOT platform.

vLLM picks an fp8 linear kernel in ``choose_scaled_mm_linear_kernel`` via
``possible_kernels[current_platform._enum]``.  QAIC's enum is
``PlatformEnum.OOT``, and none of the kernel tables in
``vllm.model_executor.kernels.linear`` has an ``OOT`` key -- so that lookup
raises ``KeyError`` before any ``can_implement`` check runs.  Adding the kernel
classes alone is therefore not enough; the tables need the key.

Two tables are populated, matching the two ways an fp8 checkpoint can carry its
scales (``Fp8LinearMethod.__init__`` branches on whether ``weight_block_size``
is set in the quant config):

* ``_POSSIBLE_FP8_KERNELS`` -- per-tensor / per-token / per-channel scales,
  served by ``QaicFP8ScaledMMLinearKernel``.
* ``_POSSIBLE_FP8_BLOCK_KERNELS`` -- block-quantized scales (e.g. DeepSeek's
  ``[128, 128]`` weight blocks), served by
  ``QaicFP8BlockScaledMMLinearKernel``.

Both kernels dequantize to fp16 ahead of the matmul rather than scaling the
product afterwards; the reasons (fp8 output saturation and the FP16 accumulator
range) are documented in the kernel modules themselves.
"""

from vllm.model_executor.kernels import linear as _linear
from vllm.platforms.interface import PlatformEnum

from vllm_qaic.logger import init_logger
from vllm_qaic.quantization.qaic_fp8_block_scaled_mm import (
    QaicFP8BlockScaledMMLinearKernel,
)
from vllm_qaic.quantization.qaic_fp8_scaled_mm import QaicFP8ScaledMMLinearKernel

logger = init_logger(__name__)

_linear._POSSIBLE_FP8_KERNELS[PlatformEnum.OOT] = [QaicFP8ScaledMMLinearKernel]
_linear._POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT] = [
    QaicFP8BlockScaledMMLinearKernel
]

logger.debug(
    "Registered %s and %s for PlatformEnum.OOT fp8 linear layers",
    QaicFP8ScaledMMLinearKernel.__name__,
    QaicFP8BlockScaledMMLinearKernel.__name__,
)
