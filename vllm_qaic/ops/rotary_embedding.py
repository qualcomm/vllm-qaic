# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Custom rotary embedding implementations for QAIC platform."""

import torch
from vllm.model_executor.layers import rotary_embedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

class QAicApplyRotaryEmb(ApplyRotaryEmb):
    """
    QAIC-specific ApplyRotaryEmbedding implementation.

    This is an out-of-tree (OOT) custom operator that replaces vLLM's default
    RotaryEmbedding implementation for QAIC devices. It delegates to the QAIC custom
    operator ``torch.ops.qaic.rotary_embedding`, which is dispatched to the native NSP
    kernel on QAIC hardware via the QAIC dispatcher.  For unsupported dtypes or
    non-contiguous inputs the custom op falls back to a CPU-side rotary_embedding
    decomposition.

    the staticmethod ensures that any calls to the static method of ApplyRotaryEmb
    are patched to QAicApplyRotaryEmb

    the forward_oot ensures that the regular forward calls on ApplyRotaryEmb
    instances are dispatched to the OOT implementation

    The function computes x -> silu(x[..., :d]) * x[..., d:]
    where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    @staticmethod
    def forward_static(x, cos, sin, is_neox_style=True, enable_fp32_compute=False) -> torch.Tensor:
        """
        QAIC-specific forward implementation.

        Calls the QAIC custom  operator which dispatches to the
        NSP tiling kernel on QAIC hardware.

        Args:
            x: [batch_size (optional), seq_len, num_heads, head_size]
            cos: [seq_len, head_size // 2]
            sin: [seq_len, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style.
            enable_fp32_compute: Temporarily convert x, cos, sin to FP32 dtype
                                 for higher accuracy.

        Returns:
            Output tensor of shape same as input x
        """
        if enable_fp32_compute:
            x = x.to(torch.float32)
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)
        out = torch.ops.qaic.rotary_embedding(x, cos, sin, is_neox_style)

        return out

    def forward_oot(self, x, cos, sin, is_neox_style=True, enable_fp32_compute=False):
        return self.forward_static(x, cos, sin, is_neox_style, enable_fp32_compute)

ApplyRotaryEmb.forward_static = staticmethod(QAicApplyRotaryEmb.forward_static)