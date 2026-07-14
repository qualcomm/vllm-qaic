# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Custom activation implementations for QAIC platform with torch.compile support."""

import torch
from vllm.model_executor.layers.activation import SiluAndMul


class QAicSiluAndMul(SiluAndMul):
    """
    QAIC-specific SwiGLU (SiluAndMul) implementation with torch.compile support.

    This is an out-of-tree (OOT) custom operator that replaces vLLM's default
    SiluAndMul implementation for QAIC devices. It delegates to the QAIC custom
    operator ``torch.ops.qaic.swiglu``, which is dispatched to the native NSP
    kernel on QAIC hardware via the QAIC dispatcher.  For unsupported dtypes or
    non-contiguous inputs the custom op falls back to a CPU-side silu+mul
    decomposition.

    The function computes x -> silu(x[..., :d]) * x[..., d:]
    where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """
        QAIC-specific forward implementation.

        Calls the QAIC custom SwiGLU operator which dispatches to the
        NSP tiling kernel on QAIC hardware.

        Args:
            x: Input tensor of shape (..., 2 * d) on QAIC device

        Returns:
            Output tensor of shape (..., d)
        """
        return torch.ops.qaic.swiglu(x)
