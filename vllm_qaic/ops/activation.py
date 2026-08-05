# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Custom activation implementations for QAIC platform with torch.compile support."""

import torch

from vllm.model_executor.layers.activation import SiluAndMul

from vllm_qaic.ops._triton_flags import triton_op_enabled


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
        # Triton fast path (opt-in): repurpose vLLM's SWIGLUSTEP Triton kernel
        # (swiglustep_and_mul_triton) as plain SiluAndMul. vLLM ships no plain
        # Triton silu-and-mul; SWIGLUSTEP computes
        #   min(silu(gate), limit) * clamp(up, -limit, limit)
        # which degenerates to silu(gate) * up when limit == +inf (limit is a
        # tl.constexpr, so it specializes cleanly). The kernel is 2D-only
        # ([B, 2*d]) and the caller allocates the output. Qwen3-VL's decoder MLP
        # already feeds 2D [num_tokens, 2*intermediate], but flatten any leading
        # dims to be safe and restore the original shape afterwards. Flag off ->
        # the QAIC NSP swiglu op below, bit-for-bit.
        if triton_op_enabled("SILU_AND_MUL"):
            from vllm.model_executor.layers.activation import (
                swiglustep_and_mul_triton,
            )

            d = x.shape[-1] // 2
            x2d = x.reshape(-1, x.shape[-1])
            out2d = torch.empty((x2d.shape[0], d), dtype=x.dtype, device=x.device)
            swiglustep_and_mul_triton(out2d, x2d, limit=float("inf"))
            return out2d.reshape(*x.shape[:-1], d)

        return torch.ops.qaic.swiglu(x)
