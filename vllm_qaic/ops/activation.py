# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Custom activation implementations for QAIC platform with torch.compile support."""

import torch

from vllm.model_executor.layers.activation import SiluAndMul

from vllm_qaic.ops._triton_flags import triton_op_enabled

triton_en = triton_op_enabled("SILU_AND_MUL")
_QAIC_SILU_AND_MUL_OP = None


class QAicSiluAndMul(SiluAndMul):
    """
    QAIC-specific SwiGLU (SiluAndMul) implementation with torch.compile support.

    This is an out-of-tree (OOT) custom operator that replaces vLLM's default
    SiluAndMul implementation for QAIC devices. When Triton is enabled, it uses
    vLLM's SWIGLUSTEP kernel with an infinite clamp limit. Otherwise it delegates
    to ``torch.ops.qaic.swiglu``, which dispatches to the native NSP kernel or its
    existing fallback for unsupported dtypes or non-contiguous inputs.

    The function computes x -> silu(x[..., :d]) * x[..., d:]
    where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # Stateless so the MoE adapter can call it without constructing a layer,
    # whose initialization requires vLLM's compilation configuration.
    @staticmethod
    def forward_oot(
        x: torch.Tensor, output: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        QAIC-specific forward implementation.

        Share the activation implementation between layer calls and MoE's
        output-buffer operator.

        Args:
            x: Input tensor of shape (..., 2 * d) on QAIC device
            output: Optional caller-provided tensor of shape (..., d)

        Returns:
            Output tensor of shape (..., d), reusing ``output`` when supplied
        """
        # Triton fast path (opt-in): repurpose vLLM's SWIGLUSTEP Triton kernel
        # (swiglustep_and_mul_triton) as plain SiluAndMul. vLLM ships no plain
        # Triton silu-and-mul; SWIGLUSTEP computes
        #   min(silu(gate), limit) * clamp(up, -limit, limit)
        # which degenerates to silu(gate) * up when limit == +inf (limit is a
        # tl.constexpr, so it specializes cleanly). The kernel is 2D-only
        # ([B, 2*d]); flatten any leading dimensions. Reuse MoE's
        # output buffer so Triton can write directly into it.
        if triton_en:
            from vllm.model_executor.layers.activation import swiglustep_and_mul_triton

            d = x.shape[-1] // 2
            if output is None:
                output = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
            x2d = x.reshape(-1, x.shape[-1])
            output2d = output.reshape(-1, d)
            swiglustep_and_mul_triton(output2d, x2d, limit=float("inf"))
            return output

        result = torch.ops.qaic.swiglu(x)
        if output is not None:
            output.copy_(result)
            return output
        return result


def register_qaic_silu_and_mul() -> None:
    """Route upstream's SiLU-and-mul output-buffer operator through forward_oot."""
    global _QAIC_SILU_AND_MUL_OP

    if hasattr(torch.ops._C, "silu_and_mul"):
        return

    @torch.library.custom_op(
        "_C::silu_and_mul",
        mutates_args=("output",),
        device_types="qaic",
    )
    def qaic_silu_and_mul(output: torch.Tensor, input: torch.Tensor) -> None:
        QAicSiluAndMul.forward_oot(input, output=output)

    @qaic_silu_and_mul.register_fake
    def _qaic_silu_and_mul_fake(output: torch.Tensor, input: torch.Tensor) -> None:
        return None

    # Keep the CustomOpDef alive for the lifetime of the process.
    _QAIC_SILU_AND_MUL_OP = qaic_silu_and_mul
