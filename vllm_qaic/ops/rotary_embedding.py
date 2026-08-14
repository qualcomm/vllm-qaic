# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Custom rotary embedding implementations for QAIC platform."""

import torch
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm_qaic.ops._triton_flags import triton_op_enabled
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)


def _rope_via_triton_mrope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    enable_fp32_compute: bool = False,
) -> torch.Tensor | None:
    """
    Run plain (single-axis) RoPE through vLLM's triton_mrope kernel.

    With mrope_section = [rotary_dim // 2, 0, 0] the kernel degenerates to plain
    Neox-style RoPE.

    Adaptation on top of that:
    x is [batch (optional), seq_len, num_heads, rotary_dim];
    the kernel wants a flat [num_tokens, num_heads * rotary_dim]
    """
    from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope

    if x.ndim < 3 or cos.shape != sin.shape or cos.ndim < 2:
        return None

    origin_shape = x.shape
    origin_dtype = x.dtype
    n_h, d_h = x.shape[-2], x.shape[-1]
    half = cos.shape[-1]

    # forward_static rotates *all* of x (it chunks the full last dim in two), so
    # ApplyRotaryEmb callers pre-slice x down to the rotary part. The kernel's
    # hd/rd are therefore both this already-sliced last dim.
    if half * 2 != d_h:
        return None
    # The kernel tiles heads as tl.arange(0, pad_hd // 2) over a power-of-2 pad,
    # so an odd rotary half would mask incorrectly.
    head_size = d_h
    rotary_dim = d_h

    num_tokens = x.numel() // (n_h * d_h)
    compute_dtype = torch.float32 if enable_fp32_compute else origin_dtype

    # [num_tokens, num_heads * head_size]; copy=True because the kernel rotates
    # q in place and the caller's x must not be clobbered.
    q = x.reshape(num_tokens, n_h * d_h).to(dtype=compute_dtype, copy=True)

    cos = cos.reshape(-1, half).to(compute_dtype)
    sin = sin.reshape(-1, half).to(compute_dtype)
    if cos.shape[0] != num_tokens:
        if cos.shape[0] == 0 or num_tokens % cos.shape[0] != 0:
            return None
        reps = num_tokens // cos.shape[0]
        cos = cos.unsqueeze(0).expand(reps, -1, -1).reshape(num_tokens, half)
        sin = sin.unsqueeze(0).expand(reps, -1, -1).reshape(num_tokens, half)

    # T row only; H/W rows are fully masked out by mrope_section = [half, 0, 0].
    cos_thw = cos.new_zeros((3, num_tokens, half))
    sin_thw = sin.new_zeros((3, num_tokens, half))
    cos_thw[0] = cos
    sin_thw[0] = sin

    # The kernel signature is fused q+k; feed it a throwaway single-head k.
    k_dummy = q.new_zeros((num_tokens, head_size))

    q, _ = triton_mrope(
        q,
        k_dummy,
        cos_thw,
        sin_thw,
        [half, 0, 0],
        head_size,
        rotary_dim,
        False,  # mrope_interleaved: T/H/W section layout, not GPT-J pairing
    )
    return q.reshape(origin_shape).to(origin_dtype)


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
    def forward_static(
        x, cos, sin, is_neox_style=True, enable_fp32_compute=False
    ) -> torch.Tensor:
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
        if triton_op_enabled("ROPE") and is_neox_style:
            out = _rope_via_triton_mrope(x, cos, sin, enable_fp32_compute)
            return out

        if enable_fp32_compute:
            x = x.to(torch.float32)
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)
        out = torch.ops.qaic.rotary_embedding(x, cos, sin, is_neox_style)

        return out

    def forward_oot(self, x, cos, sin, is_neox_style=True, enable_fp32_compute=False):
        return self.forward_static(x, cos, sin, is_neox_style, enable_fp32_compute)


ApplyRotaryEmb.forward_static = staticmethod(QAicApplyRotaryEmb.forward_static)
