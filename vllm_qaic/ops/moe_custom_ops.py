# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC fallbacks for vLLM MoE custom operations unavailable in ``_moe_C``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_qaic.ops._triton_flags import triton_op_enabled

_PATCH_APPLIED_ATTR = "_qaic_moe_custom_ops_patch_applied"
_ORIGINAL_MOE_SUM_ATTR = "_qaic_original_moe_sum"
_QAIC_SILU_AND_MUL_OP = None


def _register_qaic_silu_and_mul() -> None:
    """Provide the compiled-op contract used by upstream fused MoE on QAIC."""
    global _QAIC_SILU_AND_MUL_OP

    if hasattr(torch.ops._C, "silu_and_mul"):
        return

    @torch.library.custom_op(
        "_C::silu_and_mul",
        mutates_args=("output",),
        device_types="qaic",
    )
    def qaic_silu_and_mul(output: torch.Tensor, input: torch.Tensor) -> None:
        if triton_op_enabled("SILU_AND_MUL"):
            from vllm_qaic.ops.activation import triton_silu_and_mul_out

            triton_silu_and_mul_out(output, input)
            return

        d = input.shape[-1] // 2
        output.copy_(F.silu(input[..., :d]) * input[..., d:])

    @qaic_silu_and_mul.register_fake
    def _qaic_silu_and_mul_fake(output: torch.Tensor, input: torch.Tensor) -> None:
        return None

    # Keep the CustomOpDef alive for the lifetime of the process.
    _QAIC_SILU_AND_MUL_OP = qaic_silu_and_mul


def register_qaic_moe_custom_ops() -> None:
    """Use PyTorch reduction when the CUDA ``_moe_C.moe_sum`` is unavailable."""
    import vllm._custom_ops as ops

    if getattr(ops, _PATCH_APPLIED_ATTR, False):
        return

    _register_qaic_silu_and_mul()

    original_moe_sum = ops.moe_sum
    setattr(ops, _ORIGINAL_MOE_SUM_ATTR, original_moe_sum)

    def _moe_sum(
        input: torch.Tensor,
        output: torch.Tensor,
        topk_ids: torch.Tensor | None = None,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        if input.device.type == "qaic" and topk_ids is None and expert_map is None:
            output.copy_(input.sum(dim=1))
            return

        original_moe_sum(input, output, topk_ids, expert_map)

    ops.moe_sum = _moe_sum
    setattr(ops, _PATCH_APPLIED_ATTR, True)
