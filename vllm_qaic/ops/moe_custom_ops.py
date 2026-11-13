# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC fallbacks for vLLM MoE custom operations unavailable in ``_moe_C``."""

from __future__ import annotations

import torch

_PATCH_APPLIED_ATTR = "_qaic_moe_custom_ops_patch_applied"
_ORIGINAL_MOE_SUM_ATTR = "_qaic_original_moe_sum"
_ORIGINAL_MOE_ALIGN_ATTR = "_qaic_original_moe_align_block_size"


def _moe_align_block_size_pytorch(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Fill upstream's alignment buffers using PyTorch tensor operations.

    Sorted indices refer to flattened token/expert assignments. Pad each expert
    group with topk_ids.numel(), which the Triton GEMMs mask out. vLLM passes an
    expert map here only when it wants remote experts excluded from alignment;
    otherwise its outer wrapper remaps the block expert IDs afterwards.
    """
    flat_ids = topk_ids.reshape(-1).to(torch.long)
    assignments = torch.arange(flat_ids.numel(), device=topk_ids.device)
    valid = (flat_ids >= 0) & (flat_ids < num_experts)
    if expert_map is not None:
        valid &= expert_map[flat_ids.clamp(0, num_experts - 1)] >= 0
    valid_experts = flat_ids[valid]
    assignments = assignments[valid]

    order = torch.argsort(valid_experts, stable=True)
    sorted_experts = valid_experts[order]
    counts = torch.bincount(valid_experts, minlength=num_experts)
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    starts = counts.cumsum(0) - counts
    padded_starts = padded_counts.cumsum(0) - padded_counts
    ranks = torch.arange(assignments.numel(), device=topk_ids.device)
    positions = ranks + (padded_starts - starts)[sorted_experts]

    sorted_token_ids.fill_(flat_ids.numel())
    sorted_token_ids[positions] = assignments[order].to(sorted_token_ids.dtype)
    block_experts = torch.repeat_interleave(
        torch.arange(num_experts, device=topk_ids.device), padded_counts // block_size
    )
    if expert_map is not None:
        block_experts = expert_map[block_experts]
    # The outer wrapper may index expert_map with the entire buffer, including
    # unused blocks. Keep those indices valid even when an expert has no tokens.
    expert_ids.zero_()
    expert_ids[: block_experts.numel()].copy_(block_experts)
    num_tokens_post_pad.copy_(padded_counts.sum().reshape(1))


def _register_qaic_silu_and_mul() -> None:
    """Delegate activation registration while preserving the MoE entry point."""
    from vllm_qaic.ops.activation import register_qaic_silu_and_mul

    register_qaic_silu_and_mul()


def register_qaic_moe_custom_ops() -> None:
    """Provide PyTorch QAIC fallbacks for upstream MoE reduction and alignment."""
    import vllm._custom_ops as ops

    if getattr(ops, _PATCH_APPLIED_ATTR, False):
        return

    _register_qaic_silu_and_mul()

    original_moe_sum = ops.moe_sum
    setattr(ops, _ORIGINAL_MOE_SUM_ATTR, original_moe_sum)
    original_moe_align_block_size = ops.moe_align_block_size
    setattr(ops, _ORIGINAL_MOE_ALIGN_ATTR, original_moe_align_block_size)

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

    def _moe_align_block_size(
        topk_ids: torch.Tensor,
        num_experts: int,
        block_size: int,
        sorted_token_ids: torch.Tensor,
        experts_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        expert_map: torch.Tensor | None = None,
    ) -> None:
        if topk_ids.device.type == "qaic":
            _moe_align_block_size_pytorch(
                topk_ids,
                num_experts,
                block_size,
                sorted_token_ids,
                experts_ids,
                num_tokens_post_pad,
                expert_map,
            )
            return

        original_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_token_ids,
            experts_ids,
            num_tokens_post_pad,
            expert_map,
        )

    ops.moe_align_block_size = _moe_align_block_size
    setattr(ops, _PATCH_APPLIED_ATTR, True)
