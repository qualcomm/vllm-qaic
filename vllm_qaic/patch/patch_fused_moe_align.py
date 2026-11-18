# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Run the fused-MoE block alignment (moe_align_block_size) without _moe_C.

``fused_experts_impl`` groups the (token, top-k slot) assignments by expert with
``moe_align_block_size``, whose wrapper allocates the output buffers and then
calls the compiled op ``torch.ops._moe_C.moe_align_block_size`` -- no CustomOp
dispatch and no native fallback -- and that op is not built for QAIC. vLLM has
no Triton version of it, so this rebinds ``vllm._custom_ops.moe_align_block_size``
to a PyTorch version reproducing the CUDA op's semantics
(csrc/libtorch_stable/moe/moe_align_sum_kernels.cu, v0.30.0):

* An assignment is kept only if its expert id is in ``[0, num_experts)`` and,
  with ``expert_map``, ``expert_map[id]`` is not negative; kept assignments are
  grouped by the (mapped) expert id.
* Each expert's count is padded up to a multiple of ``block_size`` and the
  experts are laid out in order at the exclusive prefix sum of the padded counts.
* ``sorted_token_ids`` is filled with ``numel`` (the padding value), then each
  expert's region gets the flat indices of its assignments.
* ``experts_ids[b]`` is the expert owning block ``b``; every block past the
  padded total is -1.
* ``num_tokens_post_pad[0]`` is the padded total.

Within an expert the CUDA op's order is unspecified (atomics); this writes the
flat indices in ascending order. The fused-MoE GEMMs compute each row
independently, so the result does not depend on that order.
"""

import torch

from vllm import _custom_ops


def _qaic_moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """moe_align_block_size with the CUDA op's semantics, in PyTorch."""
    # Mirrors the CUDA op's input check for parity; not a limit of this impl.
    padded_num_experts = (num_experts + 31) // 32 * 32
    if padded_num_experts >= 1024:
        raise RuntimeError(
            "moe_align_block_size: padded_num_experts must be less than 1024"
        )

    device = topk_ids.device
    numel = topk_ids.numel()
    flat = topk_ids.reshape(-1).long()

    # Keep the assignments the CUDA op keeps, as their local expert id.
    valid = (flat >= 0) & (flat < num_experts)
    local = torch.where(valid, flat, 0)
    if expert_map is not None:
        local = expert_map[local].long()
        valid &= local >= 0
    flat_idx = torch.arange(numel, device=device)[valid]
    local = local[valid]

    # Per-expert counts, padded to block_size, and each expert's start offset.
    counts = torch.bincount(local, minlength=num_experts)[:num_experts]
    padded = (counts + block_size - 1) // block_size * block_size
    offsets = torch.cumsum(padded, 0) - padded
    total = int(padded.sum().item())

    # Group by expert (stable, so ascending flat index within an expert) and
    # give each assignment its rank inside its expert's region.
    order = torch.argsort(local, stable=True)
    local = local[order]
    flat_idx = flat_idx[order]
    rank = (
        torch.arange(local.numel(), device=device)
        - (torch.cumsum(counts, 0) - counts)[local]
    )

    sorted_token_ids.fill_(numel)
    sorted_token_ids[offsets[local] + rank] = flat_idx.to(sorted_token_ids.dtype)

    experts_ids.fill_(-1)
    num_blocks = total // block_size
    if num_blocks:
        experts_ids[:num_blocks] = torch.repeat_interleave(
            torch.arange(num_experts, device=device), padded // block_size
        ).to(experts_ids.dtype)

    num_tokens_post_pad.fill_(total)


# moe_align_block_size.py does `from vllm import _custom_ops as ops` and resolves
# ops.moe_align_block_size at call time, so rebinding the module attribute is
# enough.
_custom_ops.moe_align_block_size = _qaic_moe_align_block_size
