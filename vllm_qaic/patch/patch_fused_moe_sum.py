# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Run the fused-MoE top-k reduction (moe_sum) without the CUDA-only _moe_C op.

``fused_experts_impl`` reduces the per-expert outputs with ``ops.moe_sum``, which
calls the compiled op ``torch.ops._moe_C.moe_sum`` -- no CustomOp dispatch and no
native fallback -- and that op is not built for QAIC. This rebinds
``vllm._custom_ops.moe_sum`` to a version that reproduces the CUDA op's semantics
(csrc/libtorch_stable/moe/moe_align_sum_kernels.cu, v0.30.0):
``output[t] = sum over the top-k slots of input[t, k]``, accumulated in fp32 and
written (not added) to ``output``. When ``topk_ids`` is given, a slot is skipped if
its expert id is negative or, with ``expert_map``, if ``expert_map[id]`` is
negative; a token whose slots are all skipped gets 0. ``expert_map`` without
``topk_ids`` is ignored, as in the CUDA op. The default runs that in PyTorch;
``VLLM_QAIC_TRITON_MOE_SUM=1`` runs upstream's Triton ``masked_moe_sum`` kernel
instead.
"""

import torch

from vllm import _custom_ops

from vllm_qaic.ops._triton_flags import triton_op_enabled


def _valid_slots(
    topk_ids: torch.Tensor, expert_map: torch.Tensor | None
) -> torch.Tensor:
    """Mask of top-k slots the CUDA op would add, mirroring its skip rule."""
    valid = topk_ids >= 0
    if expert_map is not None:
        # Clamp only to index safely; slots with a negative id are already False.
        mapped = expert_map[topk_ids.clamp(min=0).long()]
        valid &= mapped >= 0
    return valid


def _qaic_moe_sum(
    input: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> None:
    """moe_sum with the CUDA op's semantics, on PyTorch or upstream Triton."""
    if not output.is_contiguous():
        raise RuntimeError("moe_sum: output must be contiguous")
    hidden_size = input.size(-1)
    num_tokens = output.numel() // hidden_size
    topk = input.size(1)
    if topk_ids is not None:
        if topk_ids.size(0) != num_tokens or topk_ids.size(1) != topk:
            raise RuntimeError("moe_sum: topk_ids must be [num_tokens, topk]")
        if expert_map is not None and expert_map.dtype != torch.int32:
            raise RuntimeError("moe_sum: expert_map must be int32")
    else:
        # The CUDA op only applies expert_map through topk_ids.
        expert_map = None

    if triton_op_enabled("MOE_SUM"):
        from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (  # noqa: E501
            masked_moe_sum,
        )

        # masked_moe_sum skips slots whose id is -1 and has no expert_map, so
        # fold the CUDA op's full skip rule into the ids it reads.
        if topk_ids is None:
            ids = torch.zeros(
                (num_tokens, topk), dtype=torch.int32, device=input.device
            )
        else:
            ids = torch.where(_valid_slots(topk_ids, expert_map), 0, -1).to(torch.int32)
        masked_moe_sum(
            input.contiguous(),
            ids.contiguous(),
            output.view(num_tokens, hidden_size),
        )
        return

    acc = input.float()
    if topk_ids is not None:
        acc = acc.masked_fill(~_valid_slots(topk_ids, expert_map).unsqueeze(-1), 0.0)
    output.copy_(acc.sum(dim=1).view_as(output))


# fused_moe does `from vllm import _custom_ops as ops` and resolves ops.moe_sum
# at call time, so rebinding the module attribute is enough.
_custom_ops.moe_sum = _qaic_moe_sum
