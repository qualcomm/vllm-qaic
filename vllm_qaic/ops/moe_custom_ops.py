# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC fallbacks for vLLM MoE custom operations unavailable in ``_moe_C``."""

from __future__ import annotations

import torch

_PATCH_APPLIED_ATTR = "_qaic_moe_custom_ops_patch_applied"
_ORIGINAL_MOE_SUM_ATTR = "_qaic_original_moe_sum"


def register_qaic_moe_custom_ops() -> None:
    """Use PyTorch reduction when the CUDA ``_moe_C.moe_sum`` is unavailable."""
    import vllm._custom_ops as ops

    if getattr(ops, _PATCH_APPLIED_ATTR, False):
        return

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
