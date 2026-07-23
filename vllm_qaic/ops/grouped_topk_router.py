# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Register QAIC grouped top-k router overrides.

The grouped router needs to override module/class routing behavior.
"""

from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.router import grouped_topk_router


_PATCH_APPLIED_ATTR = "_qaic_grouped_topk_router_patch_applied"
_ORIGINAL_GROUPED_TOPK_ATTR = "_qaic_original_grouped_topk"
_ORIGINAL_COMPUTE_ROUTING_ATTR = "_qaic_original_compute_routing"


def _get_qaic_ops():
    try:
        from vllm_qaic import _custom_ops as qaic_ops

        return qaic_ops
    except Exception:
        return None


def _has_valid_expert_grouping(
    num_experts: int, num_expert_group: int, topk_group: int
) -> bool:
    return (
        num_expert_group > 0
        and topk_group > 0
        and topk_group <= num_expert_group
        and num_experts > num_expert_group
        and num_experts % num_expert_group == 0
    )


def _supports_qaic_grouped_topk(
    gating_output: torch.Tensor,
    topk: int,
    num_expert_group: int,
    topk_group: int,
) -> bool:
    return (
        gating_output.device.type == "qaic"
        and gating_output.dtype == torch.float16
        and gating_output.dim() == 2
        and num_expert_group <= 128
        and topk_group <= 128
        and topk <= 64
        and gating_output.shape[-1] <= 1024
        and _has_valid_expert_grouping(
            gating_output.shape[-1], num_expert_group, topk_group
        )
    )


def _supports_qaic_regular_topk(router_logits: torch.Tensor, topk: int) -> bool:
    return (
        router_logits.device.type == "qaic"
        and router_logits.dtype == torch.float16
        and router_logits.dim() == 2
        and topk <= 64
        and router_logits.shape[-1] <= 1024
    )


def _patched_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    qaic_ops = _get_qaic_ops()
    if qaic_ops is not None and _supports_qaic_grouped_topk(
        gating_output, topk, num_expert_group, topk_group
    ):
        return qaic_ops.grouped_topk(
            gating_output,
            e_score_correction_bias,
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            scoring_func,
        )

    original_grouped_topk = getattr(grouped_topk_router, _ORIGINAL_GROUPED_TOPK_ATTR)
    return original_grouped_topk(
        hidden_states=hidden_states,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
    )


def _patched_compute_routing(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    indices_type: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_experts = router_logits.shape[-1]
    valid_grouping = (
        num_experts > self.num_expert_group and num_experts % self.num_expert_group == 0
    )

    qaic_ops = _get_qaic_ops()
    if (
        not valid_grouping
        and qaic_ops is not None
        and _supports_qaic_regular_topk(router_logits, self.top_k)
    ):
        return qaic_ops.regular_topk(
            router_logits,
            self.e_score_correction_bias,
            self.top_k,
            self.renormalize,
            self.routed_scaling_factor
            if self.e_score_correction_bias is not None
            else 1.0,
            self.scoring_func,
        )

    original_compute_routing = getattr(
        grouped_topk_router.GroupedTopKRouter, _ORIGINAL_COMPUTE_ROUTING_ATTR
    )
    return original_compute_routing(self, hidden_states, router_logits, indices_type)


def register_qaic_grouped_topk_router() -> None:
    """Register QAIC grouped-router overrides for valid and fallback paths."""
    if getattr(grouped_topk_router, _PATCH_APPLIED_ATTR, False):
        return

    setattr(
        grouped_topk_router,
        _ORIGINAL_GROUPED_TOPK_ATTR,
        grouped_topk_router.grouped_topk,
    )
    setattr(
        grouped_topk_router.GroupedTopKRouter,
        _ORIGINAL_COMPUTE_ROUTING_ATTR,
        grouped_topk_router.GroupedTopKRouter._compute_routing,
    )

    grouped_topk_router.grouped_topk = _patched_grouped_topk
    grouped_topk_router.GroupedTopKRouter._compute_routing = _patched_compute_routing
    setattr(grouped_topk_router, _PATCH_APPLIED_ATTR, True)
