# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Register QAIC regular top-k router overrides for vLLM custom ops."""

from __future__ import annotations

from collections.abc import Callable

import torch

_PATCH_APPLIED_ATTR = "_qaic_topk_router_patch_applied"
_QAIC_REGULAR_TOPK: Callable | None = None


def _get_qaic_regular_topk():
    try:
        from vllm_qaic._custom_ops import regular_topk as qaic_regular_topk

        return qaic_regular_topk
    except Exception:
        return None


def _supports_qaic_regular_topk(
    qaic_regular_topk,
    gating_output: torch.Tensor,
    topk: int,
    e_score_correction_bias: torch.Tensor | None,
) -> bool:
    return (
        qaic_regular_topk is not None
        and e_score_correction_bias is None
        and gating_output.device.type == "qaic"
        and gating_output.dtype == torch.float16
        and gating_output.dim() == 2
        and topk <= 64
        and gating_output.shape[-1] <= 1024
    )


def _copy_topk_outputs(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> None:
    topk_weights.copy_(weights)
    topk_ids.copy_(ids)
    token_expert_indices.copy_(ids)


def _topk_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    scoring_func: str,
) -> None:
    qaic_regular_topk = _QAIC_REGULAR_TOPK
    if _supports_qaic_regular_topk(
        qaic_regular_topk,
        gating_output,
        topk_weights.shape[1],
        e_score_correction_bias,
    ):
        assert qaic_regular_topk is not None
        weights, ids = qaic_regular_topk(
            gating_output,
            None,
            topk_weights.shape[1],
            renormalize,
            1.0,
            scoring_func,
        )
        _copy_topk_outputs(topk_weights, topk_ids, token_expert_indices, weights, ids)
        return

    logits = gating_output
    if e_score_correction_bias is not None:
        logits = logits + e_score_correction_bias.unsqueeze(0)
    if scoring_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    else:
        scores = torch.sigmoid(logits)
    weights, ids = torch.topk(scores, topk_weights.shape[1], dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)

    _copy_topk_outputs(topk_weights, topk_ids, token_expert_indices, weights, ids)


def _topk_softmax_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    _topk_torch(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        "softmax",
    )


def _topk_sigmoid_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    _topk_torch(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        "sigmoid",
    )


def register_qaic_topk_router() -> None:
    """Patch vLLM regular top-k custom ops to dispatch to QAIC when supported."""
    import vllm._custom_ops as ops

    if getattr(ops, _PATCH_APPLIED_ATTR, False):
        return

    global _QAIC_REGULAR_TOPK
    _QAIC_REGULAR_TOPK = _get_qaic_regular_topk()

    ops.topk_softmax = _topk_softmax_torch
    ops.topk_sigmoid = _topk_sigmoid_torch
    setattr(ops, _PATCH_APPLIED_ATTR, True)
