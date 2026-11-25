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


_TOPK_SOFTMAX_HAS_EPILOGUE: bool | None = None


def _topk_softmax_has_epilogue() -> bool:
    """True if this torch_qaic's ``qaic::topk_softmax`` takes the vLLM epilogue.

    Newer torch_qaic builds add ``token_expert_indices`` and ``is_padding`` to
    the schema; older ones (e.g. SDK 1.24.0.24) only take
    ``(topk_weights, topk_ids, gating_output, renormalize, bias)``.
    """
    global _TOPK_SOFTMAX_HAS_EPILOGUE
    if _TOPK_SOFTMAX_HAS_EPILOGUE is None:
        try:
            schema = torch.ops.qaic.topk_softmax.default._schema
            names = {arg.name for arg in schema.arguments}
            _TOPK_SOFTMAX_HAS_EPILOGUE = {"token_expert_indices", "is_padding"} <= names
        except (AttributeError, RuntimeError):
            _TOPK_SOFTMAX_HAS_EPILOGUE = False
    return _TOPK_SOFTMAX_HAS_EPILOGUE


def _try_qaic_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    is_padding: torch.Tensor | None = None,
) -> bool:
    """Run the fused ``qaic::topk_softmax`` kernel; False if not applicable.

    One NSP kernel (softmax + top-K + optional renormalize) writing straight
    into the caller's buffers. Benchmarked against the plugin's
    ``qaic::topk_router`` and the eager decomposition (E in {64,128,256},
    K in {6,8}, 1..4096 tokens, fp16/fp32): 1.5x-4x faster than topk_router in
    every case, 1.3x-36x faster than eager, with identical expert selection.

    Follows vLLM's ``topk_softmax`` contract: an fp32
    ``e_score_correction_bias`` only affects which experts are selected, and the
    weights are still the unbiased softmax probabilities.

    When the kernel has the fused epilogue, it also fills
    ``token_expert_indices`` and sets the ids of ``is_padding`` rows to -1 in
    the same launch. Older kernels leave ``token_expert_indices`` unwritten.
    Every QAIC router consumer (``FusedTopKRouter`` / ``GroupedTopKRouter``)
    discards it, so the copy that used to fill it was dead device work.
    Without the epilogue, ``is_padding`` rows are masked with one extra op.
    """
    num_experts = gating_output.shape[-1]
    if (
        gating_output.device.type != "qaic"
        or gating_output.dtype not in (torch.float16, torch.float32)
        or gating_output.dim() != 2
        or not (1 <= topk_weights.shape[1] <= num_experts)
        or topk_weights.dtype != torch.float32
        or topk_ids.dtype != torch.int32
        or not topk_weights.is_contiguous()
        or not topk_ids.is_contiguous()
    ):
        return False
    bias = e_score_correction_bias
    if bias is not None and (
        bias.dtype != torch.float32
        or bias.dim() != 1
        or bias.shape[0] != num_experts
        or bias.device != gating_output.device
    ):
        return False
    if _topk_softmax_has_epilogue():
        fuse_indices = (
            token_expert_indices.dtype == torch.int32
            and token_expert_indices.shape == topk_ids.shape
            and token_expert_indices.is_contiguous()
        )
        fuse_padding = (
            is_padding is not None
            and is_padding.dtype == torch.bool
            and is_padding.dim() == 1
            and is_padding.shape[0] >= topk_ids.shape[0]
        )
        torch.ops.qaic.topk_softmax(
            topk_weights,
            topk_ids,
            gating_output,
            renormalize,
            bias,
            token_expert_indices if fuse_indices else None,
            is_padding if fuse_padding else None,
        )
        if is_padding is not None and not fuse_padding:
            topk_ids.masked_fill_(is_padding[: topk_ids.shape[0], None], -1)
    else:
        torch.ops.qaic.topk_softmax(
            topk_weights, topk_ids, gating_output, renormalize, bias
        )
        if is_padding is not None:
            topk_ids.masked_fill_(is_padding[: topk_ids.shape[0], None], -1)
    return True


def _topk_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor | None,
    scoring_func: str,
    is_padding: torch.Tensor | None = None,
) -> None:
    if scoring_func == "softmax" and _try_qaic_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        is_padding,
    ):
        return

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
    is_padding: torch.Tensor | None = None,
) -> None:
    _topk_torch(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
        "softmax",
        is_padding,
    )


def _topk_sigmoid_torch(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
    is_padding: torch.Tensor | None = None,
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
