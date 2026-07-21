# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
import torch
from torch import Tensor

from vllm.logger import init_logger
from vllm.platforms import current_platform
import torch_qaic.custom_ops as _qaic_custom_ops

logger = init_logger(__name__)

_rms_norm_kernel = _qaic_custom_ops.rms_norm_dispatch
_NSP_COUNT = current_platform.get_num_cores()
_THREAD_COUNT = current_platform.get_num_hvx_threads()

def rms_norm_hexagon(
    attn_out: torch.Tensor,  # previous layer's output (acts as the residual)
    x: torch.Tensor,  # current layer's input
    weight: torch.Tensor,  # per-channel scale (gamma)
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the fused add+RMS-norm to the Hexagon NSP kernel.

    Returns (normed, new_residual) where:
      new_residual = attn_out + x   (stored in FP16 by the kernel)
      normed = new_residual / rms(new_residual) * weight
    """
    out = torch.empty_like(attn_out)
    dst = torch.empty_like(attn_out)
    # Launch grid: [_NSP_COUNT cores, _THREAD_COUNT HVX threads per core]
    _rms_norm_kernel[_NSP_COUNT, _THREAD_COUNT](
        attn_out,
        x,
        weight,
        out,      # written as residual by the kernel
        dst,      # written as normed output by the kernel
        float(epsilon),
        attn_out.shape[-1],
        attn_out.numel(),
        int(attn_out.dtype == torch.bfloat16),
    )

    return dst, out


current_platform.import_kernels()
# Set QAIC_ROUTER_FP32_SCORING=1 to use fp32 softmax/sigmoid math instead of
# the default fp16 HVX math.  The kernel receives one score-mode id:
# 0=fp16 softmax, 1=fp32 softmax, 2=fp16 sigmoid, 3=fp32 sigmoid.
_FP32_SCORING = os.environ.get("QAIC_ROUTER_FP32_SCORING", "0") == "1"


def _kernel(name: str):
    """Return a compiled QAIC Hexagon kernel by exported symbol name."""
    return getattr(_qaic_custom_ops, name)


def _kernel_score_mode(scoring_func: int) -> int:
    """Map public scoring id to the fp16/fp32 score-mode id expected by kernels.

    Kernel score-mode encoding:
      0 = fp16 softmax  (default for scoring_func=0)
      1 = fp32 softmax  (QAIC_ROUTER_FP32_SCORING=1, scoring_func=0)
      2 = fp16 sigmoid  (default for scoring_func=1)
      3 = fp32 sigmoid  (QAIC_ROUTER_FP32_SCORING=1, scoring_func=1)
    """
    if scoring_func == 0:
        return 1 if _FP32_SCORING else 0
    return 3 if _FP32_SCORING else 2


@torch.library.custom_op(
    "qaic::grouped_topk_router", mutates_args=(), device_types="qaic"
)
def _grouped_topk_router_op(
    x: Tensor,
    bias: Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    use_bias: bool,
    scoring_func: int,
) -> tuple[Tensor, Tensor]:
    """Low-level QAIC custom-op wrapper for grouped top-k MoE routing.

    This is the raw torch.library op registered as "qaic::grouped_topk_router".
    Call ``grouped_topk()`` (the public entry point) instead of this directly —
    it handles bias preparation and dtype normalisation before dispatching here.

    Grouped routing divides experts into ``num_expert_group`` groups, selects
    ``topk_group`` groups per token, then selects ``topk`` experts overall from
    the winning groups.

    Called from:
      ``grouped_topk()`` in this file, which is in turn called from:
      - ``ops.grouped_topk_router._patched_grouped_topk()``   (valid-group path)

    Args:
        x: Router logits, shape (num_tokens, num_experts), dtype float16.
        bias: Per-expert score-correction bias, shape (num_experts,), dtype float16.
              Passed but ignored when use_bias=False.
        num_expert_group: Number of expert groups experts are partitioned into.
        topk_group: Number of groups to select per token.
        topk: Total number of experts to select per token.
        renormalize: Whether to renormalize selected weights to sum to 1.
        routed_scaling_factor: Scalar multiplier applied to routed weights.
        use_bias: Whether to add ``bias`` to logits before scoring.
        scoring_func: Kernel score-mode id (0=fp16 softmax … 3=fp32 sigmoid).

    Returns:
        (topk_weights, topk_ids): weight tensor (num_tokens, topk) float32 and
        expert-index tensor (num_tokens, topk) int32.
    """

    num_tokens = x.shape[0]
    num_experts = x.shape[1]
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=x.device)

    kernel = _kernel("multinsp_multithreaded_grouped_topk_router")
    kernel[_NSP_COUNT, _THREAD_COUNT](
        x,
        bias,
        topk_weights,
        topk_ids,
        num_tokens,
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        int(renormalize),
        float(routed_scaling_factor),
        int(use_bias),
        _kernel_score_mode(scoring_func),
    )
    return topk_weights, topk_ids


@torch.library.custom_op("qaic::topk_router", mutates_args=(), device_types="qaic")
def _topk_router_op(
    x: Tensor,
    bias: Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    use_bias: bool,
    scoring_func: int,
) -> tuple[Tensor, Tensor]:
    """Low-level QAIC custom-op wrapper for regular (flat) top-k MoE routing.

    This is the raw torch.library op registered as "qaic::topk_router".
    Call ``regular_topk()`` (the public entry point) instead of this directly —
    it handles bias preparation and dtype normalisation before dispatching here.

    Regular routing scores all experts flatly (softmax or sigmoid over all
    num_experts logits), then picks the top-k experts.  No group structure is
    assumed.

    Called from:
      ``regular_topk()`` in this file, which is in turn called from:
      - ``ops.grouped_topk_router._patched_compute_routing()``  (invalid-group
        fallback inside GroupedTopKRouter._compute_routing)
      - ``qaic_base.QAICPlatformBase._patch_custom_ops()`` patches
        ``vllm._custom_ops.topk_softmax`` / ``topk_sigmoid`` so that
        ``FusedTopKRouter._compute_routing()`` → ``fused_topk()`` →
        ``vllm_topk_softmax()`` dispatches here on QAIC devices.

    Args:
        x: Router logits, shape (num_tokens, num_experts), dtype float16.
        bias: Per-expert bias, shape (num_experts,), dtype float16.
              Passed but ignored when use_bias=False.
        topk: Number of experts to select per token.
        renormalize: Whether to renormalize selected weights to sum to 1.
        routed_scaling_factor: Scalar multiplier applied to routed weights.
        use_bias: Whether to add ``bias`` to logits before scoring.
        scoring_func: Kernel score-mode id (0=fp16 softmax … 3=fp32 sigmoid).

    Returns:
        (topk_weights, topk_ids): weight tensor (num_tokens, topk) float32 and
        expert-index tensor (num_tokens, topk) int32.
    """

    num_tokens = x.shape[0]
    num_experts = x.shape[1]
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=x.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=x.device)

    kernel = _kernel("multinsp_multithreaded_topk_router")
    kernel[_NSP_COUNT, _THREAD_COUNT](
        x,
        bias,
        topk_weights,
        topk_ids,
        num_tokens,
        num_experts,
        topk,
        int(renormalize),
        float(routed_scaling_factor),
        int(use_bias),
        _kernel_score_mode(scoring_func),
    )
    return topk_weights, topk_ids


def _scoring_func_id(scoring_func: str) -> int:
    """Convert vLLM scoring-function string to the kernel-facing scoring id.

    Kernel scoring ids:
      0 = softmax
      1 = sigmoid
    """
    if scoring_func == "softmax":
        return 0
    if scoring_func == "sigmoid":
        return 1
    raise ValueError(f"Unsupported scoring function: {scoring_func}")


def grouped_topk(
    x: Tensor,
    bias: Tensor | None,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    scoring_func: str,
) -> tuple[Tensor, Tensor]:
    """Run the QAIC grouped top-k router for grouped MoE architectures.

    This is the public entry point called by the QAIC monkey-patch for
    GroupedTopKRouter (``ops.grouped_topk_router._patched_grouped_topk``).
    It normalises the bias tensor and then dispatches to the
    ``qaic::grouped_topk_router`` HVX kernel on the QAIC device.

    Args:
        x: Router logits, shape (num_tokens, num_experts), dtype float16.
        bias: Optional per-expert score-correction bias, shape (num_experts,).
              Converted to float16 on x's device if needed.
        num_expert_group: Number of expert groups.
        topk_group: Number of top groups to select per token.
        topk: Total number of experts to select per token.
        renormalize: Whether to renormalize selected weights to sum to 1.
        routed_scaling_factor: Scalar multiplier applied to routed weights.
        scoring_func: "softmax" or "sigmoid".

    Returns:
        (topk_weights, topk_ids): weight tensor (num_tokens, topk) float32 and
        expert-index tensor (num_tokens, topk) int32.
    """
    x = x.contiguous()
    use_bias = bias is not None
    if bias is None:
        bias = torch.empty((x.shape[-1],), dtype=torch.float16, device=x.device)
    elif bias.dtype != torch.float16 or bias.device != x.device:
        bias = bias.to(device=x.device, dtype=torch.float16).contiguous()
    else:
        bias = bias.contiguous()

    return _grouped_topk_router_op(
        x,
        bias,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        use_bias,
        _scoring_func_id(scoring_func),
    )


def regular_topk(
    x: Tensor,
    bias: Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    scoring_func: str,
) -> tuple[Tensor, Tensor]:
    """Run the QAIC regular (flat) top-k router for non-grouped MoE architectures.

    Public entry point called from two places:

    1. ``ops.grouped_topk_router._patched_compute_routing()`` — the invalid-group
       fallback inside GroupedTopKRouter._compute_routing when the group constraints
       cannot be met on this device.

    2. ``qaic_base.QAICPlatformBase._patch_custom_ops()`` patches
       ``vllm._custom_ops.topk_softmax`` / ``topk_sigmoid`` so that the call chain
       FusedTopKRouter._compute_routing → fused_topk → vllm_topk_softmax/sigmoid
       reaches this function on QAIC devices.

    Also serves as the fallback kernel inside GroupedTopKRouter when the
    validated-group path cannot be used (e.g. invalid num_expert_group / topk_group
    for the current batch).

    Args:
        x: Router logits, shape (num_tokens, num_experts), dtype float16.
        bias: Optional per-expert bias, shape (num_experts,).
              Converted to float16 on x's device if needed.
        topk: Number of experts to select per token.
        renormalize: Whether to renormalize selected weights to sum to 1.
        routed_scaling_factor: Scalar multiplier applied to routed weights.
        scoring_func: "softmax" or "sigmoid".

    Returns:
        (topk_weights, topk_ids): weight tensor (num_tokens, topk) float32 and
        expert-index tensor (num_tokens, topk) int32.
    """
    x = x.contiguous()
    use_bias = bias is not None
    if bias is None:
        bias = torch.empty((x.shape[-1],), dtype=torch.float16, device=x.device)
    elif bias.dtype != torch.float16 or bias.device != x.device:
        bias = bias.to(device=x.device, dtype=torch.float16).contiguous()
    else:
        bias = bias.contiguous()

    return _topk_router_op(
        x,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        use_bias,
        _scoring_func_id(scoring_func),
    )
