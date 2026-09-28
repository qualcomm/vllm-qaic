# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
import torch
from torch import Tensor

from vllm_qaic.logger import init_logger
from vllm.platforms import current_platform
import torch_qaic.custom_ops as _qaic_custom_ops

logger = init_logger(__name__)

_rms_norm_kernel = _qaic_custom_ops.rms_norm_dispatch
_NSP_COUNT = current_platform.get_num_cores()
_THREAD_COUNT = current_platform.get_num_hvx_threads()
_HMX_THREAD_COUNT = _THREAD_COUNT + 1


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
    is_bf16 = attn_out.dtype == torch.bfloat16
    out_dtype = torch.float32 if is_bf16 else attn_out.dtype
    out = torch.empty(attn_out.shape, dtype=out_dtype, device=attn_out.device)
    dst = torch.empty(attn_out.shape, dtype=out_dtype, device=attn_out.device)
    # Launch grid: [_NSP_COUNT cores, _THREAD_COUNT HVX threads per core]
    _rms_norm_kernel[_NSP_COUNT, _THREAD_COUNT](
        attn_out,
        x,
        weight,
        out,  # written as residual by the kernel
        dst,  # written as normed output by the kernel
        float(epsilon),
        attn_out.shape[-1],
        attn_out.numel(),
        int(is_bf16),
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


_paged_attention_store_kernel = _kernel("multinsp_multithreaded_paged_attention_store")
_paged_attention_hvx_kernel = _kernel("multinsp_multithreaded_paged_attention")
_paged_attention_hmx_prefill_cache_kernel = _kernel(
    "multinsp_multithreaded_paged_attention_hmx_prefill"
)
_paged_attention_hmx_prefill_kernel = _kernel(
    "multinsp_multithreaded_paged_attention_hmx_prefill_direct"
)
_paged_attention_hmx_decode_kernel = _kernel(
    "multinsp_multithreaded_paged_attention_hmx_decode"
)
_paged_attention_hmx_decode_cache_kernel = _kernel(
    "multinsp_multithreaded_paged_attention_hmx_decode_cache"
)


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


@torch.library.custom_op(
    "qaic::paged_attention",
    mutates_args=("key_cache", "value_cache"),
    device_types="qaic",
)
def _paged_attention_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    slot_mapping: Tensor,
    query_start_loc: Tensor,
    seq_lens: Tensor,
    scale: float,
    causal: bool,
    write_cache: bool,
) -> Tensor:
    """Run QAIC paged attention over vLLM block-cache metadata."""
    query = query.contiguous()
    if write_cache:
        key = key.contiguous()
        value = value.contiguous()
    if (
        block_table.device != query.device
        or block_table.dtype != torch.int32
        or not block_table.is_contiguous()
    ):
        block_table = block_table.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
    if write_cache and (
        slot_mapping.device != query.device
        or slot_mapping.dtype != torch.int32
        or not slot_mapping.is_contiguous()
    ):
        slot_mapping = slot_mapping.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
    if (
        query_start_loc.device != query.device
        or query_start_loc.dtype != torch.int32
        or not query_start_loc.is_contiguous()
    ):
        query_start_loc = query_start_loc.to(
            device=query.device, dtype=torch.int32
        ).contiguous()
    if (
        seq_lens.device != query.device
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        seq_lens = seq_lens.to(device=query.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(query)

    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    num_reqs = seq_lens.shape[0]
    max_blocks_per_seq = block_table.shape[1]

    backend = os.environ.get("QAIC_PAGED_ATTN_BACKEND")
    if backend is None:
        backend = "hvx" if os.environ.get("QAIC_PAGED_ATTN_HMX", "1") == "0" else "hmx"
    backend = backend.lower()
    if backend not in ("hmx", "hvx"):
        raise RuntimeError(
            f"QAIC_PAGED_ATTN_BACKEND must be 'hmx' or 'hvx', got {backend!r}"
        )

    use_hmx = backend == "hmx"
    is_decode = num_tokens == num_reqs
    hmx_supported = head_dim > 0 and num_kv_heads > 0 and num_heads % num_kv_heads == 0
    if use_hmx and not hmx_supported:
        raise RuntimeError(
            "HMX paged attention requested for unsupported shape: "
            f"tokens={num_tokens}, reqs={num_reqs}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, head_dim={head_dim}"
        )
    if not use_hmx and head_dim > 256:
        raise RuntimeError(
            "HVX paged attention supports head_dim <= 256; "
            f"got head_dim={head_dim}. Set QAIC_PAGED_ATTN_BACKEND=hmx."
        )

    if use_hmx:
        if is_decode:
            attn_kernel = (
                _paged_attention_hmx_decode_kernel
                if write_cache
                else _paged_attention_hmx_decode_cache_kernel
            )
        else:
            attn_kernel = (
                _paged_attention_hmx_prefill_kernel
                if write_cache
                else _paged_attention_hmx_prefill_cache_kernel
            )
    else:
        attn_kernel = _paged_attention_hvx_kernel

    fused_hmx_store = use_hmx and write_cache

    if not fused_hmx_store and write_cache:
        _paged_attention_store_kernel[_NSP_COUNT, _THREAD_COUNT](
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            num_tokens,
            num_kv_heads,
            head_dim,
            block_size,
        )
    if use_hmx and not write_cache:
        attn_kernel[_NSP_COUNT, _HMX_THREAD_COUNT](
            query,
            key_cache,
            value_cache,
            output,
            block_table,
            query_start_loc,
            seq_lens,
            num_reqs,
            num_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            int(causal),
            float(scale),
        )
    elif use_hmx:
        attn_kernel[_NSP_COUNT, _HMX_THREAD_COUNT](
            query,
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            output,
            block_table,
            query_start_loc,
            seq_lens,
            num_reqs,
            num_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            int(causal),
            float(scale),
        )
    else:
        attn_kernel[_NSP_COUNT, _THREAD_COUNT](
            query,
            key_cache,
            value_cache,
            output,
            block_table,
            query_start_loc,
            seq_lens,
            num_reqs,
            num_tokens,
            num_heads,
            num_kv_heads,
            head_dim,
            block_size,
            max_blocks_per_seq,
            int(causal),
            float(scale),
        )
    return output


def paged_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    slot_mapping: Tensor,
    query_start_loc: Tensor,
    seq_lens: Tensor,
    scale: float,
    causal: bool = True,
    write_cache: bool = True,
) -> Tensor:
    """Public test/development entry point for the QAIC paged-attention MVP."""
    return _paged_attention_op(
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        slot_mapping,
        query_start_loc,
        seq_lens,
        scale,
        causal,
        write_cache,
    )
