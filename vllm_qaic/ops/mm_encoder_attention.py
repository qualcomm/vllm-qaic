# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/layers/attention/mm_encoder_attention.py
"""QAIC-specific ViT (multimodal encoder) attention CustomOp.

Out-of-tree replacement for vLLM's
``vllm.model_executor.layers.attention.mm_encoder_attention.MMEncoderAttention``.

QAIC dispatches to ``forward_oot`` (since ``QAicPlatform._enum = PlatformEnum.OOT``),
where we run an optimized batched SDPA. Three dispatch paths, in priority order:

1. **No segmentation** (``cu_seqlens is None``): a single SDPA call over the full
   packed sequence. Matches upstream ``apply_sdpa`` exactly.

2. **Equal-length fast path**: All segments in cu_seqlens have the same length.
   A single batched SDPA call (reshape segments into the batch dim) handles
   everything. No masks, no loops. This is the common case for same-resolution
   images and for full-attention layers.

3. **Grouped-by-length path**: Segments have variable lengths (edge windows
   in Qwen2.5-VL windowed attention). Instead of N individual SDPA calls,
   segments are grouped by their length and one batched SDPA is issued per
   unique length. For Qwen2.5-VL this means at most 4 SDPA calls (interior,
   right-edge, bottom-edge, corner) instead of up to 40+ individual calls.
"""

import einops
import torch
import torch.nn.functional as F
from vllm.model_executor.layers.attention.mm_encoder_attention import MMEncoderAttention


def _batched_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """SDPA over a (b, s, h, d) packed tensor — matches upstream ``apply_sdpa``."""
    q, k, v = (einops.rearrange(x, "b s h d -> b h s d") for x in (q, k, v))
    out = F.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
    )
    return einops.rearrange(out, "b h s d -> b s h d")


def _grouped_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_lens: torch.Tensor,
    b: int,
    h_q: int,
    h_kv: int,
    d: int,
    total_s: int,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """Run one batched SDPA per unique segment length — no attention masks needed.

    For Qwen2.5-VL window attention with non-divisible grids, there are at most
    4 distinct window sizes (e.g. 64, 32, 32, 16 ViT patches). This function
    groups all same-size windows together and issues one mask-free batched SDPA
    per group, then scatters results back to the original sequence positions.

    Args:
        q: Shape (b, total_s, h_q, d).
        k, v: Shape (b, total_s, h_kv, d).
        cu_seqlens: Shape (n+1,) int32/int64 cumulative lengths.
        seg_lens: Shape (n,) per-segment lengths.
        b, h_q, h_kv, d, total_s: Dimension metadata.
        scale: Optional softmax scale; forwarded to SDPA.
        enable_gqa: Forwarded to SDPA when h_q != h_kv.

    Returns:
        context_layer: Shape (b, total_s, h_q, d).
    """
    cu = cu_seqlens.long()

    # Find unique segment lengths and group indices.
    unique_lens, inverse_indices = torch.unique(seg_lens, return_inverse=True)
    num_groups = unique_lens.shape[0]

    # Pre-allocate output buffer in the (b, s, h_q, d) layout.
    output = torch.empty(b, total_s, h_q, d, dtype=q.dtype, device=q.device)

    for g in range(num_groups):
        group_len = int(unique_lens[g])
        # Indices of segments belonging to this group.
        group_seg_indices = torch.where(inverse_indices == g)[0]
        group_count = group_seg_indices.shape[0]

        # Gather all segments of this length into contiguous tensors.
        segments_q = torch.empty(
            b, group_count, group_len, h_q, d, dtype=q.dtype, device=q.device
        )
        segments_k = torch.empty(
            b, group_count, group_len, h_kv, d, dtype=k.dtype, device=k.device
        )
        segments_v = torch.empty_like(segments_k)

        for local_idx in range(group_count):
            seg_idx = int(group_seg_indices[local_idx])
            start = int(cu[seg_idx])
            end = int(cu[seg_idx + 1])
            segments_q[:, local_idx] = q[:, start:end]
            segments_k[:, local_idx] = k[:, start:end]
            segments_v[:, local_idx] = v[:, start:end]

        # Reshape to (b * group_count, h, group_len, d) for SDPA.
        sq = segments_q.reshape(b * group_count, group_len, h_q, d).transpose(1, 2)
        sk = segments_k.reshape(b * group_count, group_len, h_kv, d).transpose(1, 2)
        sv = segments_v.reshape(b * group_count, group_len, h_kv, d).transpose(1, 2)

        # Single batched SDPA for this group — no mask needed!
        out_g = F.scaled_dot_product_attention(
            sq, sk, sv, dropout_p=0.0, scale=scale, enable_gqa=enable_gqa
        )  # (b * group_count, h_q, group_len, d)

        # Reshape back to (b, group_count, group_len, h_q, d).
        out_g = out_g.transpose(1, 2).reshape(b, group_count, group_len, h_q, d)

        # Scatter results back into the output buffer at the correct positions.
        for local_idx in range(group_count):
            seg_idx = int(group_seg_indices[local_idx])
            start = int(cu[seg_idx])
            end = int(cu[seg_idx + 1])
            output[:, start:end] = out_g[:, local_idx]

    return output


def _qaic_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None,
    cu_seqlens: torch.Tensor | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """QAIC-optimized batched SDPA.

    Args:
        q: Shape (b, total_s, h_q, d) — packed multi-image token sequences.
        k, v: Shape (b, total_s, h_kv, d). For GQA, h_kv may be < h_q and
            ``enable_gqa`` must be set.
        scale: Optional softmax scale forwarded to SDPA.
        cu_seqlens: Optional shape (n+1,) int32 cumulative per-segment lengths,
            e.g. [0, 49, 98, 147] for three 49-token images. ``None`` means a
            single SDPA over the full sequence.
        enable_gqa: Forwarded to ``F.scaled_dot_product_attention``.

    Returns:
        context_layer: Shape (b, total_s, h_q, d).
    """
    # ── No segmentation: single SDPA over the full packed sequence ───────────
    if cu_seqlens is None:
        return _batched_sdpa(q, k, v, scale, enable_gqa)

    b, total_s, h_q, d = q.shape
    h_kv = k.shape[2]
    n = cu_seqlens.shape[0] - 1  # number of segments

    if int(cu_seqlens[-1]) != total_s:
        raise ValueError(
            f"cu_seqlens[-1]={int(cu_seqlens[-1])} != total_s={total_s}; "
            "cu_seqlens must sum to the packed sequence length"
        )

    if n == 0:
        # No segments — return an empty tensor with the expected output layout.
        return torch.empty((b, 0, h_q, d), dtype=q.dtype, device=q.device)

    # Per-segment lengths.
    seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    seg_len = total_s // n

    # ── Fast path: all segments have the same length ─────────────────────────
    # Pure reshape into the batch dimension — no padding, no masking, no .item() calls.
    # Fully static shapes for QAIC. This is the common case when all images are resized
    # to the same resolution before tokenisation.
    if bool(torch.all(seg_lens == seg_len)):
        # (b, n*seg_len, h, d) → (b*n, h, seg_len, d). Use reshape (not view) so
        # non-contiguous QKV inputs fall back to a copy instead of raising.
        def _pack(x: torch.Tensor, h: int) -> torch.Tensor:
            return x.reshape(b * n, seg_len, h, d).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            _pack(q, h_q),
            _pack(k, h_kv),
            _pack(v, h_kv),
            dropout_p=0.0,
            scale=scale,
            enable_gqa=enable_gqa,
        )  # (b*n, h_q, seg_len, d)

        return out.transpose(1, 2).reshape(b, total_s, h_q, d)

    # ── Grouped path: variable-length segments, batched by unique length ─────
    # Instead of N individual SDPA calls, group segments by length and run one
    # batched SDPA per unique length. For Qwen2.5-VL this reduces up to 40+
    # per-window SDPA calls down to at most 4 (one per window type: interior,
    # right-edge, bottom-edge, corner).
    return _grouped_sdpa(
        q,
        k,
        v,
        cu_seqlens,
        seg_lens,
        b,
        h_q,
        h_kv,
        d,
        total_s,
        scale,
        enable_gqa,
    )


class QAicMMEncoderAttention(MMEncoderAttention):
    """QAIC-specific multimodal-encoder attention.

    Out-of-tree CustomOp that replaces vLLM's ``MMEncoderAttention`` for the
    QAIC platform. Inherits ``__init__`` (and therefore the ``num_heads`` /
    ``num_kv_heads`` / ``scale`` / ``view_qkv_to_4d`` plumbing) from the
    upstream class and overrides ``forward_oot`` with the batched-SDPA
    fast paths above.
    """

    def forward_oot(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = _qaic_sdpa(
            q=query,
            k=key,
            v=value,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            enable_gqa=self.num_heads > self.num_kv_heads,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output
