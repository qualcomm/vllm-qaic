# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared helpers for the QAIC paged-attention hardware tests."""

import torch


def lbnhc_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    device,
    fill: float = 0.0,
) -> torch.Tensor:
    """The kv_cache vLLM's create_kv_cache_views hands forward().

    A [Nb, HKV, B, 2D] view (QAicTorchAttentionBackend.get_kv_cache_shape)
    of token-major [Nb, B, HKV, 2D] pages; split(D, -1) gives K and V.
    """
    kv = torch.full(
        (num_blocks, block_size, num_kv_heads, 2 * head_size),
        fill,
        dtype=torch.float16,
        device=device,
    )
    return kv.permute(0, 2, 1, 3)


def ref_attention(q, k_full, v_full, q_lens, seq_lens, scale, sinks, window_left):
    """Causal fp32 reference over per-request context + new rows.

    q: [T, H, D]; k_full/v_full: per-request [seq_len, HKV, D]. The q_len
    query rows of a request sit at positions seq_len - q_len .. seq_len - 1.
    """
    out, t = [], 0
    num_heads = q.shape[1]
    for q_len, seq_len, k, v in zip(q_lens, seq_lens, k_full, v_full, strict=True):
        group = num_heads // k.shape[1]
        qi = q[t : t + q_len].float().movedim(0, 1)  # [H, Lq, D]
        t += q_len
        k = k.float().repeat_interleave(group, dim=1).movedim(0, 1)
        v = v.float().repeat_interleave(group, dim=1).movedim(0, 1)
        scores = qi @ k.transpose(-1, -2) * scale
        q_pos = torch.arange(seq_len - q_len, seq_len).unsqueeze(1)
        k_pos = torch.arange(seq_len).unsqueeze(0)
        mask = k_pos <= q_pos
        if window_left >= 0:
            mask &= (q_pos - k_pos) <= window_left
        scores = scores.masked_fill(~mask, float("-inf"))
        if sinks is not None:
            sink = sinks.float().view(-1, 1, 1).expand(num_heads, q_len, 1)
            probs = torch.softmax(torch.cat([scores, sink], -1), -1)[..., :seq_len]
        else:
            probs = torch.softmax(scores, -1)
        out.append((probs @ v).movedim(1, 0))
    return torch.cat(out)
