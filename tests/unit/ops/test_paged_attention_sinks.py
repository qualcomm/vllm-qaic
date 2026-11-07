# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""qaic::paged_attention with attention sinks and a sliding window (gpt-oss),
on vLLM 0.30.0's native LBNHC cache, against an fp32 reference."""

import pytest
import torch

from tests.unit.ops.moe_test_utils import requires_qaic

H, HKV, D, B = 8, 2, 64, 16
NUM_BLOCKS = 64


def _lbnhc_cache(device):
    # What vLLM's create_kv_cache_views hands forward(): a [Nb, H, B, 2D] view
    # of token-major [Nb, B, H, 2D] pages, split into K and V halves.
    kv = torch.zeros(NUM_BLOCKS, B, HKV, 2 * D, dtype=torch.float16, device=device)
    return kv.permute(0, 2, 1, 3).split(D, dim=-1)


def _ref(q, k_full, v_full, q_lens, seq_lens, scale, sinks, window_left):
    """q: [T, H, D]; k_full/v_full: per-request [seq_len, HKV, D]."""
    out, t = [], 0
    for q_len, seq_len, k, v in zip(q_lens, seq_lens, k_full, v_full, strict=True):
        qi = q[t : t + q_len].float().movedim(0, 1)  # [H, Lq, D]
        t += q_len
        k = k.float().repeat_interleave(H // HKV, dim=1).movedim(0, 1)
        v = v.float().repeat_interleave(H // HKV, dim=1).movedim(0, 1)
        scores = qi @ k.transpose(-1, -2) * scale
        q_pos = torch.arange(seq_len - q_len, seq_len).unsqueeze(1)
        k_pos = torch.arange(seq_len).unsqueeze(0)
        mask = k_pos <= q_pos
        if window_left >= 0:
            mask &= (q_pos - k_pos) <= window_left
        scores = scores.masked_fill(~mask, float("-inf"))
        if sinks is not None:
            sink = sinks.float().view(-1, 1, 1).expand(H, q_len, 1)
            probs = torch.softmax(torch.cat([scores, sink], -1), -1)[..., :seq_len]
        else:
            probs = torch.softmax(scores, -1)
        out.append((probs @ v).movedim(1, 0))
    return torch.cat(out)


def _run(q_lens, seq_lens, backend, sinks, window_left, monkeypatch):
    from vllm_qaic import _custom_ops as qaic_ops

    monkeypatch.setenv("QAIC_PAGED_ATTN_BACKEND", backend)
    dev = "qaic:0"
    max_blocks = max(-(-s // B) for s in seq_lens)
    block_table = torch.randperm(NUM_BLOCKS)[: len(seq_lens) * max_blocks]
    block_table = block_table.view(len(seq_lens), max_blocks).to(torch.int32)

    key_cache, value_cache = _lbnhc_cache(dev)
    k_full = [torch.randn(s, HKV, D, dtype=torch.float16) for s in seq_lens]
    v_full = [torch.randn(s, HKV, D, dtype=torch.float16) for s in seq_lens]

    # Write the cached prefix of every request with the store path, then let
    # the attention call write the new tokens itself.
    def slots(req, positions):
        return [int(block_table[req, p // B]) * B + p % B for p in positions]

    prefix_slots, prefix_k, prefix_v = [], [], []
    new_slots, new_k, new_v = [], [], []
    for req, (q_len, s) in enumerate(zip(q_lens, seq_lens, strict=True)):
        prefix_slots += slots(req, range(s - q_len))
        prefix_k.append(k_full[req][: s - q_len])
        prefix_v.append(v_full[req][: s - q_len])
        new_slots += slots(req, range(s - q_len, s))
        new_k.append(k_full[req][s - q_len :])
        new_v.append(v_full[req][s - q_len :])
    kc_cpu = key_cache.cpu()
    vc_cpu = value_cache.cpu()
    for slot, k, v in zip(
        prefix_slots, torch.cat(prefix_k), torch.cat(prefix_v), strict=True
    ):
        kc_cpu[slot // B, :, slot % B] = k
        vc_cpu[slot // B, :, slot % B] = v
    key_cache.copy_(kc_cpu)
    value_cache.copy_(vc_cpu)

    num_tokens = sum(q_lens)
    q = torch.randn(num_tokens, H, D, dtype=torch.float16)
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    scale = D**-0.5
    out = qaic_ops.paged_attention(
        q.to(dev),
        torch.cat(new_k).to(dev),
        torch.cat(new_v).to(dev),
        key_cache,
        value_cache,
        block_table.to(dev),
        torch.tensor(new_slots, dtype=torch.int32).to(dev),
        qsl.to(dev),
        torch.tensor(seq_lens, dtype=torch.int32).to(dev),
        scale,
        True,
        write_cache=True,
        sinks=None if sinks is None else sinks.to(dev),
        window_left=window_left,
    ).cpu()

    expected = _ref(q, k_full, v_full, q_lens, seq_lens, scale, sinks, window_left)
    return out, expected, key_cache, value_cache, k_full, v_full, block_table


# gpt-oss stores sinks ~N(0, 1)-ish; the large case makes the sink dominate
# every row, which exercises the rebased denominator.
_SINKS = {
    "none": None,
    "normal": lambda: torch.randn(H),
    "dominant": lambda: torch.full((H,), 12.0),
}


@requires_qaic
@pytest.mark.parametrize("backend", ["hmx", "hvx"])
@pytest.mark.parametrize("sinks", list(_SINKS))
@pytest.mark.parametrize("window_left", [-1, 0, 19, 127])
@pytest.mark.parametrize(
    "q_lens, seq_lens",
    [
        ([1, 1, 1], [5, 50, 200]),  # decode
        ([37, 70], [37, 70]),  # prefill
        ([1, 20, 9], [50, 70, 9]),  # mixed decode + chunked prefill
    ],
    ids=["decode", "prefill", "mixed"],
)
def test_paged_attention_sinks_window(
    monkeypatch, backend, sinks, window_left, q_lens, seq_lens
):
    torch.manual_seed(0)
    sinks_t = None if _SINKS[sinks] is None else _SINKS[sinks]()
    out, expected, *_ = _run(
        q_lens, seq_lens, backend, sinks_t, window_left, monkeypatch
    )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@requires_qaic
def test_paged_attention_writes_lbnhc_cache_in_place(monkeypatch):
    """New tokens land in the interleaved K/V halves of the right page rows."""
    torch.manual_seed(0)
    q_lens, seq_lens = [3, 1], [3, 20]
    _, _, key_cache, value_cache, k_full, v_full, block_table = _run(
        q_lens, seq_lens, "hmx", torch.randn(H), 19, monkeypatch
    )
    kc, vc = key_cache.cpu(), value_cache.cpu()
    for req, s in enumerate(seq_lens):
        for p in range(s):
            blk = int(block_table[req, p // B])
            torch.testing.assert_close(
                kc[blk, :, p % B], k_full[req][p], atol=0, rtol=0
            )
            torch.testing.assert_close(
                vc[blk, :, p % B], v_full[req][p], atol=0, rtol=0
            )
