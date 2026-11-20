# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
from dataclasses import dataclass

import pytest
import torch

from tests.unit.ops.moe_test_utils import requires_qaic
from tests.unit.ops.paged_attention_test_utils import lbnhc_kv_cache, ref_attention
from vllm_qaic.attention.backends.qaic_attn import (
    QAicAttentionBackendImpl,
    QAicAttentionMetadata,
)


def _metadata(
    request_ids: list[str], seq_lens: list[int], query_lens: list[int]
) -> QAicAttentionMetadata:
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lens, dtype=torch.int32).cumsum(0).tolist()],
        dtype=torch.int32,
    )
    return QAicAttentionMetadata(
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        query_start_loc=query_start_loc,
        max_seq_len=max(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.empty(0, dtype=torch.int32),
        slot_mapping=torch.empty(0, dtype=torch.int32),
        slot_mapping_i32=torch.empty(0, dtype=torch.int32),
        max_num_seqs=2,
        max_model_len=16,
        scheduler_metadata=None,
        req_ids=request_ids,
    )


def _run_decode(
    attention: QAicAttentionBackendImpl,
    request_ids: list[str],
    seq_lens: list[int],
    query_lens: list[int],
    key_values: list[float],
) -> torch.Tensor:
    num_tokens = sum(query_lens)
    query = torch.zeros(num_tokens, 1, 2)
    key = torch.tensor(key_values, dtype=torch.float32).view(num_tokens, 1, 1)
    key = key.repeat(1, 1, 2)
    value = key.clone()
    output = torch.empty_like(query)
    attention._run_sdpa_decode_forward(
        query,
        key,
        value,
        _metadata(request_ids, seq_lens, query_lens),
        output,
    )
    return output


def _attention() -> QAicAttentionBackendImpl:
    return QAicAttentionBackendImpl(
        num_heads=1,
        head_size=2,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def test_speculative_rejection_overwrites_and_truncates_request_kv_cache():
    attention = _attention()

    _run_decode(
        attention,
        ["request-a", "request-b"],
        [3, 3],
        [3, 3],
        [1, 2, 3, 10, 11, 12],
    )
    _run_decode(
        attention,
        ["request-a", "request-b"],
        [6, 6],
        [3, 3],
        [4, 5, 6, 13, 14, 15],
    )
    output = _run_decode(
        attention, ["request-b", "request-a"], [4, 4], [1, 1], [20, 30]
    )

    a_key, _, a_cached = attention._kv_cache["request-a"]
    b_key, _, b_cached = attention._kv_cache["request-b"]
    assert a_cached == b_cached == 4
    assert a_key[:a_cached, 0, 0].tolist() == [1, 2, 3, 30]
    assert b_key[:b_cached, 0, 0].tolist() == [10, 11, 12, 20]

    # With a zero query, attention is the mean of V over the visible prefix
    # kv[:seq_len]; the rejected tail (4-6 / 13-15) must not contribute.
    assert output[:, 0, 0].tolist() == pytest.approx([13.25, 9.0])


def test_non_contiguous_kv_update_raises():
    attention = _attention()
    _run_decode(attention, ["request-a"], [3], [3], [1, 2, 3])

    with pytest.raises(ValueError, match="Non-contiguous KV update"):
        _run_decode(attention, ["request-a"], [6], [1], [4])


# ---------------------------------------------------------------------------
# Native paged-attention path (QAIC hardware).
#
# The tests above cover the SDPA fallback on CPU. The ones below drive
# QAicAttentionBackendImpl.forward() on a QAIC device with fp16 tensors, so it
# takes the native qaic::paged_attention kernel, and replay a speculative
# decoding rejection: verify 1+k rows, accept only `a` drafts, then verify
# again from position P+a+1, overwriting the rejected (stale) tail in place.
# ---------------------------------------------------------------------------

H, HKV, D, BLOCK = 8, 2, 64, 16
NUM_BLOCKS = 64
SCALE = D**-0.5
HUGE = 1e4  # fp16-representable; any read of a stale/unused row blows up
DEV = "qaic:0"


@dataclass(frozen=True)
class _Req:
    prompt_len: int
    num_drafts: int  # k; 0 is a plain single-token decode request
    num_accepted: int  # a <= k


def _qaic_impl(monkeypatch, backend):
    for var in (
        "QAIC_NATIVE_PAGED_ATTN",
        "QAIC_PAGED_ATTN_HMX",
        "QAIC_SPLIT_MIXED_PAGED_ATTN",
        "QAIC_SPLIT_MIXED_MIN_DECODE_REQS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("QAIC_PAGED_ATTN_BACKEND", backend)
    impl = QAicAttentionBackendImpl(
        num_heads=H,
        head_size=D,
        scale=SCALE,
        num_kv_heads=HKV,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    def _no_sdpa(*args, **kwargs):
        raise AssertionError("SDPA fallback taken instead of native paged attention")

    monkeypatch.setattr(impl, "_run_sdpa_decode_forward", _no_sdpa)

    from vllm_qaic import _custom_ops as qaic_ops

    # Kernel launches per forward(): 1 = single native call, 2 = split mixed
    # (decode rows and prefill/verify rows in separate calls).
    launches: list[int] = []
    kernel = qaic_ops.paged_attention

    def _count(*args, **kwargs):
        launches[-1] += 1
        return kernel(*args, **kwargs)

    native = impl._run_native_paged_attention

    def _record(*args, **kwargs):
        launches.append(0)
        return native(*args, **kwargs)

    monkeypatch.setattr(qaic_ops, "paged_attention", _count)
    monkeypatch.setattr(impl, "_run_native_paged_attention", _record)
    return impl, launches


def _forward(impl, kv_cache, block_table, starts, new_k, new_v, num_decode_tokens):
    """One forward() step; request i writes/queries positions starts[i]...

    Returns (output on CPU, query on CPU, seq_lens).
    """
    q_lens = [k.shape[0] for k in new_k]
    seq_lens = [s + q for s, q in zip(starts, q_lens, strict=True)]
    slots = [
        int(block_table[r, p // BLOCK]) * BLOCK + p % BLOCK
        for r, (s, e) in enumerate(zip(starts, seq_lens, strict=True))
        for p in range(s, e)
    ]
    num_tokens = sum(q_lens)
    q = torch.randn(num_tokens, H, D, dtype=torch.float16)
    qsl = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEV)
    metadata = QAicAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=max(q_lens),
        query_start_loc=qsl.to(DEV),
        max_seq_len=max(seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        block_table=block_table.to(DEV),
        slot_mapping=slot_mapping,
        slot_mapping_i32=slot_mapping.to(torch.int32),
        max_num_seqs=len(q_lens),
        max_model_len=block_table.shape[1] * BLOCK,
        scheduler_metadata=None,
        causal=True,
        use_sdpa_prefill=True,
        num_decode_tokens=num_decode_tokens,
        sdpa_start_loc=qsl.to(DEV),
        req_ids=[f"req-{i}" for i in range(len(q_lens))],
    )
    output = torch.empty(num_tokens, H, D, dtype=torch.float16, device=DEV)
    impl.forward(
        None,
        q.to(DEV),
        torch.cat(new_k).to(DEV),
        torch.cat(new_v).to(DEV),
        kv_cache,
        metadata,
        output=output,
    )
    return output.cpu(), q, seq_lens


def _kv(n):
    return torch.randn(n, HKV, D, dtype=torch.float16)


def _rows(q_lens, keep):
    """Flat token mask selecting the first keep[i] rows of request i."""
    return torch.cat([torch.arange(q) < n for q, n in zip(q_lens, keep, strict=True)])


def _run_rejection_scenario(monkeypatch, backend, stale, reqs, expected_launches):
    """prefill P -> verify P..P+k -> accept a -> verify from P+a+1.

    Decode requests (k == 0) must come first, as the builder reorders them.
    With stale == "huge" the rejected drafts and every never-written cache row
    hold HUGE, so any read outside the committed history corrupts the output.
    """
    assert all(r.num_drafts == 0 for r in reqs[: sum(r.num_drafts == 0 for r in reqs)])
    torch.manual_seed(0)
    impl, launches = _qaic_impl(monkeypatch, backend)
    fill = HUGE if stale == "huge" else 0.0
    kv_cache = lbnhc_kv_cache(NUM_BLOCKS, BLOCK, HKV, D, DEV, fill=fill)
    final_lens = [r.prompt_len + r.num_accepted + 2 + r.num_drafts for r in reqs]
    max_blocks = -(-max(final_lens) // BLOCK)
    assert len(reqs) * max_blocks <= NUM_BLOCKS
    block_table = torch.randperm(NUM_BLOCKS)[: len(reqs) * max_blocks]
    block_table = block_table.view(len(reqs), max_blocks).to(torch.int32)
    num_decode = sum(r.num_drafts == 0 for r in reqs)
    tol = dict(atol=2e-2, rtol=2e-2)

    def check(out, q, hist_k, hist_v, q_lens, seq_lens, rows=None):
        expected = ref_attention(q, hist_k, hist_v, q_lens, seq_lens, SCALE, None, -1)
        if rows is not None:
            out, expected = out[rows], expected[rows]
        torch.testing.assert_close(out.float(), expected, **tol)

    # Step 1: prefill P tokens per request.
    new_k = [_kv(r.prompt_len) for r in reqs]
    new_v = [_kv(r.prompt_len) for r in reqs]
    out, q, seq_lens = _forward(
        impl, kv_cache, block_table, [0] * len(reqs), new_k, new_v, 0
    )
    check(out, q, new_k, new_v, [r.prompt_len for r in reqs], seq_lens)
    hist_k, hist_v = list(new_k), list(new_v)

    # Step 2: verify 1+k rows at P..P+k. Rejected drafts are the stale tail.
    q_lens = [1 + r.num_drafts for r in reqs]
    new_k, new_v = [_kv(n) for n in q_lens], [_kv(n) for n in q_lens]
    if stale == "huge":
        for r, k, v in zip(reqs, new_k, new_v, strict=True):
            k[r.num_accepted + 1 :] = HUGE
            v[r.num_accepted + 1 :] = HUGE
    starts = [r.prompt_len for r in reqs]
    out, q, seq_lens = _forward(
        impl, kv_cache, block_table, starts, new_k, new_v, num_decode
    )
    step_k = [torch.cat(p) for p in zip(hist_k, new_k, strict=True)]
    step_v = [torch.cat(p) for p in zip(hist_v, new_v, strict=True)]
    # Causality: rows 0..a never see the rejected drafts after them. With HUGE
    # drafts the later rows are HUGE-dominated, so only those rows are checked.
    keep = [r.num_accepted + 1 for r in reqs] if stale == "huge" else q_lens
    check(out, q, step_k, step_v, q_lens, seq_lens, _rows(q_lens, keep))
    # Commit position P (the last sampled token) plus the `a` accepted drafts.
    for i, r in enumerate(reqs):
        hist_k[i] = torch.cat([hist_k[i], new_k[i][: r.num_accepted + 1]])
        hist_v[i] = torch.cat([hist_v[i], new_v[i][: r.num_accepted + 1]])

    # Step 3: only `a` drafts accepted -> next verify (bonus + k drafts)
    # starts at P+a+1 and overwrites the stale tail with fresh K/V.
    starts = [r.prompt_len + r.num_accepted + 1 for r in reqs]
    assert [h.shape[0] for h in hist_k] == starts
    new_k, new_v = [_kv(n) for n in q_lens], [_kv(n) for n in q_lens]
    out, q, seq_lens = _forward(
        impl, kv_cache, block_table, starts, new_k, new_v, num_decode
    )
    assert seq_lens == final_lens
    hist_k = [torch.cat(p) for p in zip(hist_k, new_k, strict=True)]
    hist_v = [torch.cat(p) for p in zip(hist_v, new_v, strict=True)]
    check(out, q, hist_k, hist_v, q_lens, seq_lens)

    # The cache holds exactly the committed history, including the fresh
    # values at the positions that held rejected drafts after step 2.
    key_cache, value_cache = (c.cpu() for c in kv_cache.split(D, dim=-1))
    for req, r in enumerate(reqs):
        overwritten = range(starts[req], r.prompt_len + r.num_drafts + 1)
        for p in range(seq_lens[req]):
            blk = int(block_table[req, p // BLOCK])
            torch.testing.assert_close(
                key_cache[blk, :, p % BLOCK], hist_k[req][p], atol=0, rtol=0
            )
            torch.testing.assert_close(
                value_cache[blk, :, p % BLOCK], hist_v[req][p], atol=0, rtol=0
            )
            if stale == "huge" and p in overwritten:
                assert key_cache[blk, :, p % BLOCK].abs().max() < HUGE

    assert launches == expected_launches


@requires_qaic
@pytest.mark.parametrize("backend", ["hmx", "hvx"])
@pytest.mark.parametrize("stale", ["random", "huge"])
def test_native_paged_attention_spec_rejection_overwrites_stale_tail(
    monkeypatch, backend, stale
):
    """All rows are verify rows (query_len 1+k > 1), which count as prefill."""
    reqs = [
        _Req(prompt_len=30, num_drafts=3, num_accepted=1),  # rewrite crosses block
        _Req(prompt_len=47, num_drafts=3, num_accepted=0),
        _Req(prompt_len=70, num_drafts=4, num_accepted=2),
        _Req(prompt_len=40, num_drafts=2, num_accepted=2),  # all accepted
    ]
    _run_rejection_scenario(monkeypatch, backend, stale, reqs, [1] * 3)


@requires_qaic
@pytest.mark.parametrize("backend", ["hmx", "hvx"])
@pytest.mark.parametrize("stale", ["random", "huge"])
def test_native_paged_attention_mixed_decode_and_verify(monkeypatch, backend, stale):
    """4 single-token decodes + 2 verify requests in one forward; on hmx this
    meets QAIC_SPLIT_MIXED_MIN_DECODE_REQS (4) and takes the split path."""
    reqs = [
        _Req(prompt_len=17, num_drafts=0, num_accepted=0),
        _Req(prompt_len=35, num_drafts=0, num_accepted=0),
        _Req(prompt_len=64, num_drafts=0, num_accepted=0),
        _Req(prompt_len=90, num_drafts=0, num_accepted=0),
        _Req(prompt_len=50, num_drafts=3, num_accepted=1),
        _Req(prompt_len=81, num_drafts=5, num_accepted=0),
    ]
    mixed = 2 if backend == "hmx" else 1
    _run_rejection_scenario(monkeypatch, backend, stale, reqs, [1, mixed, mixed])
