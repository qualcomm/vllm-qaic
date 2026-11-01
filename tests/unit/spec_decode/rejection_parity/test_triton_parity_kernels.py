# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Randomized equivalence tests: Numba kernels vs upstream Triton-CPU kernels.

Complements ``test_triton_parity_prod.py`` with int32 draft ids, a 7-row
ragged batch with empty requests, draft ids at row edges, and odd vocab sizes
(1, and not multiples of the 32-lane block or the 8192 Triton tile).

Tier A (``triton_parity``): skipped without triton-cpu.

    TRITON_CPU_BACKEND=1 .venv_aot/bin/python -m pytest -s -q \
        tests/unit/spec_decode/rejection_parity/test_triton_parity_kernels.py
"""

import os

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import _parity_common as common  # noqa: E402  (captures upstream kernels first)
import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.v1.sample import rejection_sampler as rs  # noqa: E402

from vllm_qaic.v1.sample import numba_rejection_kernels as nrk  # noqa: E402

pytestmark = pytest.mark.triton_parity


@pytest.fixture(autouse=True)
def _require_triton_kernels():
    common.assert_triton_originals()


SEEDS = range(2)
VOCABS = (1, 33, 8193, 20011)


def _batch(seed, vocab, max_len=6, batch=7):
    g = torch.Generator().manual_seed(seed)
    counts = torch.randint(0, max_len + 1, (batch,), generator=g)
    counts[0] = 0  # empty first request exercises start-offset logic
    cu = torch.cumsum(counts, 0).to(torch.int32)
    total = int(cu[-1])
    draft = torch.randint(0, vocab, (total,), generator=g, dtype=torch.int32)
    if total:
        draft[0], draft[-1] = 0, vocab - 1  # draft column at both row edges
    return g, cu, draft, total, batch


def _probs(g, total, vocab, ties):
    p = torch.rand(total, vocab, generator=g)
    if ties:
        p = (p * 4).floor() / 4  # many exact duplicates
    return (p / p.sum(-1, keepdim=True).clamp_min(1e-9)).contiguous()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("vocab", VOCABS)
@pytest.mark.parametrize("no_draft", (True, False))
@pytest.mark.parametrize("fp64", (False, True))
@pytest.mark.parametrize("ties", (False, True))
def test_recovered(seed, vocab, no_draft, fp64, ties):
    g, cu, draft, total, batch = _batch(seed, vocab)
    target = _probs(g, total, vocab, ties)
    dprobs = None if no_draft else _probs(g, total, vocab, ties)
    q = torch.empty(
        batch, vocab, dtype=torch.float64 if fp64 else torch.float32
    ).exponential_(generator=g)
    if ties:
        # Integer q values create exact score ties.  Keep q >= 1: q == 0 gives
        # inv_q == inf and 0 * inf == NaN, on which upstream Triton itself
        # returns an out-of-vocab id (e.g. 1 for vocab_size == 1).
        q = q.round().clamp_min(1)
    inv_q = q.reciprocal()
    ref = torch.full((total,), -1, dtype=torch.int32)
    out = ref.clone()
    rs.sample_recovered_tokens_kernel[(batch, 6)](
        ref,
        cu,
        draft,
        dprobs,
        target,
        inv_q,
        vocab,
        BLOCK_SIZE=8192,
        NO_DRAFT_PROBS=no_draft,
        USE_FP64_GUMBEL=fp64,
    )
    nrk.recovered_numba(out, cu, draft, dprobs, target, inv_q, vocab, no_draft)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("vocab", (31, 1000))
@pytest.mark.parametrize("no_draft", (True, False))
@pytest.mark.parametrize("synthetic", (False, True))
def test_greedy_and_random(seed, vocab, no_draft, synthetic):
    max_len = 6
    g, cu, draft, total, batch = _batch(seed, vocab, max_len)
    # Mix accepted/rejected positions: argmax equals draft ~60% of the time.
    argmax = torch.where(
        torch.rand(total, generator=g) < 0.6,
        draft.long(),
        torch.randint(0, vocab, (total,), generator=g),
    )
    bonus = torch.randint(0, vocab, (batch, 1), generator=g, dtype=torch.int32)
    is_greedy = torch.rand(batch, generator=g) < 0.5
    uniform = torch.rand(total, generator=g, dtype=torch.float64)
    rates = torch.rand(max_len, generator=g)
    target = _probs(g, total, vocab, False)
    dprobs = None if no_draft else _probs(g, total, vocab, False)
    if dprobs is not None and total:
        dprobs[0, draft[0]] = 0.0  # draft_prob == 0 must reject
    recovered = torch.randint(0, vocab, (total,), generator=g, dtype=torch.int32)

    for mask in (None, is_greedy):
        ref = torch.full((batch, max_len + 1), -1, dtype=torch.int32)
        out = ref.clone()
        rs.rejection_greedy_sample_kernel[(batch,)](
            ref,
            cu,
            draft,
            argmax,
            bonus,
            mask,
            max_len,
            uniform,
            rates,
            SYNTHETIC_MODE=synthetic,
        )
        nrk.greedy_numba(
            out,
            cu,
            draft,
            argmax,
            bonus,
            mask,
            max_len,
            uniform,
            rates,
            synthetic,
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    ref = torch.full((batch, max_len + 1), -1, dtype=torch.int32)
    out = ref.clone()
    rs.rejection_random_sample_kernel[(batch,)](
        ref,
        cu,
        draft,
        dprobs,
        target,
        bonus,
        recovered,
        uniform,
        is_greedy,
        max_len,
        vocab,
        rates,
        NO_DRAFT_PROBS=no_draft,
        SYNTHETIC_MODE=synthetic,
    )
    nrk.random_numba(
        out,
        cu,
        draft,
        dprobs,
        target,
        bonus,
        recovered,
        uniform,
        is_greedy,
        max_len,
        vocab,
        rates,
        no_draft,
        synthetic,
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("seed", SEEDS)
def test_expand(seed):
    g = torch.Generator().manual_seed(seed)
    batch = 9
    counts = torch.randint(0, 5, (batch,), generator=g)
    cu = torch.cumsum(counts, 0).to(torch.int32)
    x = torch.randint(0, 4, (batch,), generator=g, dtype=torch.int32)
    ref = torch.full((int(cu[-1]),), -1, dtype=torch.int32)
    out = ref.clone()
    rs.expand_kernel[(batch,)](ref, x, cu, 2, 99, MAX_NUM_TOKENS=8)
    nrk.expand_numba(out, x, cu, 2, 99)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
