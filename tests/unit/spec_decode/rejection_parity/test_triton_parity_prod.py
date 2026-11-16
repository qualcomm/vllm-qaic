# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Bit-exact Numba-vs-Triton tests on AOT *production* dtypes and call layouts.

Complements ``test_triton_parity_kernels.py`` (int32 draft ids, int32 expand
input).  Every case builds the positional args exactly as the upstream call
sites in ``vllm/v1/sample/rejection_sampler.py`` (v0.23.0) do, with the AOT
dtypes (draft int64, cu int32, bonus int32 [B,1], argmax int64, output int32
[B,K+1], recovered int64, uniform fp64, target/draft probs fp32, inv_q fp32 or
fp64, temperature/top_p fp32, top_k int32), launches the real upstream Triton
``JITFunction`` with ``kernel[grid](*args, **kwargs)``, and compares with
``torch.equal`` against the wrappers that
``vllm_qaic.v1.sample.rejection_sampler_numba.install()`` installs (``prod``).
Add ``"raw"`` to ``IMPLS`` to also run the bare
``vllm_qaic.v1.sample.numba_rejection_kernels`` functions through an
upstream-layout adapter, e.g. to tell a wrapper bug from a kernel bug.

Tier A (``triton_parity``): skipped without triton-cpu.

    TRITON_CPU_BACKEND=1 .venv_aot/bin/python -m pytest -s -q \
        tests/unit/spec_decode/rejection_parity/test_triton_parity_prod.py
    RS_PROD_HEAVY=0 ...   # skip the 128x16 x V=128256 recovered case (~1 GB probs)
"""

from __future__ import annotations

import os

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import _parity_common as common  # noqa: E402  (captures upstream kernels first)
import pytest  # noqa: E402
import torch  # noqa: E402
from vllm.v1.sample import rejection_sampler as rs  # noqa: E402

pytestmark = pytest.mark.triton_parity

PROD_VOCAB = 128256
HEAVY = os.environ.get("RS_PROD_HEAVY", "1") != "0"

_WRAPPERS = common.production_wrappers()
IMPLS = ("prod",)


@pytest.fixture(autouse=True)
def _require_triton_kernels():
    common.assert_triton_originals()


def _launch(impl: str, name: str, grid, args, kwargs) -> None:
    if impl == "raw":
        common.launch_raw(name, grid, args, kwargs)
    else:
        if _WRAPPERS is None:
            pytest.skip(f"{common.PROD_MODULE} not importable yet")
        common.launch_wrapper(_WRAPPERS, name, grid, args, kwargs)


KNOWN_TRITON_SYNTHETIC_INT64 = common.KNOWN_TRITON_SYNTHETIC_INT64


def triton_synthetic_int64_broken(name, args, kwargs) -> bool:
    """Upstream greedy kernel + SYNTHETIC_MODE + int64 draft ids fails to compile.

    ``token_id = draft_token_id if accepted else target_argmax_id`` mixes the
    int64 draft id with the ``.to(tl.int32)`` argmax; triton-cpu rejects the
    dynamic ternary.  AOT's draft ids are int64 (model_runner input_ids_cpu),
    so the Triton baseline cannot run synthetic mode with any greedy request.
    """
    return (
        name == "rejection_greedy_sample_kernel"
        and kwargs.get("SYNTHETIC_MODE")
        and args[2].dtype == torch.int64
    )


def _check(impl: str, name: str, grid, args, kwargs) -> torch.Tensor:
    """Run Triton and ``impl`` on independent clones; assert bit equality."""
    t_args, n_args = common.clone_args(args), common.clone_args(args)
    if triton_synthetic_int64_broken(name, args, kwargs):
        with pytest.raises(Exception, match=KNOWN_TRITON_SYNTHETIC_INT64):
            common.launch_triton(name, grid, common.clone_args(args), kwargs)
        # Reference: Triton on the int32-draft equivalent (ids < 2^31 => same values).
        t_args[2] = t_args[2].to(torch.int32)
    common.launch_triton(name, grid, t_args, kwargs)
    _launch(impl, name, grid, n_args, kwargs)
    ref, out = t_args[0], n_args[0]
    assert ref.dtype == out.dtype and ref.shape == out.shape
    if not torch.equal(ref, out):
        diff = (ref != out).nonzero()
        pytest.fail(
            f"{name}[{grid}] {impl} mismatch at {diff[:8].tolist()} "
            f"(n={diff.shape[0]}): triton={ref.flatten()[:16].tolist()} "
            f"numba={out.flatten()[:16].tolist()}"
        )
    # Inputs must not be mutated differently either (only arg 0 is an output).
    for i, (a, b) in enumerate(zip(t_args[1:], n_args[1:], strict=True), 1):
        if isinstance(a, torch.Tensor) and a.dtype == b.dtype:
            assert torch.equal(a, b) or (
                a.is_floating_point() and torch.equal(a.isnan(), b.isnan())
            ), i
    return ref


# --------------------------------------------------------------------------
# Production-shaped inputs
# --------------------------------------------------------------------------
def _counts(g, batch, k, pattern):
    if pattern == "full":
        c = torch.full((batch,), k, dtype=torch.int64)
    else:  # ragged: includes 0 (no-draft requests) and k
        c = torch.randint(0, k + 1, (batch,), generator=g)
        c[0] = 0
        c[-1] = k
    return c


def _probs(g, n, vocab, mode):
    if mode == "peaked":  # softmax of logits, like target_logits.softmax(fp32)
        logits = torch.randn(n, vocab, generator=g) * 4
        p = logits.softmax(-1, dtype=torch.float32)
    else:
        p = torch.rand(n, vocab, generator=g)
        if mode == "ties":
            p = (p * 4).floor() / 4  # many exact duplicates, many exact zeros
        elif mode == "zeros":
            p = p * (torch.rand(n, vocab, generator=g) < 0.02)  # 98% exact 0.0
            if n:
                p[0].zero_()  # an all-zero row
        p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    return p.contiguous()


class Case:
    def __init__(
        self,
        seed,
        batch,
        k,
        vocab,
        *,
        pattern="ragged",
        probs="peaked",
        accept=0.6,
        fp64=False,
        q_mode="exp",
    ):
        g = torch.Generator().manual_seed(seed)
        self.g, self.batch, self.k, self.vocab = g, batch, k, vocab
        counts = _counts(g, batch, k, pattern)
        self.counts = counts.tolist()
        self.cu = torch.cumsum(counts, 0).to(torch.int32)
        self.n = n = int(self.cu[-1])
        self.target = _probs(g, n, vocab, probs)
        argmax = self.target.argmax(-1)  # int64, first index on ties
        # Drafts equal argmax with prob `accept`; else random, incl. row edges.
        rnd = torch.randint(0, vocab, (n,), generator=g)
        if n:
            rnd[0], rnd[-1] = 0, vocab - 1
        self.draft = torch.where(
            torch.rand(n, generator=g) < accept, argmax, rnd
        ).contiguous()
        self.argmax = argmax
        self.bonus = torch.randint(0, vocab, (batch, 1), generator=g, dtype=torch.int32)
        self.uniform = torch.rand(n, generator=g, dtype=torch.float64)
        self.fp64 = fp64
        q = torch.empty(batch, vocab, dtype=torch.float64 if fp64 else torch.float32)
        q.exponential_(generator=g)
        if q_mode == "ties":
            q = q.round().clamp_min(1)  # integer q -> exact score ties, finite inv_q
        elif q_mode == "large_inv":
            # Tiny q -> inv_q up to ~1e30 (fp32) / 1e300 (fp64), still finite.
            tiny = 1e-30 if not fp64 else 1e-300
            mask = torch.rand(batch, vocab, generator=g) < 0.01
            q = torch.where(mask, q * tiny, q)
        self.inv_q = q.reciprocal()
        self.dprobs = _probs(g, n, vocab, "peaked")
        self.is_greedy = torch.rand(batch, generator=g) < 0.5
        self.rates = torch.rand(k, generator=g).to(torch.float32)

    def out(self):
        return torch.full((self.batch, self.k + 1), -1, dtype=torch.int32)

    def recovered_args(self, no_draft):
        rec = torch.full_like(self.draft, -7)  # empty_like(draft) -> int64
        args = [
            rec,
            self.cu,
            self.draft,
            None if no_draft else self.dprobs,
            self.target,
            self.inv_q,
            self.vocab,
            8192,
        ]
        kwargs = dict(NO_DRAFT_PROBS=no_draft, USE_FP64_GUMBEL=self.fp64)
        return (self.batch, self.k), args, kwargs

    def recovered(self, no_draft):
        grid, args, kwargs = self.recovered_args(no_draft)
        common.launch_triton("sample_recovered_tokens_kernel", grid, args, kwargs)
        return args[0]

    def greedy_args(self, mask, synthetic):
        uniform = self.uniform if (synthetic or mask is not None) else None
        args = [
            self.out(),
            self.cu,
            self.draft,
            self.argmax,
            self.bonus,
            mask,
            self.k,
            uniform,
            self.rates if synthetic else None,
        ]
        return (self.batch,), args, dict(SYNTHETIC_MODE=synthetic)

    def random_args(self, no_draft, synthetic, recovered):
        args = [
            self.out(),
            self.cu,
            self.draft,
            None if no_draft else self.dprobs,
            self.target,
            self.bonus,
            recovered,
            self.uniform,
            self.is_greedy,
            self.k,
            self.vocab,
            self.rates if synthetic else None,
        ]
        return (
            (self.batch,),
            args,
            dict(NO_DRAFT_PROBS=no_draft, SYNTHETIC_MODE=synthetic),
        )


SHAPES = [  # (batch, k, vocab)
    (1, 1, 31),
    (8, 4, 1000),
    (8, 4, PROD_VOCAB),
    (128, 16, 1000),
    (128, 4, 8193),
]


# --------------------------------------------------------------------------
# expand: apply_sampling_constraints / expand_batch_to_tokens call sites
# --------------------------------------------------------------------------
EXPAND_PARAMS = {
    # name: (dtype, values pool, replace_from, replace_to) per call site
    "temperature": (
        torch.float32,
        [0.0, 0.7, 1.0, 1e-5, 2.0, 0.3333333],
        rs.GREEDY_TEMPERATURE,
        1,
    ),
    "top_k": (torch.int32, [0, 1, 50, PROD_VOCAB, 2**31 - 1], 0, 0),
    "top_p": (torch.float32, [1.0, 0.9, 0.0, 1e-7, 0.95], 0, 0),
}


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("param", sorted(EXPAND_PARAMS))
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("batch,k", [(1, 1), (8, 4), (128, 16)])
def test_expand_prod(impl, param, seed, batch, k):
    dtype, pool, replace_from, replace_to = EXPAND_PARAMS[param]
    g = torch.Generator().manual_seed(seed)
    counts = _counts(g, batch, k, "ragged" if seed % 2 else "full")
    cu = torch.cumsum(counts, 0).to(torch.int32)
    pool_t = torch.tensor(pool, dtype=dtype)
    x = pool_t[torch.randint(0, len(pool), (batch,), generator=g)].contiguous()
    n = int(cu[-1])
    out = x.new_full((n,), -3)  # x.new_empty(num_tokens), sentinel-filled
    args = [out, x, cu, replace_from, replace_to]
    ref = _check(
        impl, "expand_kernel", (batch,), args, dict(MAX_NUM_TOKENS=rs.MAX_SPEC_LEN)
    )
    if param == "temperature":
        assert not (ref == 0).any()  # 0 -> 1 replacement happened


def test_expand_through_upstream_helper():
    """End-to-end through the real upstream helper with the wrappers installed."""
    if _WRAPPERS is None:
        pytest.skip(f"{common.PROD_MODULE} not importable yet")
    temp = torch.tensor([0.0, 0.7, 0.0, 1.3], dtype=torch.float32)
    cu = torch.tensor([2, 2, 6, 7], dtype=torch.int32)
    ref = rs.expand_batch_to_tokens(temp, cu, 7, replace_from=0, replace_to=1)
    saved = {n: getattr(rs, n) for n in common.KERNEL_NAMES}
    try:
        rs.expand_kernel = _WRAPPERS["expand_kernel"]
        out = rs.expand_batch_to_tokens(temp, cu, 7, replace_from=0, replace_to=1)
    finally:
        for name, obj in saved.items():
            setattr(rs, name, obj)
    assert torch.equal(ref, out) and out.dtype == torch.float32


# --------------------------------------------------------------------------
# recovered
# --------------------------------------------------------------------------
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("no_draft", (True, False))
@pytest.mark.parametrize("fp64", (False, True))
@pytest.mark.parametrize(
    "probs,q_mode",
    [("peaked", "exp"), ("ties", "ties"), ("zeros", "exp"), ("peaked", "large_inv")],
)
@pytest.mark.parametrize("seed", range(2))
def test_recovered_prod(impl, shape, no_draft, fp64, probs, q_mode, seed):
    batch, k, vocab = shape
    if vocab == PROD_VOCAB and (seed or probs == "zeros"):
        pytest.skip("keep V=128256 count small")
    c = Case(
        seed,
        batch,
        k,
        vocab,
        probs=probs,
        fp64=fp64,
        q_mode=q_mode,
        pattern="ragged" if seed == 0 else "full",
    )
    _check(impl, "sample_recovered_tokens_kernel", *c.recovered_args(no_draft))


@pytest.mark.skipif(not HEAVY, reason="RS_PROD_HEAVY=0")
@pytest.mark.parametrize("impl", IMPLS)
def test_recovered_prod_heavy(impl):
    """128 x 16 = 2048 tokens at the production vocab (~1 GB fp32 target_probs)."""
    c = Case(11, 128, 16, PROD_VOCAB, pattern="full")
    _check(impl, "sample_recovered_tokens_kernel", *c.recovered_args(True))


def test_recovered_inf_inv_q_tie():
    """q == +0.0 -> inv_q == +inf: prob > 0 columns score +inf; first index wins."""
    c = Case(3, 4, 4, 1000, pattern="full")
    c.inv_q[:, [17, 400, 999]] = float(
        "inf"
    )  # all have target prob > 0 (peaked softmax)
    for impl in IMPLS:
        for no_draft in (False, True):
            ref = _check(
                impl, "sample_recovered_tokens_kernel", *c.recovered_args(no_draft)
            )
            # Without draft probs, +inf scores beat everything; ties -> first
            # index (unless it is the draft column).  With draft probs,
            # max(target - draft, 0) can zero those columns (0 * inf = NaN, see
            # test_recovered_nan_score_documented), so only parity holds.
            if no_draft:
                assert set(ref.tolist()) <= {17, 400}, ref.tolist()


def _nan_case(no_draft):
    """inv_q == inf multiplied by a zero prob -> NaN score (documented divergence)."""
    c = Case(5, 2, 2, 64, pattern="full")
    if no_draft:
        # Masked draft column has prob 0 -> 0 * inf = NaN.
        for req in range(2):
            for pos in range(2):
                tok = req * 2 + pos
                c.inv_q[req, int(c.draft[tok])] = float("inf")
    else:
        # Clamp target-draft to 0 on column 5 and put inf there.
        c.dprobs[:, 5] = c.target[:, 5] + 0.5
        c.inv_q[:, 5] = float("inf")
    return c


@pytest.mark.parametrize("no_draft", (True, False))
def test_recovered_nan_score_documented(no_draft):
    """Record (not hide) the NaN-score behavior of both backends.

    Upstream Triton: NaN poisons tl.max and the strict ``local_max > max_val``
    never fires (or fires with a NaN index), so the stored id can be an
    out-of-vocab value.  The Numba port's find-first returns -1 when no score
    equals the (NaN) max and clamps to 0 via ``max(idx, 0)``.  The production
    reachability is q == +0.0 from exponential_, i.e. a 53-bit uniform == 0
    (p = 2^-53 per element; see probe_exponential_zero.py).
    """
    c = _nan_case(no_draft)
    grid, args, kwargs = c.recovered_args(no_draft)
    t_args = common.clone_args(args)
    common.launch_triton("sample_recovered_tokens_kernel", grid, t_args, kwargs)
    results = {"triton": t_args[0].tolist()}
    for impl in IMPLS:
        if impl == "prod" and _WRAPPERS is None:
            continue
        n_args = common.clone_args(args)
        _launch(impl, "sample_recovered_tokens_kernel", grid, n_args, kwargs)
        results[impl] = n_args[0].tolist()
    print(f"\n[NaN-score no_draft={no_draft}] vocab={c.vocab} recovered ids: {results}")
    for impl, ids in results.items():
        if impl != "triton":
            assert all(0 <= i < c.vocab for i in ids), (
                impl,
                ids,
            )  # numba stays in-vocab


# --------------------------------------------------------------------------
# greedy
# --------------------------------------------------------------------------
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("mask_mode", ("none", "mixed"))
@pytest.mark.parametrize("synthetic", (False, True))
@pytest.mark.parametrize("seed", range(3))
def test_greedy_prod(impl, shape, mask_mode, synthetic, seed):
    batch, k, vocab = shape
    if vocab == PROD_VOCAB and seed:
        pytest.skip("keep V=128256 count small")
    c = Case(
        seed,
        batch,
        k,
        vocab,
        probs="ties" if seed == 2 else "peaked",
        pattern="ragged" if seed != 1 else "full",
    )
    if synthetic and c.n:
        c.uniform[0] = float(c.rates[0])  # uniform == rate -> reject (strict <)
    mask = None if mask_mode == "none" else c.is_greedy
    _check(impl, "rejection_greedy_sample_kernel", *c.greedy_args(mask, synthetic))


# --------------------------------------------------------------------------
# random
# --------------------------------------------------------------------------
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("no_draft", (True, False))
@pytest.mark.parametrize("synthetic", (False, True))
@pytest.mark.parametrize("greedy_mode", ("mixed", "all_random"))
@pytest.mark.parametrize("seed", range(2))
def test_random_prod(impl, shape, no_draft, synthetic, greedy_mode, seed):
    batch, k, vocab = shape
    if vocab == PROD_VOCAB and seed:
        pytest.skip("keep V=128256 count small")
    c = Case(seed, batch, k, vocab, pattern="ragged" if seed == 0 else "full")
    if greedy_mode == "all_random":
        c.is_greedy = torch.zeros(batch, dtype=torch.bool)
    n = c.n
    if n:
        # Adversarial: uniform exactly equal to the accept ratio (>= accepts),
        # uniform == rate (strict < rejects), draft_prob == 0 (must reject).
        idx = torch.arange(n)
        sel = idx[idx % 3 == 0]
        d = c.draft[sel]
        tp = c.target[sel, d]
        # target_prob / 1, or fp32 / fp32 as the kernel computes.
        ratio = tp if no_draft else tp / c.dprobs[sel, d]
        c.uniform[sel] = ratio.to(torch.float64)
        if not no_draft:
            c.dprobs[idx[idx % 5 == 1], c.draft[idx[idx % 5 == 1]]] = 0.0
        if synthetic:
            pos_in_req = idx - torch.repeat_interleave(
                torch.cat([torch.zeros(1, dtype=torch.int64), c.cu[:-1].long()]),
                torch.tensor(c.counts),
            )
            hit = idx[idx % 4 == 2]
            c.uniform[hit] = c.rates[pos_in_req[hit]].to(torch.float64)
    recovered = c.recovered(no_draft)  # int64, from the real Triton kernel
    _check(
        impl,
        "rejection_random_sample_kernel",
        *c.random_args(no_draft, synthetic, recovered),
    )


# --------------------------------------------------------------------------
# full upstream rejection_sample() with wrappers installed
# --------------------------------------------------------------------------
def _metadata(c: Case, mode: str, seeded: bool):
    from vllm.v1.sample.metadata import SamplingMetadata

    temp = torch.full((c.batch,), 0.7, dtype=torch.float32)
    if mode == "mixed":
        temp[c.is_greedy] = 0.0
    elif mode == "greedy":
        temp.zero_()
    gens = {}
    if seeded:
        gens = {i: torch.Generator().manual_seed(1000 + i) for i in range(c.batch)}
    return SamplingMetadata(
        temperature=temp,
        all_greedy=mode == "greedy",
        all_random=mode == "random",
        top_p=None,
        top_k=None,
        generators=gens,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(c.batch),
        presence_penalties=torch.zeros(c.batch),
        repetition_penalties=torch.ones(c.batch),
        output_token_ids=[[] for _ in range(c.batch)],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )


@pytest.mark.parametrize("mode", ("greedy", "mixed", "random"))
@pytest.mark.parametrize("synthetic", (False, True))
@pytest.mark.parametrize("fp64", (False, True))
@pytest.mark.parametrize(
    "shape", [(8, 4, 1000), (8, 4, PROD_VOCAB)], ids=lambda s: "x".join(map(str, s))
)
def test_rejection_sample_end_to_end(mode, synthetic, fp64, shape):
    """Upstream rejection_sample() with Triton vs with the installed Numba wrappers.

    Seeded generators make every random draw (uniform, exponential q) identical
    across the two runs, so the outputs must be bit-equal.
    """
    if _WRAPPERS is None:
        pytest.skip(f"{common.PROD_MODULE} not importable yet")
    batch, k, vocab = shape
    c = Case(7, batch, k, vocab, pattern="ragged")
    logits = torch.randn(c.n, vocab, generator=c.g) * 3
    rates = c.rates if synthetic else None

    def run(draft=c.draft):
        return rs.rejection_sample(
            draft,
            c.counts,
            k,
            c.cu,
            None,
            logits.clone(),
            c.bonus,
            _metadata(c, mode, seeded=True),
            synthetic_mode=synthetic,
            synthetic_conditional_rates=rates,
            use_fp64_gumbel=fp64,
        )

    saved = {n: getattr(rs, n) for n in common.KERNEL_NAMES}
    try:
        for n in common.KERNEL_NAMES:
            setattr(rs, n, common.ORIGINAL_KERNELS[n])
        if synthetic and mode != "random" and c.draft.dtype == torch.int64:
            # See triton_synthetic_int64_broken: upstream Triton cannot compile
            # this path with AOT's int64 draft ids.  Reference = Triton on the
            # int32-cast ids (same values; output buffer is int32 either way).
            with pytest.raises(Exception, match=KNOWN_TRITON_SYNTHETIC_INT64):
                run()
            ref = run(c.draft.to(torch.int32))
        else:
            ref = run()
        for n in common.KERNEL_NAMES:
            setattr(rs, n, _WRAPPERS[n])
        out = run()
    finally:
        for n, obj in saved.items():
            setattr(rs, n, obj)
    assert torch.equal(ref, out), (ref, out)
