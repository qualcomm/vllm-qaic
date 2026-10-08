# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/rejection_sampler.py

"""Numba CPU ports of vLLM's rejection-sampler Triton kernels.

Each ``*_numba`` function takes the same CPU tensors as the matching upstream
Triton kernel (vllm/v1/sample/rejection_sampler.py, v0.23.0) and writes into
the same preallocated output, so results can be compared bit-exactly.  The
tensor -> NumPy view conversion happens inside the function and is therefore
part of any timed call.

Semantics mirrored from upstream:
  * expand_kernel                  -> expand_numba
  * rejection_greedy_sample_kernel -> greedy_numba
  * rejection_random_sample_kernel -> random_numba
  * sample_recovered_tokens_kernel -> recovered_numba

The three small kernels are dispatch-bound, so they run serially; recovered
is vocab-bound and parallelizes over draft tokens with ``prange``.

Integer inputs may be int32 or int64 (Numba specializes per dtype); stores
narrow to the output dtype exactly like a Triton ``tl.store``.  The
production entry points, with validation, live in
``vllm_qaic.v1.sample.rejection_sampler_numba``.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange

_EMPTY_BOOL = np.ones(0, dtype=np.bool_)
_EMPTY_F32 = np.zeros(0, dtype=np.float32)
_EMPTY_F64 = np.zeros(0, dtype=np.float64)
_EMPTY_F32_2D = np.zeros((0, 0), dtype=np.float32)


def _np(t: torch.Tensor) -> np.ndarray:
    # A non-contiguous output would make reshape(-1) copy and silently drop
    # writes; .numpy() itself rejects non-CPU tensors and bf16.  Dtypes are
    # allowlisted by rejection_sampler_numba._check.
    if not t.is_contiguous():
        raise ValueError("numba rejection kernels require contiguous CPU tensors")
    return t.detach().numpy()


# --------------------------------------------------------------------------
# expand
# --------------------------------------------------------------------------
@njit(nogil=True, cache=True)
def _expand(out, x, cu, replace_from, replace_to):
    start = 0
    for req in range(cu.shape[0]):
        end = cu[req]
        value = x[req]
        if value == replace_from:
            value = replace_to
        for i in range(start, end):
            out[i] = value
        start = end


def expand_numba(output, input_, cu_num_tokens, replace_from, replace_to):
    _expand(_np(output), _np(input_), _np(cu_num_tokens), replace_from, replace_to)


# --------------------------------------------------------------------------
# greedy
# --------------------------------------------------------------------------
@njit(nogil=True, cache=True, inline="always")
def _greedy_row(out, req, cu, draft, argmax, bonus, stride, uniform, rates, synthetic):
    start = 0 if req == 0 else cu[req - 1]
    n = cu[req] - start
    base = req * stride
    rejected = False
    for pos in range(n):
        d = draft[start + pos]
        t = np.int32(argmax[start + pos])
        if synthetic:
            accepted = uniform[start + pos] < rates[pos]
            out[base + pos] = d if accepted else t
            rejected = not accepted
        else:
            out[base + pos] = t
            rejected = d != t
        if rejected:
            break
    if not rejected:
        out[base + n] = bonus[req]


@njit(nogil=True, cache=True)
def _greedy(
    out,
    cu,
    draft,
    argmax,
    bonus,
    is_greedy,
    has_mask,
    stride,
    uniform,
    rates,
    synthetic,
):
    for req in range(cu.shape[0]):
        if has_mask and not is_greedy[req]:
            continue
        _greedy_row(
            out, req, cu, draft, argmax, bonus, stride, uniform, rates, synthetic
        )


def greedy_numba(
    output,
    cu_num_draft_tokens,
    draft_token_ids,
    target_argmax,
    bonus_token_ids,
    is_greedy,
    max_spec_len,
    uniform_probs,
    synthetic_rates,
    synthetic_mode,
):
    _greedy(
        _np(output).reshape(-1),
        _np(cu_num_draft_tokens),
        _np(draft_token_ids),
        _np(target_argmax),
        _np(bonus_token_ids).reshape(-1),
        _EMPTY_BOOL if is_greedy is None else _np(is_greedy),
        is_greedy is not None,
        max_spec_len + 1,
        _EMPTY_F64 if uniform_probs is None else _np(uniform_probs),
        _EMPTY_F32 if synthetic_rates is None else _np(synthetic_rates),
        bool(synthetic_mode),
    )


# --------------------------------------------------------------------------
# random
# --------------------------------------------------------------------------
@njit(nogil=True, cache=True, inline="always")
def _random_row(
    out,
    req,
    cu,
    draft,
    draft_probs,
    target_probs,
    bonus,
    recovered,
    uniform,
    stride,
    rates,
    no_draft_probs,
    synthetic,
):
    start = 0 if req == 0 else cu[req - 1]
    n = cu[req] - start
    base = req * stride
    rejected = False
    for pos in range(n):
        tok = start + pos
        d = draft[tok]
        if synthetic:
            accepted = uniform[tok] < rates[pos]
        else:
            # Keep the division in the input dtype (fp32/fp32) as Triton does;
            # with no draft probs Triton divides by 1, which is exact.
            target_prob = target_probs[tok, d]
            if no_draft_probs:
                accepted = target_prob >= uniform[tok]
            else:
                draft_prob = draft_probs[tok, d]
                accepted = draft_prob > 0 and target_prob / draft_prob >= uniform[tok]
        out[base + pos] = d if accepted else recovered[tok]
        if not accepted:
            rejected = True
            break
    if not rejected:
        out[base + n] = bonus[req]


@njit(nogil=True, cache=True)
def _random(
    out,
    cu,
    draft,
    draft_probs,
    target_probs,
    bonus,
    recovered,
    uniform,
    is_greedy,
    stride,
    rates,
    no_draft_probs,
    synthetic,
):
    for req in range(cu.shape[0]):
        if is_greedy[req]:
            continue
        _random_row(
            out,
            req,
            cu,
            draft,
            draft_probs,
            target_probs,
            bonus,
            recovered,
            uniform,
            stride,
            rates,
            no_draft_probs,
            synthetic,
        )


def random_numba(
    output,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    bonus_token_ids,
    recovered_token_ids,
    uniform_probs,
    is_greedy,
    max_spec_len,
    vocab_size,
    synthetic_rates,
    no_draft_probs,
    synthetic_mode,
):
    del vocab_size  # implied by target_probs.shape
    _random(
        _np(output).reshape(-1),
        _np(cu_num_draft_tokens),
        _np(draft_token_ids),
        _EMPTY_F32_2D if draft_probs is None else _np(draft_probs),
        _np(target_probs),
        _np(bonus_token_ids).reshape(-1),
        _np(recovered_token_ids),
        _np(uniform_probs),
        _np(is_greedy),
        max_spec_len + 1,
        _EMPTY_F32 if synthetic_rates is None else _np(synthetic_rates),
        bool(no_draft_probs or draft_probs is None),
        bool(synthetic_mode),
    )


# --------------------------------------------------------------------------
# recovered
# --------------------------------------------------------------------------
# LLVM only emits packed vmaxps/vcmpeqps for these reductions when the loop
# body is a fixed-width lane block over a contiguous slice: a single running
# max is a loop-carried scalar dependence, and lo/hi bounds defeat the
# vectorizer; the find-first passes also need the target in the score dtype
# and an integer hit count rather than a boolean OR.  32 lanes is the
# narrowest width that vectorized on AVX-512.
_LANES = 32


@njit(nogil=True, cache=True)
def _max_score_masked(t, q, d):
    """max(t' * q) where t' is t with column d zeroed, in the promoted dtype."""
    n = t.shape[0]
    nb = n // _LANES
    zero = t.dtype.type(0)
    best = (t[:1] * q[:1]).dtype.type(-np.inf)
    if nb:
        acc = np.empty(_LANES, dtype=(t[:1] * q[:1]).dtype)
        for j in range(_LANES):
            acc[j] = (zero if j == d else t[j]) * q[j]
        for i in range(1, nb):
            base = i * _LANES
            for j in range(_LANES):
                acc[j] = max(
                    acc[j], (zero if base + j == d else t[base + j]) * q[base + j]
                )
        for j in range(_LANES):
            best = max(best, acc[j])
    for v in range(nb * _LANES, n):
        best = max(best, (zero if v == d else t[v]) * q[v])
    return best


@njit(nogil=True, cache=True)
def _max_score_clamped(t, d, q):
    """max(max(t - d, 0) * q) in the promoted dtype; -inf for an empty slice."""
    n = t.shape[0]
    nb = n // _LANES
    zero = t.dtype.type(0)
    best = (t[:1] * q[:1]).dtype.type(-np.inf)
    if nb:
        acc = np.empty(_LANES, dtype=(t[:1] * q[:1]).dtype)
        for j in range(_LANES):
            acc[j] = max(t[j] - d[j], zero) * q[j]
        for i in range(1, nb):
            base = i * _LANES
            for j in range(_LANES):
                acc[j] = max(acc[j], max(t[base + j] - d[base + j], zero) * q[base + j])
        for j in range(_LANES):
            best = max(best, acc[j])
    for v in range(nb * _LANES, n):
        best = max(best, max(t[v] - d[v], zero) * q[v])
    return best


@njit(nogil=True, cache=True)
def _find_score_masked(t, q, d, target):
    """First v with t'[v] * q[v] == target (t' = t with column d zeroed), or -1."""
    n = t.shape[0]
    nb = n // _LANES
    zero = t.dtype.type(0)
    for i in range(nb):
        base = i * _LANES
        hits = 0
        for j in range(_LANES):
            hits += (zero if base + j == d else t[base + j]) * q[base + j] == target
        if hits:
            for j in range(_LANES):
                if (zero if base + j == d else t[base + j]) * q[base + j] == target:
                    return base + j
    for v in range(nb * _LANES, n):
        if (zero if v == d else t[v]) * q[v] == target:
            return v
    return -1


@njit(nogil=True, cache=True)
def _find_score_clamped(t, d, q, target):
    """First v with max(t[v] - d[v], 0) * q[v] == target, or -1."""
    n = t.shape[0]
    nb = n // _LANES
    zero = t.dtype.type(0)
    for i in range(nb):
        base = i * _LANES
        hits = 0
        for j in range(_LANES):
            hits += max(t[base + j] - d[base + j], zero) * q[base + j] == target
        if hits:
            for j in range(_LANES):
                if max(t[base + j] - d[base + j], zero) * q[base + j] == target:
                    return base + j
    for v in range(nb * _LANES, n):
        if max(t[v] - d[v], zero) * q[v] == target:
            return v
    return -1


@njit(nogil=True, cache=True)
def _recovered_token(
    out, tok, req, draft, draft_probs, target_probs, inv_q, no_draft_probs
):
    # Upstream takes argmax_v(prob[v] * inv_q[req, v]) with strict ">" across
    # tiles, i.e. the first index of the maximum.  Two passes (vectorized max,
    # then vectorized find-first) give the same answer bit-exactly because
    # every score is recomputed with identical operations and dtypes.
    t = target_probs[tok]
    q = inv_q[req]
    if no_draft_probs:
        # The draft column's probability is masked to 0 before scaling.
        # Masking inline beats slicing around d: slices break 64-byte
        # alignment and split the row into two short reductions.
        d = draft[tok]
        best = _max_score_masked(t, q, d)
        idx = _find_score_masked(t, q, d, best)
    else:
        dp = draft_probs[tok]
        best = _max_score_clamped(t, dp, q)
        idx = _find_score_clamped(t, dp, q, best)
    out[tok] = max(idx, 0)


@njit(nogil=True, cache=True)
def _token_to_req(cu, num_tokens):
    req_of = np.empty(num_tokens, dtype=np.int32)
    start = 0
    for req in range(cu.shape[0]):
        for i in range(start, cu[req]):
            req_of[i] = req
        start = cu[req]
    return req_of


@njit(nogil=True, cache=True, parallel=True)
def _recovered(out, cu, draft, draft_probs, target_probs, inv_q, no_draft_probs):
    num_tokens = cu[cu.shape[0] - 1] if cu.shape[0] else 0
    req_of = _token_to_req(cu, num_tokens)
    for tok in prange(num_tokens):
        _recovered_token(
            out,
            tok,
            req_of[tok],
            draft,
            draft_probs,
            target_probs,
            inv_q,
            no_draft_probs,
        )


def recovered_numba(
    output,
    cu_num_draft_tokens,
    draft_token_ids,
    draft_probs,
    target_probs,
    inv_q,
    vocab_size,
    no_draft_probs,
):
    del vocab_size
    _recovered(
        _np(output),
        _np(cu_num_draft_tokens),
        _np(draft_token_ids),
        _EMPTY_F32_2D if draft_probs is None else _np(draft_probs),
        _np(target_probs),
        _np(inv_q),
        bool(no_draft_probs or draft_probs is None),
    )
