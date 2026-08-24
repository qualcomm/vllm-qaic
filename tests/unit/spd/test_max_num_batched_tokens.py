# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for the QAIC max_num_batched_tokens buffer-sizing invariant.

No QAIC hardware, no model loading required.

Background
----------
QAIC's _prepare_qaic_inputs expands each decode request from 1 scheduled token
to max_decode_tokens = 1 + num_spec_tokens tokens so that the fixed-shape decode
QPC kernel receives a full batch.  The GPU model runner allocates token-indexed
buffers (positions, input_ids) to max_num_batched_tokens elements, so the
post-expansion token count must never exceed that budget.

The QAIC platform sets:
    max_num_batched_tokens = max_num_seqs * long_prefill_token_threshold

For a schedule with k decode requests (0 ≤ k ≤ max_num_seqs) the post-expansion
token count is:
    expanded(k) = k * max_decode_tokens
                + (max_num_seqs - k) * long_prefill_token_threshold

This is maximised at k=0 (all prefill):
    max expanded = max_num_seqs * long_prefill_token_threshold = budget  ✓

The tests below prove this invariant holds across realistic QAIC configurations
and demonstrate that the previous formula (max_num_seqs*(1+S)+seq_len) overflowed
on the crash scenario that triggered the bug fix.
"""

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QAIC_CONFIGS = [
    # (max_num_seqs, long_prefill_token_threshold, num_spec_tokens)
    (16, 128, 4),  # crash configuration: Llama-3.1-8B + ngram/suffix
    (4, 128, 4),  # small batch
    (8, 64, 3),  # shorter prefill chunks
    (32, 128, 5),  # larger batch, more spec tokens
    (16, 128, 0),  # no speculative decoding
    (16, 64, 0),  # no spec, shorter chunks
]


def _new_budget(max_num_seqs: int, long_prefill_token_threshold: int) -> int:
    """Current (correct) formula."""
    return max_num_seqs * long_prefill_token_threshold


def _old_budget(
    max_num_seqs: int,
    long_prefill_token_threshold: int,
    num_spec_tokens: int,
) -> int:
    """Previous (buggy) formula, kept here for regression comparison."""
    return max_num_seqs * (1 + num_spec_tokens) + long_prefill_token_threshold


def _max_expanded(
    k: int,
    max_num_seqs: int,
    long_prefill_token_threshold: int,
    num_spec_tokens: int,
) -> int:
    """Post-expansion token count for a schedule with k decode requests."""
    max_decode_tokens = 1 + num_spec_tokens
    prefill_slots = max_num_seqs - k
    return k * max_decode_tokens + prefill_slots * long_prefill_token_threshold


# ---------------------------------------------------------------------------
# Invariant: expanded ≤ budget for every valid k
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_num_seqs,long_prefill_token_threshold,num_spec_tokens", _QAIC_CONFIGS
)
def test_expanded_never_exceeds_budget(
    max_num_seqs: int,
    long_prefill_token_threshold: int,
    num_spec_tokens: int,
) -> None:
    """Post-expansion token count ≤ max_num_batched_tokens for all k."""
    budget = _new_budget(max_num_seqs, long_prefill_token_threshold)
    for k in range(max_num_seqs + 1):
        expanded = _max_expanded(
            k, max_num_seqs, long_prefill_token_threshold, num_spec_tokens
        )
        assert expanded <= budget, (
            f"Buffer overflow: expanded={expanded} > budget={budget} "
            f"(k={k}, max_num_seqs={max_num_seqs}, "
            f"long_prefill_token_threshold={long_prefill_token_threshold}, "
            f"num_spec_tokens={num_spec_tokens})"
        )


@pytest.mark.parametrize(
    "max_num_seqs,long_prefill_token_threshold,num_spec_tokens", _QAIC_CONFIGS
)
def test_budget_is_tight_at_k0(
    max_num_seqs: int,
    long_prefill_token_threshold: int,
    num_spec_tokens: int,
) -> None:
    """Budget equals the maximum expanded count, reached at k=0 (all prefill)."""
    budget = _new_budget(max_num_seqs, long_prefill_token_threshold)
    expanded_at_k0 = _max_expanded(
        0, max_num_seqs, long_prefill_token_threshold, num_spec_tokens
    )
    assert expanded_at_k0 == budget, (
        "Budget should equal the maximum expanded count (at k=0); "
        f"got expanded={expanded_at_k0}, budget={budget}"
    )


# ---------------------------------------------------------------------------
# Regression: crash scenario that triggered the bug fix
# ---------------------------------------------------------------------------


def test_crash_scenario_overflowed_old_formula() -> None:
    """The crash schedule was valid under the old 208-token budget but overflowed
    the 208-element buffer after decode expansion.

    Crash details:
        max_num_seqs=16, long_prefill_token_threshold=128, num_spec_tokens=4
        Scheduler scheduled: 1 decode (1 token) + 207 prefill tokens = 208 total
        After expansion:     5 tokens  + 207 prefill tokens = 212 > 208  ← crash
    """
    max_num_seqs = 16
    long_prefill_token_threshold = 128
    num_spec_tokens = 4
    max_decode_tokens = 1 + num_spec_tokens  # 5

    old_b = _old_budget(max_num_seqs, long_prefill_token_threshold, num_spec_tokens)
    new_b = _new_budget(max_num_seqs, long_prefill_token_threshold)

    # The exact crash schedule: 1 decode + 207 prefill tokens = 208 unexpanded
    crash_unexpanded = 1 + 207
    crash_expanded = max_decode_tokens + 207  # 212

    assert old_b == 208, f"Old budget should be 208, got {old_b}"
    assert new_b == 2048, f"New budget should be 2048, got {new_b}"

    # Schedule was valid under the old budget (unexpanded fit)
    assert crash_unexpanded <= old_b, (
        "Crash schedule unexpectedly invalid under old budget"
    )
    # But expanded overflowed the old buffer
    assert crash_expanded > old_b, "Old formula did not overflow as expected"
    # New budget handles it without overflow
    assert crash_expanded <= new_b, "New formula unexpectedly overflows on crash schedule"


@pytest.mark.parametrize("k", [1, 4, 8, 16])
def test_old_formula_overflows_for_mixed_batches(k: int) -> None:
    """Old formula overflowed whenever decodes and prefill were mixed.

    For the crash config (16, 128, 4), the old budget was 208.
    The scheduler could schedule k decodes + (208-k) prefill tokens = 208 total.
    After expansion: k*5 + (208-k) = 208 + 4k > 208 for any k ≥ 1.
    """
    max_num_seqs = 16
    long_prefill_token_threshold = 128
    num_spec_tokens = 4
    max_decode_tokens = 1 + num_spec_tokens

    old_b = _old_budget(max_num_seqs, long_prefill_token_threshold, num_spec_tokens)
    new_b = _new_budget(max_num_seqs, long_prefill_token_threshold)

    # Schedule: k decodes + remaining budget as prefill tokens
    prefill_tokens = old_b - k
    expanded = k * max_decode_tokens + prefill_tokens

    # Old formula overflows for any k ≥ 1
    assert expanded > old_b, (
        f"Expected overflow with old formula at k={k}: expanded={expanded}, old_b={old_b}"
    )
    # New formula never overflows (expanded = old_b + 4k ≤ 2048 for k ≤ 16)
    assert expanded <= new_b, (
        f"New formula overflowed at k={k}: expanded={expanded}, new_b={new_b}"
    )
