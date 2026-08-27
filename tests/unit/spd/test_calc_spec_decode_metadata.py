# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QaicModelRunner._calc_spec_decode_metadata.

These tests require no QAIC hardware and no model loading. They verify that
draft_token_ids are extracted from the correct padded slots in input_ids_cpu,
and that target_logits_indices / bonus_logits_indices are correct in compacted
space.
"""

import types

import numpy as np
import torch

from vllm_qaic.worker.model_runner import QaicModelRunnerAoT as QaicModelRunner


def _make_mock_runner(
    max_decode_tokens: int,
    signal_num_scheduled_tokens: list[int],
    input_ids_cpu: list[int],
) -> types.SimpleNamespace:
    """Build a minimal mock that satisfies _calc_spec_decode_metadata.

    _calc_spec_decode_metadata accesses exactly:
      self.max_decode_tokens
      self.signal_num_scheduled_tokens
      self.device
      self.input_ids_cpu
      self.arange_np
      self._get_cumsum_and_arange  (inherited from GPUModelRunner)
    """
    max_n = max(len(input_ids_cpu) + 1, 512)
    arange_np = np.arange(max_n, dtype=np.int64)

    ns = types.SimpleNamespace(
        max_decode_tokens=max_decode_tokens,
        signal_num_scheduled_tokens=np.array(signal_num_scheduled_tokens, dtype=np.int32),
        device=torch.device("cpu"),
        input_ids_cpu=torch.tensor(input_ids_cpu, dtype=torch.int64),
        arange_np=arange_np,
    )
    # Bind the inherited method from GPUModelRunner so the mock behaves like
    # a real QaicModelRunner instance.
    ns._get_cumsum_and_arange = types.MethodType(
        QaicModelRunner._get_cumsum_and_arange, ns
    )
    return ns


def _call_metadata(runner, num_draft_tokens, cu_num_scheduled_tokens):
    """Call the unbound method on the mock runner."""
    return QaicModelRunner._calc_spec_decode_metadata(
        runner,
        np.array(num_draft_tokens, dtype=np.int32),
        np.array(cu_num_scheduled_tokens, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Test 1: Uniform draft counts, no padding variation
# ---------------------------------------------------------------------------


def test_uniform_no_padding():
    """2 requests, both fully packed (no padding).

    max_decode_tokens = 4  (3 spec tokens)
    signal = [4, 4]  -> num_pads = [0, 0]

    Padded layout (8 slots total):
      [req0_tok0, req0_tok1, req0_tok2, req0_tok3,
       req1_tok0, req1_tok1, req1_tok2, req1_tok3]

    num_draft_tokens = [3, 3]
    cu_num_scheduled_tokens = [4, 8]

    draft_token_indices = [0,1,2, 4,5,6]
    draft_token_ids = input_ids_cpu[[1,2,3, 5,6,7]]
    """
    mdt = 4
    signal = [4, 4]
    # Fill each slot with its index so we can verify extraction easily.
    input_ids_cpu = list(range(mdt * len(signal)))  # [0,1,2,3,4,5,6,7]

    runner = _make_mock_runner(mdt, signal, input_ids_cpu)
    meta = _call_metadata(runner, [3, 3], [4, 8])

    expected_draft_ids = torch.tensor([1, 2, 3, 5, 6, 7], dtype=torch.int64)
    assert torch.equal(meta.draft_token_ids, expected_draft_ids), (
        f"draft_token_ids mismatch: {meta.draft_token_ids} != {expected_draft_ids}"
    )

    # target_logits_indices in compacted space: [0,1,2, 4,5,6]
    expected_target = torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int32)
    assert torch.equal(meta.target_logits_indices, expected_target), (
        f"target_logits_indices mismatch: {meta.target_logits_indices}"
    )

    # bonus_logits_indices: last position of each request in compacted space
    # cu_num_sampled_tokens = [4, 8], bonus = [3, 7]
    expected_bonus = torch.tensor([3, 7], dtype=torch.int32)
    assert torch.equal(meta.bonus_logits_indices, expected_bonus), (
        f"bonus_logits_indices mismatch: {meta.bonus_logits_indices}"
    )


# ---------------------------------------------------------------------------
# Test 2: Mixed draft counts with padding
# ---------------------------------------------------------------------------


def test_mixed_draft_counts_with_padding():
    """2 requests with different actual token counts — the core padding test.

    max_decode_tokens = 5  (4 spec tokens)
    signal = [4, 2]  -> num_pads = [1, 3]

    Padded layout (10 slots):
      slot: 0  1  2  3  4  | 5  6  7  8  9
      req:  0  0  0  0  0  | 1  1  1  1  1
      val: 10 11 12 13 14  |20 21 22 23 24

    num_draft_tokens = [3, 1]
    cu_num_scheduled_tokens = [4, 6]  (compacted: 4 signal + 2 signal)

    From spd_internals.md worked example:
      draft_token_indices = [0, 1, 2, 5]
      draft_token_ids = input_ids_cpu[[1, 2, 3, 6]] = [11, 12, 13, 21]
    """
    mdt = 5
    signal = [4, 2]
    # Req0 occupies slots 0-4, req1 occupies slots 5-9.
    # Use distinctive values to catch off-by-one errors.
    input_ids_cpu = [
        10,
        11,
        12,
        13,
        14,  # req0 slots 0-4
        20,
        21,
        22,
        23,
        24,
    ]  # req1 slots 5-9

    runner = _make_mock_runner(mdt, signal, input_ids_cpu)
    meta = _call_metadata(runner, [3, 1], [4, 6])

    expected_draft_ids = torch.tensor([11, 12, 13, 21], dtype=torch.int64)
    assert torch.equal(meta.draft_token_ids, expected_draft_ids), (
        f"draft_token_ids mismatch: {meta.draft_token_ids} != {expected_draft_ids}"
    )

    # target_logits_indices in compacted space: [0,1,2, 4]
    expected_target = torch.tensor([0, 1, 2, 4], dtype=torch.int32)
    assert torch.equal(meta.target_logits_indices, expected_target), (
        f"target_logits_indices mismatch: {meta.target_logits_indices}"
    )

    # bonus_logits_indices: [3, 5]
    expected_bonus = torch.tensor([3, 5], dtype=torch.int32)
    assert torch.equal(meta.bonus_logits_indices, expected_bonus), (
        f"bonus_logits_indices mismatch: {meta.bonus_logits_indices}"
    )


# ---------------------------------------------------------------------------
# Test 3: Zero draft tokens for some requests
# ---------------------------------------------------------------------------


def test_zero_draft_tokens_for_some_requests():
    """3 requests where the middle one has no draft tokens.

    max_decode_tokens = 5
    signal = [4, 3, 4]  -> num_pads = [1, 2, 1]

    num_draft_tokens = [3, 0, 2]
    cu_num_scheduled_tokens = [4, 7, 11]

    draft_token_ids should have length 5 (3+0+2) and skip req1 entirely.

    Padded layout (15 slots):
      req0: slots  0- 4  -> values 100-104
      req1: slots  5- 9  -> values 200-204
      req2: slots 10-14  -> values 300-304

    num_sampled_tokens = [4, 1, 3]
    cu_num_sampled_tokens = [4, 5, 8]

    num_pads = [1, 2, 1]
    cu_num_pads = [1, 3, 4]

    draft_token_indices for req0 (3 drafts):
      base = (cu_num_sampled[0] - num_sampled[0]) + (cu_num_pads[0] - num_pads[0])
           = (4 - 4) + (1 - 1) = 0
      indices = [0, 1, 2]

    draft_token_indices for req1 (0 drafts): empty

    draft_token_indices for req2 (2 drafts):
      base = (cu_num_sampled[2] - num_sampled[2]) + (cu_num_pads[2] - num_pads[2])
           = (8 - 3) + (4 - 1) = 5 + 3 = 8
      indices = [8, 9]

    draft_token_ids = input_ids_cpu[[1,2,3, 9,10]]
                    = [101, 102, 103, 204, 300]
    """
    mdt = 5
    signal = [4, 3, 4]
    input_ids_cpu = (
        [100, 101, 102, 103, 104]  # req0 slots 0-4
        + [200, 201, 202, 203, 204]  # req1 slots 5-9
        + [300, 301, 302, 303, 304]  # req2 slots 10-14
    )

    runner = _make_mock_runner(mdt, signal, input_ids_cpu)
    meta = _call_metadata(runner, [3, 0, 2], [4, 7, 11])

    assert len(meta.draft_token_ids) == 5, (
        f"Expected 5 draft tokens, got {len(meta.draft_token_ids)}"
    )
    expected_draft_ids = torch.tensor([101, 102, 103, 204, 300], dtype=torch.int64)
    assert torch.equal(meta.draft_token_ids, expected_draft_ids), (
        f"draft_token_ids mismatch: {meta.draft_token_ids} != {expected_draft_ids}"
    )

    # target_logits_indices in compacted space:
    # req0 base = 0, indices [0,1,2]
    # req1: skipped (0 drafts)
    # req2 base = cu_num_sampled[2] - num_sampled[2] = 8 - 3 = 5, indices [5,6]
    expected_target = torch.tensor([0, 1, 2, 5, 6], dtype=torch.int32)
    assert torch.equal(meta.target_logits_indices, expected_target), (
        f"target_logits_indices mismatch: {meta.target_logits_indices}"
    )

    # bonus_logits_indices: cu_num_sampled - 1 = [3, 4, 7]
    expected_bonus = torch.tensor([3, 4, 7], dtype=torch.int32)
    assert torch.equal(meta.bonus_logits_indices, expected_bonus), (
        f"bonus_logits_indices mismatch: {meta.bonus_logits_indices}"
    )


# ---------------------------------------------------------------------------
# Test 4: Single request, maximum padding
# ---------------------------------------------------------------------------


def test_single_request_maximum_padding():
    """Single request with only 2 signal tokens out of 5 slots (3 pads).

    max_decode_tokens = 5
    signal = [2]  -> num_pads = [3]

    num_draft_tokens = [1]
    cu_num_scheduled_tokens = [2]

    Padded layout (5 slots):
      slot: 0   1   2   3   4
      val:  50  51  52  53  54

    num_sampled_tokens = [2]
    cu_num_sampled_tokens = [2]

    num_pads = [3], cu_num_pads = [3]

    draft_token_indices:
      base = (2 - 2) + (3 - 3) = 0
      indices = [0]

    draft_token_ids = input_ids_cpu[[1]] = [51]
    """
    mdt = 5
    signal = [2]
    input_ids_cpu = [50, 51, 52, 53, 54]

    runner = _make_mock_runner(mdt, signal, input_ids_cpu)
    meta = _call_metadata(runner, [1], [2])

    expected_draft_ids = torch.tensor([51], dtype=torch.int64)
    assert torch.equal(meta.draft_token_ids, expected_draft_ids), (
        f"draft_token_ids mismatch: {meta.draft_token_ids} != {expected_draft_ids}"
    )

    expected_target = torch.tensor([0], dtype=torch.int32)
    assert torch.equal(meta.target_logits_indices, expected_target), (
        f"target_logits_indices mismatch: {meta.target_logits_indices}"
    )

    # bonus_logits_indices: cu_num_sampled - 1 = [1]
    expected_bonus = torch.tensor([1], dtype=torch.int32)
    assert torch.equal(meta.bonus_logits_indices, expected_bonus), (
        f"bonus_logits_indices mismatch: {meta.bonus_logits_indices}"
    )
