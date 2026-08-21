# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Unit-level regression test for the DFlash decode-side KV-advance-on-skip fix.

Reviewer comment (PR #71): when one request's sequence exceeds
effective_drafter_max_model_len, the step-wide input_fits_in_drafter gate goes
False and full draft-proposing is skipped for *every* request that step, not
just the offending one. Before the fix, QaicDFlashProposer.propose() was only
ever called when that gate was True, so every other in-decode-phase request's
DLM KV cache (position_counter) silently fell behind the TLM. This test drives
QaicDFlashProposer.propose() directly (no QAIC device / QPC needed — the DLM
forward is replaced by a fake session) to check that a commit=False call still
advances position_counter for active requests while not offering discarded
candidates.
"""

import numpy as np
import pytest

from vllm_qaic.spec_decode.dflash_draft_model import QaicDFlashProposer


class _FakeSession:
    """Stands in for QAICInferenceSession: records calls, returns canned logits."""

    def __init__(self, logits_buf: np.ndarray, vocab_size: int):
        self.logits_buf = logits_buf
        self.vocab_size = vocab_size
        self.run_count = 0

    def np_run(self, dlm_inputs, is_prefill=True):
        self.run_count += 1
        # argmax-friendly: put a distinct winning logit per (slot, block) so
        # tests can trivially decode the "output" candidates if needed.
        self.logits_buf[:] = 0.0
        self.logits_buf[..., 0] = 1.0
        return 0

    def complete_inf(self, exec_obj_idx, is_prefill=True):
        pass


class _FakeInputBatch:
    def __init__(self, req_ids, num_prompt_tokens, num_tokens_no_spec):
        self.num_reqs = len(req_ids)
        self.req_ids = list(req_ids)
        self.num_prompt_tokens = np.array(num_prompt_tokens, dtype=np.int64)
        self.num_tokens_no_spec = np.array(num_tokens_no_spec, dtype=np.int64)


def _make_proposer(decode_bsz: int, block_size: int, hidden_size: int = 8):
    """Build a QaicDFlashProposer without going through __init__'s VllmConfig /
    QPC-loading path — this proposer only needs its numpy input buffers and a
    fake session to exercise propose()'s control flow."""
    proposer = QaicDFlashProposer.__new__(QaicDFlashProposer)
    proposer.block_size = block_size
    proposer.decode_bsz = decode_bsz
    proposer.mask_token_id = 999
    proposer.hidden_size = hidden_size
    vocab_size = 32
    proposer.vocab_size = vocab_size

    proposer._mask_row = np.full((block_size,), proposer.mask_token_id, dtype=np.int64)
    proposer._dlm_input_ids = np.tile(proposer._mask_row, (decode_bsz, 1))
    proposer._dlm_position_ids = np.full((decode_bsz, block_size), -1, dtype=np.int64)
    proposer._dlm_position_ids_target = np.full(
        (decode_bsz, block_size), -1, dtype=np.int64
    )
    proposer._dlm_target_hidden = np.zeros(
        (decode_bsz, block_size, hidden_size), dtype=np.float32
    )
    proposer._dlm_batch_index = np.full((decode_bsz, 1), -1, dtype=np.int64)
    proposer._dlm_logits_buf = np.zeros(
        (decode_bsz, block_size, vocab_size), dtype=np.float32
    )

    fake_session = _FakeSession(proposer._dlm_logits_buf, vocab_size)

    class _FakeModel:
        session = fake_session

    proposer.model = _FakeModel()
    proposer._req_state = {}
    proposer._prefill_pending = []
    return proposer, fake_session


@pytest.fixture
def block_size():
    return 4


@pytest.fixture
def decode_bsz():
    return 4


def test_commit_false_advances_kv_for_all_active_requests(block_size, decode_bsz):
    """The core regression check: a commit=False propose() call (the new skip-path
    the runner takes when input_fits_in_drafter is False) must still advance
    position_counter for every active in-decode-phase request, exactly like a
    commit=True call would — otherwise those requests' DLM KV cache falls behind
    the TLM the moment one sibling request gets too long for the drafter."""
    proposer, session = _make_proposer(decode_bsz, block_size)
    input_batch = _FakeInputBatch(
        req_ids=["short_a", "short_b", "long_c"],
        num_prompt_tokens=[10, 10, 10],
        num_tokens_no_spec=[15, 15, 15],  # all in decode phase
    )
    batch_indices = np.array([0, 1, 2], dtype=np.int64)
    target_hidden = np.zeros((3, block_size, proposer.hidden_size), dtype=np.float32)

    for req_id in input_batch.req_ids:
        proposer._state_for(req_id).position_counter = 20

    sampled_token_ids = [[101], [102], [103]]

    draft_token_ids = proposer.propose(
        input_batch, sampled_token_ids, batch_indices, target_hidden, commit=False
    )

    assert session.run_count == 1, "commit=False must still issue the DLM forward"
    for req_id in input_batch.req_ids:
        assert proposer._req_state[req_id].position_counter == 21, (
            f"DLM KV position_counter for {req_id} did not advance on a "
            "commit=False call — it would silently fall behind the TLM."
        )
    # Discarded candidates: nothing should be offered to the scheduler this step.
    assert draft_token_ids == [[], [], []]


def test_commit_false_does_not_set_candidates(block_size, decode_bsz):
    """A commit=False step must not leave dlm_candidates looking like a real
    offer: it should stay None so a subsequent commit=True call's candidates
    are unambiguously the result of that later, real forward."""
    proposer, _session = _make_proposer(decode_bsz, block_size)
    input_batch = _FakeInputBatch(
        req_ids=["req_a"],
        num_prompt_tokens=[10],
        num_tokens_no_spec=[15],
    )
    batch_indices = np.array([0], dtype=np.int64)
    target_hidden = np.zeros((1, block_size, proposer.hidden_size), dtype=np.float32)
    proposer._state_for("req_a").position_counter = 5

    draft_token_ids = proposer.propose(
        input_batch, [[201]], batch_indices, target_hidden, commit=False
    )
    st = proposer._req_state["req_a"]
    assert st.dlm_candidates is None
    assert draft_token_ids == [[]]

    # A subsequent commit=True call should produce real candidates from this
    # (correctly, unconditionally advanced) position_counter.
    draft_token_ids = proposer.propose(
        input_batch, [[202]], batch_indices, target_hidden, commit=True
    )
    assert st.dlm_candidates is not None
    assert draft_token_ids != [[]]


def test_commit_true_after_commit_false_keeps_position_counter_in_sync(
    block_size, decode_bsz
):
    """End-to-end sequencing check: a commit=False step (KV-advance only) followed
    by a commit=True step must leave position_counter advanced by both steps'
    accepted lengths, matching what would happen if the gate had never skipped
    proposing at all."""
    proposer, _session = _make_proposer(decode_bsz, block_size)
    input_batch = _FakeInputBatch(
        req_ids=["req_a"],
        num_prompt_tokens=[10],
        num_tokens_no_spec=[15],
    )
    batch_indices = np.array([0], dtype=np.int64)
    target_hidden = np.zeros((1, block_size, proposer.hidden_size), dtype=np.float32)
    proposer._state_for("req_a").position_counter = 5

    # Step 1: skipped for the whole batch (another request was too long).
    proposer.propose(
        input_batch, [[301, 302]], batch_indices, target_hidden, commit=False
    )
    assert proposer._req_state["req_a"].position_counter == 7

    # Step 2: gate reopens, this request proposes normally.
    proposer.propose(input_batch, [[303]], batch_indices, target_hidden, commit=True)
    assert proposer._req_state["req_a"].position_counter == 8
