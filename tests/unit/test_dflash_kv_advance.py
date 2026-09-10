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


class _RecordingSession(_FakeSession):
    """_FakeSession that also records the active slot's target-hidden block fed
    to each np_run call, so prefill sub-block fan-out can be inspected."""

    def __init__(self, logits_buf: np.ndarray, vocab_size: int):
        super().__init__(logits_buf, vocab_size)
        self.target_hidden_calls: list[np.ndarray] = []

    def np_run(self, dlm_inputs, is_prefill=True):
        self.target_hidden_calls.append(dlm_inputs["target_hidden"][0].copy())
        return super().np_run(dlm_inputs, is_prefill=is_prefill)


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


def test_candidates_from_prefill_frozen_on_closed_gate(block_size, decode_bsz):
    """Regression for the prefill-desync case: a request fresh from prefill
    (candidates_from_prefill=True) must not freeze on a commit=False step. Before
    the fix it hit `continue` — no DLM forward, position_counter pinned — while
    the TLM kept advancing it, so its cached drafts and KV went stale. The fix
    discards the un-servable prefill drafts and advances the KV like any other
    active request."""
    proposer, session = _make_proposer(decode_bsz, block_size)
    input_batch = _FakeInputBatch(
        req_ids=["fresh_r"],
        num_prompt_tokens=[10],
        num_tokens_no_spec=[15],  # in decode phase
    )
    batch_indices = np.array([0], dtype=np.int64)
    target_hidden = np.zeros((1, block_size, proposer.hidden_size), dtype=np.float32)

    st = proposer._state_for("fresh_r")
    st.position_counter = 20
    st.candidates_from_prefill = True
    st.dlm_candidates = np.arange(block_size, dtype=np.int64)

    draft_token_ids = proposer.propose(
        input_batch, [[101]], batch_indices, target_hidden, commit=False
    )

    assert session.run_count == 1, (
        "commit=False on a candidates_from_prefill request must still issue the "
        "DLM forward instead of freezing on the cached-candidates skip-path."
    )
    assert st.position_counter == 21, (
        "position_counter did not advance on a closed-gate step — the DLM KV "
        "would fall behind the TLM for a freshly-prefilled request."
    )
    assert st.candidates_from_prefill is False
    assert st.dlm_candidates is None, "stale prefill drafts must be discarded"
    assert draft_token_ids == [[]]


def test_candidates_from_prefill_served_on_open_gate(block_size, decode_bsz):
    """Happy path is unchanged: on a commit=True step a candidates_from_prefill
    request serves its cached prefill drafts once (no DLM forward, position
    frozen until the next real decode step) and clears the flag."""
    proposer, session = _make_proposer(decode_bsz, block_size)
    input_batch = _FakeInputBatch(
        req_ids=["fresh_r"],
        num_prompt_tokens=[10],
        num_tokens_no_spec=[15],
    )
    batch_indices = np.array([0], dtype=np.int64)
    target_hidden = np.zeros((1, block_size, proposer.hidden_size), dtype=np.float32)

    st = proposer._state_for("fresh_r")
    st.position_counter = 20
    st.candidates_from_prefill = True
    st.dlm_candidates = np.array([7, 1, 2, 3], dtype=np.int64)

    draft_token_ids = proposer.propose(
        input_batch, [[101]], batch_indices, target_hidden, commit=True
    )

    assert session.run_count == 0, "serving cached prefill drafts issues no forward"
    assert draft_token_ids[0] == [1, 2, 3], "must offer dlm_candidates[1:]"
    assert st.candidates_from_prefill is False
    assert st.position_counter == 20, "the serve step does not advance the counter"


def _prep_prefill_proposer(decode_bsz, block_size, sub_blocks_per_chunk=2):
    """Proposer set up for prefill_step: adds the tlm_prefill_seq_len /
    num_sub_blocks the decode-only _make_proposer leaves unset."""
    proposer, _session = _make_proposer(decode_bsz, block_size)
    proposer.tlm_prefill_seq_len = sub_blocks_per_chunk * block_size
    proposer.num_sub_blocks = sub_blocks_per_chunk
    return proposer


def test_build_prefill_pending_maps_each_request_to_its_own_chunks(
    block_size, decode_bsz
):
    """Regression for >1 prefill chunk per request (upstream qualcomm/vllm-qaic
    PR #74 lets the scheduler feed multiple chunks per request in a step).
    build_prefill_pending must advance its chunk cursor by
    ceil(req_tokens / tlm_prefill_seq_len), so a request spanning two TLM chunks
    consumes both hidden buffers and the next request still lands on its own
    buffer. The old cursor += 1 per request handed request 1 request 0's second
    chunk, silently corrupting its DLM KV."""
    proposer = _prep_prefill_proposer(decode_bsz, block_size)
    tlm_pfl = proposer.tlm_prefill_seq_len  # 8

    # "r0": 12 tokens -> ceil(12/8) = 2 chunks; "r1": 5 tokens -> 1 chunk.
    n0, n1 = 12, 5
    prefill_cum_sum = np.array([n0, n0 + n1], dtype=np.int64)
    prefill_positions = np.arange(n0 + n1, dtype=np.int64)
    prefill_block_ids = np.array([0, 1], dtype=np.int64)
    prefill_is_partial = np.array([False, False])
    # One (1, tlm_pfl, hidden) buffer per TLM chunk, fingerprinted by fill value:
    # chunks 0 and 1 belong to r0; chunk 2 belongs to r1.
    chunks = [
        np.full((1, tlm_pfl, proposer.hidden_size), float(c), dtype=np.float32)
        for c in range(3)
    ]

    proposer.build_prefill_pending(
        prefill_cum_sum,
        prefill_positions,
        prefill_block_ids,
        prefill_is_partial,
        None,  # hidden_states_prefill (last-chunk logits) unused here
        chunks,
        ["r0", "r1"],
    )

    pending = proposer._prefill_pending
    assert len(pending) == 2

    r0 = pending[0]
    assert r0["req_id"] == "r0"
    assert len(r0["target_hidden"]) == 2, "12-token request must span 2 TLM chunks"
    assert r0["target_hidden"][0] is chunks[0]
    assert r0["target_hidden"][1] is chunks[1]

    r1 = pending[1]
    assert r1["req_id"] == "r1"
    assert len(r1["target_hidden"]) == 1
    # The crux: r1 gets chunk 2 (its own), not chunk 1 (r0's second chunk).
    assert r1["target_hidden"][0] is chunks[2]
    assert float(r1["target_hidden"][0][0, 0, 0]) == 2.0


def test_prefill_step_fans_sub_blocks_across_multiple_chunks(block_size, decode_bsz):
    """prefill_step must fan a multi-chunk request into ceil(n_tokens/block_size)
    DLM sub-block calls, pulling each sub-block's target hidden from the correct
    chunk buffer (a block never straddles a chunk boundary)."""
    proposer = _prep_prefill_proposer(decode_bsz, block_size)
    rec = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = rec
    tlm_pfl = proposer.tlm_prefill_seq_len  # 8

    n_tokens = 12  # spans 2 chunks; ceil(12/4) = 3 sub-blocks of real tokens
    # Fingerprint: chunk 0 rows = 100 + t, chunk 1 rows = 200 + t.
    c0 = np.zeros((1, tlm_pfl, proposer.hidden_size), dtype=np.float32)
    c1 = np.zeros((1, tlm_pfl, proposer.hidden_size), dtype=np.float32)
    for t in range(tlm_pfl):
        c0[0, t, :] = 100 + t
        c1[0, t, :] = 200 + t

    proposer.prefill_step(
        target_hidden=[c0, c1],
        prefill_positions=np.arange(n_tokens, dtype=np.int64),
        prefill_cum_sum=np.array([n_tokens], dtype=np.int64),
        batch_indices=np.array([0], dtype=np.int64),
        prefill_is_partial=np.array([False]),  # final chunk of the request
        last_chunk_logits=None,
        req_id="r0",
    )

    assert rec.run_count == 3, "ceil(12/4) sub-block forwards expected"
    assert len(rec.target_hidden_calls) == 3
    # sub-block 0 -> chunk0[0:4]; 1 -> chunk0[4:8]; 2 -> chunk1[0:4].
    assert rec.target_hidden_calls[0][0, 0] == 100  # chunk 0, token 0
    assert rec.target_hidden_calls[1][0, 0] == 104  # chunk 0, token 4
    assert rec.target_hidden_calls[2][0, 0] == 200  # chunk 1, token 0

    st = proposer._req_state["r0"]
    assert st.position_counter == n_tokens - 1  # last real position
    assert st.candidates_from_prefill is True
    assert st.dlm_candidates is not None


def test_prefill_step_partial_multichunk_advances_without_candidates(
    block_size, decode_bsz
):
    """A non-final (partial) multi-chunk chunk advances the DLM KV across all its
    sub-blocks and sets position_counter, but offers no candidates."""
    proposer = _prep_prefill_proposer(decode_bsz, block_size)
    rec = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = rec
    tlm_pfl = proposer.tlm_prefill_seq_len

    n_tokens = 12  # 2 chunks, ceil(12/4) = 3 sub-blocks
    target_hidden = [
        np.zeros((1, tlm_pfl, proposer.hidden_size), dtype=np.float32),
        np.zeros((1, tlm_pfl, proposer.hidden_size), dtype=np.float32),
    ]

    proposer.prefill_step(
        target_hidden=target_hidden,
        prefill_positions=np.arange(n_tokens, dtype=np.int64),
        prefill_cum_sum=np.array([n_tokens], dtype=np.int64),
        batch_indices=np.array([0], dtype=np.int64),
        prefill_is_partial=np.array([True]),  # NOT the final chunk
        last_chunk_logits=None,
        req_id="r0",
    )

    assert rec.run_count == 3
    st = proposer._req_state["r0"]
    assert st.position_counter == n_tokens - 1
    assert st.candidates_from_prefill is False
    assert st.dlm_candidates is None
