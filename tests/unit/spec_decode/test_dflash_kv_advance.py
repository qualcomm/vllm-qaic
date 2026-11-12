# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Unit-level regression test for the DFlash decode-side KV-advance-on-skip fix."""

import json
from types import SimpleNamespace

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
        self.input_calls: list[dict[str, np.ndarray]] = []

    def np_run(self, dlm_inputs, is_prefill=True):
        self.input_calls.append({k: v.copy() for k, v in dlm_inputs.items()})
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
    proposer.target_seq_len = block_size
    proposer.decode_bsz = decode_bsz
    proposer.max_model_len = 4096
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


@pytest.mark.parametrize("decode_bsz", [1, 4])
@pytest.mark.parametrize("target_width", [4, 8])
@pytest.mark.parametrize("commit", [False, True])
def test_queued_prompt_matches_existing_prefill_inputs(
    decode_bsz, target_width, commit
):
    lengths = [13] if decode_bsz == 1 else [13, 5]
    slots = [0] if decode_bsz == 1 else [3, 0]
    new_ids = [f"new_{i}" for i in range(len(lengths))]
    existing = [] if decode_bsz == 1 else ["decode"]
    results = []
    for queued in (False, True):
        proposer = _prep_prefill_proposer(decode_bsz, 4)
        proposer.target_seq_len = target_width
        proposer.num_sub_blocks = proposer.tlm_prefill_seq_len // target_width
        proposer._dlm_position_ids_target = np.full(
            (decode_bsz, target_width), -1, dtype=np.int64
        )
        proposer._dlm_target_hidden = np.zeros(
            (decode_bsz, target_width, proposer.hidden_size), dtype=np.float32
        )
        recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
        proposer.model.session = recorder
        if existing:
            proposer._state_for("decode").position_counter = 20
        logits = np.zeros((len(lengths), 1, proposer.vocab_size), dtype=np.float32)
        chunks = []
        for i, length in enumerate(lengths):
            features = np.arange(length * proposer.hidden_size, dtype=np.float32)
            features = features.reshape(length, proposer.hidden_size) + 1000 * i
            logits[i, 0, 7 + i] = 1
            if queued:
                proposer.queue_prefill(
                    new_ids[i], slots[i], features, length, logits[i : i + 1]
                )
                features.fill(-999)
            else:
                hidden = np.zeros(
                    (-(-length // 8), 1, 8, proposer.hidden_size), dtype=np.float32
                )
                hidden.reshape(-1, proposer.hidden_size)[:length] = features
                chunks.extend(hidden)
        if not queued:
            proposer.build_prefill_pending(
                np.cumsum(lengths, dtype=np.int64),
                np.concatenate([np.arange(n, dtype=np.int64) for n in lengths]),
                np.array(slots),
                np.zeros(len(lengths), dtype=bool),
                logits,
                chunks,
                new_ids,
            )
            proposer.update_prefill_kv()
        batch = _FakeInputBatch(
            existing + new_ids,
            ([8] if existing else []) + lengths,
            ([22] if existing else []) + [n + 1 for n in lengths],
        )
        batch_indices = np.array(([2] if existing else []) + slots)
        sampled = ([[3]] if existing else []) + [[7 + i] for i in range(len(lengths))]
        decode_hidden = np.full(
            (batch.num_reqs, 4, proposer.hidden_size), 42, np.float32
        )
        first = proposer.propose(batch, sampled, batch_indices, decode_hidden, commit)
        expected = sum(-(-n // target_width) for n in lengths)
        assert recorder.run_count == expected + int(bool(existing) or not commit)
        assert not proposer._prefill_pending
        before = recorder.run_count
        batch.num_tokens_no_spec += 1
        second = proposer.propose(batch, sampled, batch_indices, decode_hidden, commit)
        assert recorder.run_count == before + 1
        states = {r: s.position_counter for r, s in proposer._req_state.items()}
        results.append((first, second, states, recorder.input_calls))
    assert results[0][:3] == results[1][:3]
    assert len(results[0][3]) == len(results[1][3])
    for old, new in zip(results[0][3], results[1][3], strict=True):
        for name in old.keys() - {"logits"}:
            np.testing.assert_array_equal(old[name], new[name], err_msg=name)


@pytest.mark.parametrize("commit", [False, True])
def test_propose_drains_partial_prefill_without_sampled_tokens(commit):
    proposer = _prep_prefill_proposer(1, 4)
    recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = recorder
    proposer.build_prefill_pending(
        np.array([5]),
        np.arange(5),
        np.array([0]),
        np.array([True]),
        None,
        [np.ones((1, 8, proposer.hidden_size), dtype=np.float32)],
        ["r"],
    )
    batch = _FakeInputBatch(["r"], [20], [20])
    assert proposer.propose(
        batch, [[]], np.array([0]), np.zeros((1, 4, proposer.hidden_size)), commit
    ) == [[]]
    assert recorder.run_count == 2
    assert not proposer._prefill_pending
    assert proposer._req_state["r"].position_counter == 4


def test_queued_prefill_reinitializes_reused_request_and_cache_slot():
    proposer = _prep_prefill_proposer(1, 4)
    recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = recorder
    for req_id, length, value in [("old", 13, 11), ("new", 5, 22), ("new", 9, 33)]:
        logits = np.zeros((1, 1, proposer.vocab_size), dtype=np.float32)
        proposer.queue_prefill(
            req_id,
            0,
            np.full((length, proposer.hidden_size), value, np.float32),
            length,
            logits,
        )
        start = len(recorder.input_calls)
        batch = _FakeInputBatch([req_id], [length], [length + 1])
        proposer.propose(batch, [[0]], np.array([0]), np.zeros((1, 4, 8)))
        assert set(proposer._req_state) == {req_id}
        assert proposer._req_state[req_id].position_counter == length - 1
        assert not proposer._prefill_pending
        calls = recorder.input_calls[start:]
        assert len(calls) == -(-length // 4)
        for inputs in calls:
            valid = inputs["position_ids_target"][0] >= 0
            assert np.all(inputs["target_hidden"][0, valid] == value)
            assert inputs["batch_index"][0, 0] == 0


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


@pytest.mark.parametrize("n_tokens", [1, 8, 13, 16])
@pytest.mark.parametrize("partial", [False, True])
def test_wide_target_prefill_preserves_features_with_fewer_calls(n_tokens, partial):
    for target_width in (4, 8):
        proposer = _prep_prefill_proposer(4, 4)
        proposer.target_seq_len = target_width
        proposer.num_sub_blocks = 8 // target_width
        proposer._dlm_position_ids_target = np.full(
            (4, target_width), -1, dtype=np.int64
        )
        proposer._dlm_target_hidden = np.zeros((4, target_width, 8), dtype=np.float32)
        recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
        proposer.model.session = recorder
        features = np.arange(16 * 8, dtype=np.float32).reshape(1, 16, 8)
        proposer.prefill_step(
            target_hidden=[features[:, :8], features[:, 8:]],
            prefill_positions=np.arange(n_tokens),
            prefill_cum_sum=np.array([n_tokens]),
            batch_indices=np.array([3]),
            prefill_is_partial=np.array([partial]),
            last_chunk_logits=None,
            req_id="request",
        )
        assert recorder.run_count == -(-n_tokens // target_width)
        state = proposer._req_state["request"]
        assert state.position_counter == n_tokens - 1
        assert state.candidates_from_prefill is (not partial)
        positions, hidden = [], []
        for inputs in recorder.input_calls:
            valid = inputs["position_ids_target"][0] >= 0
            positions.extend(inputs["position_ids_target"][0, valid])
            hidden.extend(inputs["target_hidden"][0, valid])
            assert inputs["position_ids"].shape == (4, 4)
            assert inputs["batch_index"][0, 0] == 3
            assert np.all(inputs["batch_index"][1:] == -1)
            assert np.all(inputs["position_ids_target"][1:] == -1)
            assert np.all(inputs["position_ids"][1:] == -1)
        np.testing.assert_array_equal(positions, np.arange(n_tokens))
        np.testing.assert_array_equal(hidden, features[0, :n_tokens])


def test_wide_decode_clears_padding_and_finished_request_state():
    proposer, _ = _make_proposer(4, 4)
    proposer.target_seq_len = 8
    proposer._dlm_position_ids_target = np.zeros((4, 8), dtype=np.int64)
    proposer._dlm_target_hidden = np.full((4, 8, 8), -123, dtype=np.float32)
    recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = recorder
    proposer._state_for("finished")
    proposer._state_for("active").position_counter = 20
    batch = _FakeInputBatch(["active"], [10], [15])
    features = np.ones((1, 4, 8), dtype=np.float32)
    proposer.propose(batch, [[10, 11]], np.array([2]), features)
    inputs = recorder.input_calls[0]
    assert "finished" not in proposer._req_state
    assert np.all(inputs["position_ids_target"][0, 2:] == -1)
    assert np.all(inputs["target_hidden"][0, 4:] == 0)
    assert np.all(inputs["target_hidden"][1:] == 0)
    assert np.all(inputs["position_ids_target"][1:] == -1)
    assert np.all(inputs["batch_index"][1:] == -1)


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("accepted", [1, 7, 15, 16])
@pytest.mark.parametrize("last_noise_pos", [4095, 4096, 4100])
@pytest.mark.parametrize("target_width", [16, 128])
def test_decode_context_boundary_keeps_short_requests_active(
    commit, accepted, last_noise_pos, target_width
):
    proposer, _ = _make_proposer(2, 16)
    proposer.target_seq_len = target_width
    proposer._dlm_position_ids_target = np.full((2, target_width), -1, dtype=np.int64)
    proposer._dlm_target_hidden = np.zeros((2, target_width, 8), dtype=np.float32)
    recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = recorder
    proposer._state_for("long").position_counter = last_noise_pos - 16 - accepted
    proposer._state_for("short").position_counter = 20
    batch = _FakeInputBatch(["long", "short"], [10, 10], [4080, 22])
    drafts = proposer.propose(
        batch,
        [list(range(accepted)), [7]],
        np.array([0, 1]),
        np.ones((2, 16, 8), dtype=np.float32),
        commit=commit,
    )
    fits = last_noise_pos < 4096
    inputs = recorder.input_calls[0]
    assert recorder.run_count == 1
    for name in ("position_ids", "position_ids_target"):
        assert np.all(
            (inputs[name] == -1) | ((inputs[name] >= 0) & (inputs[name] < 4096))
        )
        if not fits:
            assert np.all(inputs[name][0] == -1)
    assert inputs["batch_index"][0, 0] == (0 if fits else -1)
    assert inputs["batch_index"][1, 0] == 1
    assert proposer._req_state["long"].position_counter == last_noise_pos - 16
    assert proposer._req_state["short"].position_counter == 21
    assert len(drafts[0]) == (15 if fits and commit else 0)
    assert len(drafts[1]) == (15 if commit else 0)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("last_noise_pos", [4095, 4096, 4111, 4200])
@pytest.mark.parametrize("target_width", [16, 128])
def test_prefill_context_boundary_skips_unusable_dlm_work(
    partial, last_noise_pos, target_width
):
    proposer = _prep_prefill_proposer(2, 16, sub_blocks_per_chunk=8)
    proposer.target_seq_len = target_width
    proposer.num_sub_blocks = 128 // target_width
    proposer._dlm_position_ids_target = np.full((2, target_width), -1, dtype=np.int64)
    proposer._dlm_target_hidden = np.zeros((2, target_width, 8), dtype=np.float32)
    recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = recorder
    last_pos = last_noise_pos - 16
    proposer.prefill_step(
        target_hidden=[np.ones((1, 128, 8), dtype=np.float32)],
        prefill_positions=np.arange(last_pos - 127, last_pos + 1),
        prefill_cum_sum=np.array([128]),
        batch_indices=np.array([1]),
        prefill_is_partial=np.array([partial]),
        last_chunk_logits=None,
        req_id="long",
    )
    fits = last_noise_pos < 4096
    assert recorder.run_count == (128 // target_width if fits else 0)
    state = proposer._req_state["long"]
    assert state.position_counter == last_pos
    assert state.candidates_from_prefill == (fits and not partial)
    for inputs in recorder.input_calls:
        for name in ("position_ids", "position_ids_target"):
            assert np.all(
                (inputs[name] == -1) | ((inputs[name] >= 0) & (inputs[name] < 4096))
            )
    if not fits:
        assert state.dlm_candidates is None
        batch = _FakeInputBatch(["long"], [128], [last_pos + 2])
        assert proposer.propose(
            batch, [[7]], np.array([1]), np.ones((1, 16, 8), dtype=np.float32)
        ) == [[]]
        assert recorder.run_count == 0


@pytest.mark.parametrize("prompt_len", [4080, 4081, 4090, 4095])
@pytest.mark.parametrize("target_width", [16, 128])
def test_update_prefill_kv_checks_final_prompt_boundary(prompt_len, target_width):
    proposer = _prep_prefill_proposer(2, 16, sub_blocks_per_chunk=8)
    proposer.target_seq_len = target_width
    proposer.num_sub_blocks = 128 // target_width
    proposer._dlm_position_ids_target = np.full((2, target_width), -1, dtype=np.int64)
    proposer._dlm_target_hidden = np.zeros((2, target_width, 8), dtype=np.float32)
    recorder = _RecordingSession(proposer._dlm_logits_buf, proposer.vocab_size)
    proposer.model.session = recorder
    fits = prompt_len + proposer.block_size <= proposer.max_model_len

    for start in range(0, prompt_len, 128):
        end = min(start + 128, prompt_len)
        proposer.build_prefill_pending(
            np.array([end - start]),
            np.arange(start, end),
            np.array([1]),
            np.array([end < prompt_len]),
            None,
            [np.ones((1, 128, 8), dtype=np.float32)],
            ["request"],
        )
        calls_before = recorder.run_count
        proposer.update_prefill_kv()
        assert not proposer._prefill_pending
        if end == prompt_len and not fits:
            assert recorder.run_count == calls_before

    state = proposer._req_state["request"]
    assert state.position_counter == prompt_len - 1
    assert state.candidates_from_prefill == fits
    for inputs in recorder.input_calls:
        for name in ("position_ids", "position_ids_target"):
            assert np.all(
                (inputs[name] == -1) | ((inputs[name] >= 0) & (inputs[name] < 4096))
            )
    calls_before = recorder.run_count
    drafts = proposer.propose(
        _FakeInputBatch(["request"], [prompt_len], [prompt_len + 1]),
        [[7]],
        np.array([1]),
        np.ones((1, 16, 8), dtype=np.float32),
    )
    assert len(drafts[0]) == (15 if fits else 0)
    assert recorder.run_count == calls_before


@pytest.mark.parametrize("compiled_ctx", [2048, 4096, 8192])
@pytest.mark.parametrize("cache_bindings", [True, False])
@pytest.mark.parametrize("named_specialization", [True, False])
@pytest.mark.parametrize("target_width", [16, 128])
def test_dlm_context_limit_uses_loaded_cache_capacity(
    monkeypatch,
    tmp_path,
    compiled_ctx,
    cache_bindings,
    named_specialization,
    target_width,
):
    proposer, _ = _make_proposer(2, 16)
    proposer.tlm_prefill_seq_len = 128
    proposer._draft_vllm_config = SimpleNamespace(
        additional_config={}, model_config=SimpleNamespace(model="draft")
    )
    model = proposer.model
    model.kv_cache_info = (
        [
            ((2, 1, capacity, 8), np.float32, 0)
            for capacity in (compiled_ctx, compiled_ctx + 128)
        ]
        if cache_bindings
        else []
    )
    model.qpc_path = str(tmp_path / "qpc")
    symbols = {"batch_size": "2", "ctx_len": str(compiled_ctx)}
    specialization = (
        {"name": "Prefill", "symbols": symbols} if named_specialization else symbols
    )
    (tmp_path / "specializations.json").write_text(
        json.dumps({"specializations": [specialization]})
    )
    model.get_io_shape_and_dtype = lambda name: (
        [2, target_width, 8] if name == "target_hidden" else [2, target_width],
        np.float32,
        0,
    )
    monkeypatch.setattr(
        "vllm_qaic.spec_decode.dflash_draft_model.load_qaic_model", lambda *args: model
    )
    proposer.load_model()
    assert proposer.max_model_len == min(4096, compiled_ctx)
    assert proposer.target_seq_len == target_width


@pytest.mark.parametrize("width", [0, -16, 8, 24, 256])
def test_dlm_target_width_rejects_invalid_chunk_shapes(width):
    proposer, _ = _make_proposer(2, 16)
    proposer.tlm_prefill_seq_len = 128
    with pytest.raises(ValueError, match="DLM target width"):
        proposer._validate_target_width(width)


@pytest.mark.parametrize("method", ["dflash", "draft_model", "ngram", "suffix"])
@pytest.mark.parametrize("seq_len", [4080, 4081, 4082, None])
def test_runner_context_gate_reserves_dflash_bonus_slot(method, seq_len):
    from vllm_qaic.worker.model_runner import QaicModelRunnerAoT

    calls = []
    metadata = SimpleNamespace(max_seq_len=seq_len) if seq_len is not None else None
    scheduler = SimpleNamespace(total_num_scheduled_tokens=1)
    runner = SimpleNamespace(
        kv_connector_output=None,
        use_async_scheduling=False,
        is_kv_producer=False,
        execute_model_state=(scheduler, None, None, metadata) + (None,) * 6,
        _sample=lambda *args: None,
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=None, sampling_metadata=None
        ),
        speculative_config=SimpleNamespace(
            method=method, use_dflash=lambda: method == "dflash"
        ),
        num_spec_tokens=15,
        effective_drafter_max_model_len=4096,
        _bookkeeping_sync=lambda *args: (0, None, [[7]], {}, ["r"], {"r": 0}, []),
        model=SimpleNamespace(),
        batch_indices=np.array([0]),
        _tlm_hidden_buf=None,
        drafter=SimpleNamespace(
            propose=lambda *args, commit=True: calls.append(commit),
        ),
        propose_draft_token_ids=lambda *args: calls.append(True) or [[]],
        _copy_draft_token_ids_to_cpu=lambda *args: None,
    )
    QaicModelRunnerAoT.sample_tokens(runner, None)
    expected = seq_len is not None and seq_len + 15 + (method == "dflash") <= 4096
    assert calls.count(True) == int(expected)
    if method == "dflash":
        assert calls == [expected]
