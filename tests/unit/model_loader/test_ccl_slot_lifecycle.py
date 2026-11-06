# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Host regressions for disaggregated prefill CCL slot ownership."""

from queue import Queue
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from vllm_qaic.model_loader import qaic, qaic_session_np
from vllm_qaic.model_loader.qaic import QaicCausalLM
from vllm_qaic.worker.model_runner import QaicModelRunnerAoT

CCL_BUCKETS = [2048, 8192, 65536]


class _Session:
    def __init__(self, cluster_id):
        self.decode_num_execObj = 0 if cluster_id == "prefill" else 1
        self.prefill_num_execObj = 0 if cluster_id == "decode" else 3
        self.prefill_available_exec_objs = Queue()
        for slot in range(
            self.decode_num_execObj,
            self.decode_num_execObj + self.prefill_num_execObj,
        ):
            self.prefill_available_exec_objs.put(slot)
        self.input_names = ["batch_index"]
        self.submitted_ccls = []
        self.completed = []
        self.fail_completion = False

    def get_logits_ndim(self):
        return 3

    def np_run_pipeline(self, inputs, last_chunk, kv_cache_buffers):
        slot = self.prefill_available_exec_objs.get(timeout=1)
        ccl = inputs.get("comp_ctx_lengths")
        self.submitted_ccls.append(None if ccl is None else ccl.size)
        return slot

    def complete_inf(self, index, is_prefill):
        if self.fail_completion:
            raise RuntimeError("inference failed")
        self.completed.append((index, is_prefill))
        if is_prefill:
            self.prefill_available_exec_objs.put(index)


@pytest.fixture
def make_model(monkeypatch):
    monkeypatch.setattr(qaic.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        qaic_session_np,
        "QAICInferenceSession",
        lambda *args, **kwargs: _Session(kwargs["cluster_id"]),
    )

    def create(role="kv_producer", ccl=True, async_scheduling=True):
        model = SimpleNamespace(
            full_batch_size=4,
            use_async_scheduling=async_scheduling,
            prefill_bsz=1,
            decode_bsz=4,
            prefill_seq_len=256,
            vocab_size=8,
            is_multimodal_model=False,
            is_dflash_target=False,
            get_io_shape_and_dtype=lambda *args, **kwargs: None,
            get_comp_ctx_lengths=lambda: (
                (CCL_BUCKETS, CCL_BUCKETS) if ccl else (None, None)
            ),
            last_decode=False,
            uses_mrope=False,
            pad=np.full(256, -1, dtype=np.int64),
            async_scheduling_exec_timeout=1,
        )
        QaicCausalLM.load_model(
            model, "unused.qpc", [0], kv_transfer_role=role, stages=2
        )
        model.complete_inf = MethodType(QaicCausalLM.complete_inf, model)
        model.list_of_comp_ctx_lengths = (
            {bucket: np.empty(bucket, dtype=np.int64) for bucket in CCL_BUCKETS}
            if ccl
            else None
        )
        return model

    return create


def _submit(model, prompt_lengths, scheduled_lengths=None, offsets=None):
    lengths = prompt_lengths if scheduled_lengths is None else scheduled_lengths
    offsets = [0] * len(lengths) if offsets is None else offsets
    pending = Queue()
    QaicCausalLM._run_pipeline_prefill(
        model,
        input_ids=np.arange(sum(lengths), dtype=np.int64),
        positions=np.concatenate(
            [
                np.arange(offset, offset + length, dtype=np.int64)
                for offset, length in zip(offsets, lengths, strict=True)
            ]
        ),
        batch_indices=np.arange(len(lengths), dtype=np.int64),
        prefill_cum_sum=np.cumsum(lengths),
        pending_exec_queue=pending,
        kv_caches=[[] for _ in lengths],
        prefill_is_partial=[
            offset + length < total
            for offset, length, total in zip(
                offsets, lengths, prompt_lengths, strict=True
            )
        ],
        num_prompt_tokens_prefill=np.asarray(prompt_lengths),
    )
    return pending


def _drain(model, pending):
    runner = SimpleNamespace(model=model, is_kv_consumer=False)
    QaicModelRunnerAoT.complete_all_inf(runner, pending, 0)


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer"])
@pytest.mark.parametrize("ccl", [False, True])
def test_tracking_exists_only_for_ccl_prefill_producers(make_model, role, ccl):
    model = make_model(role=role, ccl=ccl)
    assert hasattr(model, "active_ccl") == (role == "kv_producer" and ccl)
    if hasattr(model, "active_ccl"):
        assert model.active_ccl == {0: 0, 1: 0, 2: 0}


def test_async_long_prompt_does_not_pin_following_short_prompt(make_model):
    model = make_model()
    _drain(model, _submit(model, [60000]))
    assert model.active_ccl == {0: 0, 1: 0, 2: 0}
    first_short_chunk = len(model.session.submitted_ccls)
    _drain(model, _submit(model, [2000]))
    assert model.session.submitted_ccls[first_short_chunk:] == [2048] * 8


def test_other_live_slots_still_constrain_next_batch(make_model):
    model = make_model()
    first = _submit(model, [60000], [256])
    second = _submit(model, [2000], [256])
    _drain(model, first)
    third = _submit(model, [2000], [256])
    assert model.session.submitted_ccls == [65536, 65536, 65536]
    _drain(model, second)
    _drain(model, third)
    assert model.active_ccl == {0: 0, 1: 0, 2: 0}
    _drain(model, _submit(model, [2000], [256]))
    assert model.session.submitted_ccls[-1] == 2048


def test_partial_long_prompt_reselects_its_total_length(make_model):
    model = make_model()
    _drain(model, _submit(model, [60000], [256]))
    assert model.active_ccl == {0: 0, 1: 0, 2: 0}
    _drain(model, _submit(model, [60000], [256], [256]))
    assert model.session.submitted_ccls == [65536, 65536]


def test_mixed_batch_keeps_one_bucket_through_slot_reuse(make_model):
    model = make_model()
    _drain(model, _submit(model, [2000, 60000], [257, 513]))
    assert model.session.submitted_ccls == [65536] * 5
    assert model.active_ccl == {0: 0, 1: 0, 2: 0}


@pytest.mark.parametrize("ccl", [False, True])
def test_sync_pipeline_completion_needs_no_extra_flag(make_model, ccl):
    model = make_model(ccl=ccl, async_scheduling=False)
    pending = _submit(model, [2000])
    assert pending.empty()
    assert model.session.prefill_available_exec_objs.qsize() == 3
    if ccl:
        assert model.active_ccl == {0: 0, 1: 0, 2: 0}


def test_async_pipeline_without_ccl_needs_no_tracking(make_model):
    model = make_model(ccl=False)
    _drain(model, _submit(model, [2000]))
    assert not hasattr(model, "active_ccl")
    assert model.session.submitted_ccls == [None] * 8


@pytest.mark.parametrize("role", [None, "kv_consumer"])
def test_nonproducer_completion_does_not_access_tracking(make_model, role):
    model = make_model(role=role)
    model.complete_inf(1, True)
    assert not hasattr(model, "active_ccl")
    assert model.session.completed == [(1, True)]


def test_decode_completion_does_not_clear_prefill_tracking(make_model):
    model = make_model()
    model.active_ccl[0] = 65536
    model.complete_inf(0, False)
    assert model.active_ccl[0] == 65536
    assert model.session.completed == [(0, False)]


def test_failed_completion_keeps_live_slot_ccl(make_model):
    model = make_model()
    pending = _submit(model, [60000], [256])
    model.session.fail_completion = True
    with pytest.raises(RuntimeError, match="inference failed"):
        _drain(model, pending)
    assert model.active_ccl[0] == 65536
    assert model.session.prefill_available_exec_objs.qsize() == 2


def test_ccl_is_cleared_only_after_session_completion(make_model, monkeypatch):
    model = make_model()
    pending = _submit(model, [60000], [256])
    original_complete = model.session.complete_inf

    def complete(index, is_prefill):
        assert model.active_ccl[index] == 65536
        original_complete(index, is_prefill)

    monkeypatch.setattr(model.session, "complete_inf", complete)
    _drain(model, pending)
    assert model.active_ccl[0] == 0
