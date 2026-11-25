# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""CPU tests for AOT worker sampling startup and guided-decoding masks."""

import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_qaic.v1.sample as sample_package
import vllm_qaic.worker.worker as worker_module
from vllm_qaic.model_loader.qaic import QaicCausalLM
from vllm_qaic.worker.model_runner import (
    QaicModelRunnerAoT,
    _producer_placeholder_tokens,
)
from vllm_qaic.worker.worker import QaicWorkerAoT


def _worker(speculative_config):
    worker = object.__new__(QaicWorkerAoT)
    worker.speculative_config = speculative_config
    return worker


def _init_worker(speculative_config):
    worker = _worker(speculative_config)
    worker.device_config = SimpleNamespace(device=torch.device("cpu"))
    worker.vllm_config = SimpleNamespace(
        additional_config={"device_group": [19]},
    )
    worker.rank = 0
    worker.local_rank = 0
    worker.distributed_init_method = ""
    worker.model_config = SimpleNamespace(seed=0)
    return worker


def _stub_init_device(monkeypatch, worker, runner):
    monkeypatch.setattr(
        worker,
        "_init_qaic_worker_distributed_environment",
        lambda *args: None,
    )
    monkeypatch.setattr(worker, "_configure_thread_parallelism", lambda: None)
    monkeypatch.setattr(worker_module, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(
        worker_module, "QaicModelRunnerAoT", lambda config, device: runner
    )
    monkeypatch.setitem(
        sys.modules,
        "qaicrt",
        SimpleNamespace(
            Util=lambda: SimpleNamespace(getDeviceIds=lambda: (None, [19]))
        ),
    )


def test_no_spec_does_not_import_or_require_numba(monkeypatch):
    real_import = builtins.__import__

    def reject_numba_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_qaic.v1.sample" and "rejection_sampler_numba" in fromlist:
            raise AssertionError("Numba rejection sampler imported for no-spec AOT")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_numba_import)

    _worker(None)._install_aot_rejection_sampler()


def test_no_spec_init_device_does_not_import_numba(monkeypatch):
    real_import = builtins.__import__

    def reject_numba_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_qaic.v1.sample" and "rejection_sampler_numba" in fromlist:
            raise AssertionError("Numba rejection sampler imported for no-spec AOT")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_numba_import)
    runner = SimpleNamespace()
    worker = _init_worker(None)
    _stub_init_device(monkeypatch, worker, runner)

    QaicWorkerAoT.init_device(worker)

    assert worker.model_runner is runner


def test_speculative_init_device_installs_numba(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(install=lambda: calls.append("install"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )
    worker = _init_worker(SimpleNamespace(method="ngram"))
    _stub_init_device(monkeypatch, worker, SimpleNamespace())

    QaicWorkerAoT.init_device(worker)

    assert calls == ["install"]


def test_speculative_aot_installs_numba(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(install=lambda: calls.append("install"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )

    _worker(SimpleNamespace(method="ngram"))._install_aot_rejection_sampler()

    assert calls == ["install"]


def test_no_spec_skips_numba_prewarm_and_drafter_warmup(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(prewarm=lambda: calls.append("prewarm"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )
    worker = _worker(None)
    worker.model_runner = SimpleNamespace(
        _qaic_warm_up_drafter=lambda: calls.append("drafter"),
        _qaic_dummy_run=lambda: calls.append("dummy"),
    )

    worker.compile_or_warm_up_model()

    assert calls == ["dummy"]


def test_speculative_aot_prewarms_before_drafter_and_dummy_run(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(prewarm=lambda: calls.append("prewarm"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )
    worker = _worker(SimpleNamespace(method="ngram"))
    worker.model_runner = SimpleNamespace(
        _qaic_warm_up_drafter=lambda: calls.append("drafter"),
        _qaic_dummy_run=lambda: calls.append("dummy"),
    )

    worker.compile_or_warm_up_model()

    assert calls == ["prewarm", "drafter", "dummy"]


@pytest.mark.parametrize("method", ["suffix", "draft_model", "dflash"])
def test_drafter_warmup_gating_for_non_ngram_and_missing_drafter(method):
    # The model-runner helper itself remains the safe boundary: only an active
    # ngram config with a drafter reaches the Numba proposer warm-up.
    from vllm_qaic.spec_decode.ngram_warmup import maybe_warm_up_ngram_proposer

    spec = SimpleNamespace(method=method)
    assert maybe_warm_up_ngram_proposer(spec, object(), object()) is None
    assert (
        maybe_warm_up_ngram_proposer(SimpleNamespace(method="ngram"), None, object())
        is None
    )


WIDTH = 40  # not a multiple of 32, so the padded tail is exercised


class _FakeRunner:
    _ods_token_bitmasks = QaicModelRunnerAoT._ods_token_bitmasks
    _add_ods_token_bitmasks = QaicModelRunnerAoT._add_ods_token_bitmasks

    def __init__(self, req_ids, num_decodes, is_kv_consumer=False, decode_bsz=2):
        self.input_batch = SimpleNamespace(
            num_reqs=len(req_ids),
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
        )
        self.model = SimpleNamespace(
            token_bitmask_width=WIDTH,
            token_bitmask_dtype=np.int8,
            decode_bsz=decode_bsz,
        )
        self.num_decodes = num_decodes
        self.is_kv_consumer = is_kv_consumer


def _grammar(req_ids, rows):
    return SimpleNamespace(
        structured_output_request_ids=req_ids,
        grammar_bitmask=np.array(rows, dtype=np.int32),
    )


def test_masks_map_to_slots_and_unpack_lsb_first():
    runner = _FakeRunner(["a", "b", "c"], num_decodes=1)
    # Rows follow structured_output_request_ids, not batch order.
    masks = runner._ods_token_bitmasks(_grammar(["c", "a"], [[0b101, 0], [-1, 1]]))

    assert masks.shape == (3, WIDTH)
    assert masks.dtype == np.int8
    assert np.array_equal(np.flatnonzero(masks[0]), np.arange(33))  # "a"
    assert masks[1].all()  # "b" has no grammar, so it stays unconstrained
    assert np.array_equal(np.flatnonzero(masks[2]), [0, 2])  # "c"


def test_no_grammar_leaves_every_slot_unconstrained():
    masks = _FakeRunner(["a", "b"], num_decodes=2)._ods_token_bitmasks(None)
    assert masks.shape == (2, WIDTH)
    assert masks.all()


def test_decode_and_prefill_params_follow_batch_layout():
    runner = _FakeRunner(["a", "b", "c"], num_decodes=1, decode_bsz=2)
    decode, prefill = {}, {}
    runner._add_ods_token_bitmasks(decode, prefill, _grammar(["c"], [[0b1, 0]]))

    # Decode row 0 is slot 0 and is padded to decode_bsz like the other params.
    assert decode["token_bitmasks"].shape == (2, WIDTH)
    assert decode["token_bitmasks"][0].all()
    # Prefill rows are the non-decode slots in batch order: "b" then "c".
    assert prefill["token_bitmasks"].shape == (2, WIDTH)
    assert prefill["token_bitmasks"][0].all()
    assert np.array_equal(np.flatnonzero(prefill["token_bitmasks"][1]), [0])


def test_disagg_consumer_decode_rows_cover_every_request():
    runner = _FakeRunner(["a", "b"], num_decodes=0, is_kv_consumer=True)
    decode, prefill = {}, {}
    runner._add_ods_token_bitmasks(decode, prefill, _grammar(["b"], [[0b10, 0]]))

    assert decode["token_bitmasks"][0].all()
    assert np.array_equal(np.flatnonzero(decode["token_bitmasks"][1]), [1])


def test_producer_placeholder_is_lowest_grammar_allowed_token():
    req_to_slot = {"a": 0, "b": 1}
    grammar = _grammar(["b"], [[0, 0b1]])  # only token 32 allowed (second word)
    tokens = _producer_placeholder_tokens(grammar, req_to_slot, 2, vocab_size=40)
    assert tokens.shape == (2, 1)
    assert tokens[0, 0] == 0  # unstructured request keeps the zero placeholder
    assert tokens[1, 0] == 32


def test_producer_placeholder_without_grammar_is_zero():
    tokens = _producer_placeholder_tokens(None, {"a": 0}, 1, vocab_size=40)
    assert tokens.tolist() == [[0]]


class _SamplingRunner:
    _cache_ods_sampling_params = QaicModelRunnerAoT._cache_ods_sampling_params
    _prepare_ods_sampling_params = QaicModelRunnerAoT._prepare_ods_sampling_params

    def __init__(self, num_decodes=1, is_kv_consumer=False):
        self.model = SimpleNamespace(decode_bsz=3, max_top_k_ids=4)
        self.num_decodes = num_decodes
        self.is_kv_consumer = is_kv_consumer
        self._ods_min_ps = {}
        self._ods_sampling_buffers = {}
        self._ods_decode_buffers = {}
        self._ods_sampling_layout = None
        self.input_ids = SimpleNamespace(cpu=torch.tensor([11, 12]))
        self.input_batch = SimpleNamespace(
            num_reqs=2,
            req_ids=["a", "b"],
            sampling_metadata=SimpleNamespace(generators={}),
            temperature_cpu=np.empty(3, dtype=np.float32),
            top_p_cpu=np.empty(3, dtype=np.float32),
            top_k_cpu=np.empty(3, dtype=np.int32),
            repetition_penalties_cpu=np.empty(3, dtype=np.float32),
            presence_penalties_cpu=np.empty(3, dtype=np.float32),
            num_computed_tokens_cpu=np.array([4, 5]),
            num_prompt_tokens=np.array([5, 5]),
        )
        self._requests = {}
        self._generators = {
            "a": torch.Generator().manual_seed(10),
            "b": torch.Generator().manual_seed(20),
        }
        self.arrive(_sampling_request("a", 0.0), _sampling_request("b", 0.7, top_k=9))

    def set_requests(self, req_ids):
        # Mirror InputBatch's request-arrival/reorder updates to its CPU buffers.
        self.input_batch.req_ids = req_ids
        self.input_batch.num_reqs = len(req_ids)
        for slot, req_id in enumerate(req_ids):
            params = self._requests[req_id].sampling_params
            self.input_batch.temperature_cpu[slot] = params.temperature
            self.input_batch.top_p_cpu[slot] = params.top_p
            self.input_batch.top_k_cpu[slot] = (
                params.top_k if params.top_k > 0 else WIDTH
            )
            self.input_batch.repetition_penalties_cpu[slot] = params.repetition_penalty
            self.input_batch.presence_penalties_cpu[slot] = params.presence_penalty
            self.input_batch.sampling_metadata.generators[slot] = self._generators[
                req_id
            ]

    def arrive(self, *requests, finished=()):
        self._requests.update({req.req_id: req for req in requests})
        self.set_requests(self.input_batch.req_ids)
        self._cache_ods_sampling_params(
            SimpleNamespace(scheduled_new_reqs=requests, finished_req_ids=finished)
        )


def _sampling_request(req_id, temperature, top_k=-1):
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SimpleNamespace(
            temperature=temperature,
            top_k=top_k,
            top_p=0.9,
            min_p=0.1,
            repetition_penalty=1.2,
            presence_penalty=0.3,
        ),
    )


def test_ods_sampling_buffers_are_reused_and_rng_advances():
    runner = _SamplingRunner()
    expected_rng = torch.Generator().manual_seed(10)
    decode, prefill = runner._prepare_ods_sampling_params()
    first_draw = decode["random_numbers"][0].copy()
    np.testing.assert_array_equal(
        first_draw, torch.rand(4, generator=expected_rng).numpy()
    )
    buffers = {key: id(value) for key, value in decode.items()}
    assert decode["top_ks"].tolist() == [[4], [4], [4]]
    assert np.isclose(prefill["temperatures"][0, 0], 0.7)
    assert np.shares_memory(prefill["temperatures"], runner.input_batch.temperature_cpu)

    second_decode, _ = runner._prepare_ods_sampling_params()
    assert {key: id(value) for key, value in second_decode.items()} == buffers
    np.testing.assert_array_equal(
        second_decode["random_numbers"][0],
        torch.rand(4, generator=expected_rng).numpy(),
    )
    assert not np.array_equal(first_draw, second_decode["random_numbers"][0])


def test_ods_forward_does_not_reallocate_parameter_arrays(monkeypatch):
    runner = _SamplingRunner()
    runner._prepare_ods_sampling_params()

    def reject_allocation(*args, **kwargs):
        raise AssertionError("sampling arrays must not be rebuilt on each forward")

    for name in ("asarray", "empty", "empty_like", "resize", "stack"):
        monkeypatch.setattr(np, name, reject_allocation)
    runner._prepare_ods_sampling_params()


def test_ods_cache_tracks_request_reorder_and_decode_prefill_split():
    runner = _SamplingRunner()
    runner._prepare_ods_sampling_params()
    runner.set_requests(["b", "a"])
    decode, prefill = runner._prepare_ods_sampling_params()
    assert np.allclose(decode["temperatures"], 0.7)
    assert prefill["temperatures"].tolist() == [[0.0]]

    runner.num_decodes = 2
    decode, prefill = runner._prepare_ods_sampling_params()
    assert np.allclose(decode["temperatures"].ravel(), [0.7, 0.0, 0.7])
    assert prefill["temperatures"].shape == (0, 1)


def test_ods_cache_survives_preemption_and_evicts_finished_requests():
    runner = _SamplingRunner()
    cached_b = runner._ods_min_ps["b"]
    runner.set_requests(["a"])
    runner._prepare_ods_sampling_params()
    runner.arrive()  # preempted request "b" is not finished
    assert runner._ods_min_ps["b"] == cached_b

    runner.set_requests(["b"])
    decode, _ = runner._prepare_ods_sampling_params()
    assert np.allclose(decode["temperatures"], 0.7)
    runner.arrive(finished=("b",))
    assert "b" not in runner._ods_min_ps
    runner.arrive(_sampling_request("b", 0.2))
    decode, _ = runner._prepare_ods_sampling_params()
    assert np.allclose(decode["temperatures"], 0.2)


def test_ods_consumer_accepted_tokens_refresh_in_place():
    runner = _SamplingRunner(num_decodes=0, is_kv_consumer=True)
    decode, _ = runner._prepare_ods_sampling_params()
    accepted = decode["last_accepted_output_tokens"]
    sentinel = np.iinfo(np.int32).max
    assert accepted.tolist() == [[sentinel], [12], [sentinel]]

    runner.input_batch.num_computed_tokens_cpu[:] = 5
    runner.input_ids.cpu[:] = torch.tensor([13, 14])
    decode, _ = runner._prepare_ods_sampling_params()
    assert decode["last_accepted_output_tokens"] is accepted
    assert accepted.tolist() == [[13], [14], [sentinel]]


@pytest.mark.parametrize("is_prompt, route", [(True, "prefill"), (False, "decode")])
def test_forward_keeps_sampling_and_paged_inputs(is_prompt, route):
    calls = []
    model = SimpleNamespace(
        is_pooling_model=False,
        is_multimodal_model=False,
        disagg_producer_en=False,
        disagg_serving_en=False,
        _run_prefill=lambda *a, **kw: calls.append(("prefill", kw)),
        _run_decode=lambda *a, **kw: calls.append(("decode", kw)),
    )
    sampling, table, slots = {}, np.zeros((1, 2)), np.zeros(1)
    QaicCausalLM.forward(
        model,
        [],
        [],
        [],
        is_prompt,
        sampling_params=sampling,
        block_table=table,
        slot_id=slots,
    )
    assert len(calls) == 1 and calls[0][0] == route
    kwargs = calls[0][1]
    assert kwargs["sampling_params"] is sampling
    assert kwargs["block_table"] is table
    assert kwargs["slot_id"] is slots
