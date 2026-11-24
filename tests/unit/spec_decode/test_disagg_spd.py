# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Device-independent behavior tests for disaggregated speculative decoding."""

from contextlib import nullcontext
from copy import deepcopy
import json
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import vllm_qaic.model_loader.qaic as qaic_model_loader
from vllm_qaic.distributed.kv_transfer import kv_connector as kv_connector_module
from vllm_qaic.model_loader.qaic import QaicCausalLM
from vllm_qaic.worker.model_runner import QaicModelRunnerAoT


@pytest.fixture
def runner_config():
    def make(method=None, role=None, num_speculative_tokens=3):
        speculative_config = None
        if method is not None:
            speculative_config = MagicMock()
            speculative_config.method = method
            speculative_config.draft_model_config = MagicMock()
            speculative_config.num_speculative_tokens = num_speculative_tokens
            speculative_config.uses_draft_model.return_value = method == "draft_model"
            speculative_config.use_dflash.return_value = method == "dflash"
            speculative_config.uses_dynamic_speculative_decoding.return_value = False
        kv_transfer_config = None
        if role is not None:
            kv_transfer_config = SimpleNamespace(kv_role=role)
        return SimpleNamespace(
            speculative_config=speculative_config,
            kv_transfer_config=kv_transfer_config,
            scheduler_config=SimpleNamespace(max_num_seqs=2, async_scheduling=False),
            additional_config={},
        )

    return make


def _patch_runner_init():
    def fake_init(self, config, device):
        self.vllm_config = config
        self.speculative_config = config.speculative_config
        self.use_async_scheduling = False
        self.drafter = MagicMock(name="parent_drafter")
        self.model_config = SimpleNamespace(
            get_num_kv_heads=lambda _parallel_config: 1,
            get_head_size=lambda: 64,
        )
        self.parallel_config = SimpleNamespace()
        self.num_spec_tokens = (
            0
            if config.speculative_config is None
            else config.speculative_config.num_speculative_tokens
        )
        self.max_num_reqs = config.scheduler_config.max_num_seqs
        self.lora_config = None
        self.uses_mrope = False

    return (
        patch(
            "vllm.v1.worker.gpu_model_runner.GPUModelRunner.__init__",
            fake_init,
        ),
        patch.object(QaicModelRunnerAoT, "_postprocess_tensors"),
    )


def test_kv_producer_constructor_clears_drafter(runner_config):
    config = runner_config("ngram", "kv_producer")
    init_patch, postprocess_patch = _patch_runner_init()
    with init_patch, postprocess_patch:
        runner = QaicModelRunnerAoT(config, torch.device("cpu"))

    assert runner.is_kv_producer
    assert runner.drafter is None


@pytest.mark.parametrize(
    ("method", "role", "expected_decode_ks"),
    [
        ("ngram", None, [0, 3]),
        ("ngram", "kv_producer", [0, 3]),
        ("ngram", "kv_consumer", [0, 3]),
        ("suffix", "kv_consumer", [0, 3]),
        ("draft_model", None, [3]),
        ("draft_model", "kv_producer", [3]),
        ("draft_model", "kv_consumer", [0, 3]),
    ],
)
def test_constructor_selects_decode_specializations(
    runner_config, method, role, expected_decode_ks
):
    config = runner_config(method, role)
    init_patch, postprocess_patch = _patch_runner_init()
    with (
        init_patch,
        postprocess_patch,
        patch("vllm.config.utils.replace", side_effect=lambda value, **_: value),
        patch(
            "vllm_qaic.spec_decode.qaic_draft_model.QaicDraftModelProposer",
            return_value=MagicMock(),
        ),
    ):
        runner = QaicModelRunnerAoT(config, torch.device("cpu"))

    assert runner.decode_ks == expected_decode_ks


def test_constructor_rejects_unsupported_speculative_method(runner_config):
    config = runner_config("eagle", "kv_consumer")
    init_patch, postprocess_patch = _patch_runner_init()
    with (
        init_patch,
        postprocess_patch,
        pytest.raises(ValueError, match="not yet supported"),
    ):
        QaicModelRunnerAoT(config, torch.device("cpu"))


@pytest.mark.parametrize(
    ("num_decodes", "scheduled_tokens", "expected"),
    [
        (0, None, 0),
        (1, {}, 0),
        (2, {"request-0": [1, 2, 3]}, 7),
    ],
)
def test_determine_active_k(num_decodes, scheduled_tokens, expected):
    runner = SimpleNamespace(decode_ks=[0, 7], num_decodes=num_decodes)
    scheduler_output = None
    if scheduled_tokens is not None:
        scheduler_output = SimpleNamespace(
            scheduled_spec_decode_tokens=scheduled_tokens
        )

    assert QaicModelRunnerAoT._determine_active_k(runner, scheduler_output) == expected


def test_determine_active_k_uses_single_specialization():
    runner = SimpleNamespace(decode_ks=[3], num_decodes=0)

    assert QaicModelRunnerAoT._determine_active_k(runner, None) == 3


def _scheduler_output(new, cached, resumed, num_scheduled, spec):
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    cached_reqs = CachedRequestData.make_empty()
    cached_reqs.req_ids = list(cached)
    cached_reqs.resumed_req_ids = set(resumed)
    return SchedulerOutput(
        scheduled_new_reqs=[SimpleNamespace(req_id=r) for r in new],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens=spec,
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_strip_waiting_spec_padding_unpads_new_and_resumed_requests():
    # vLLM >= 0.30 pads a waiting request with one remaining token (consumer
    # after KV transfer) to 1 + K tokens with -1 drafts. Only requests from the
    # waiting queue are stripped; running requests keep whatever drafts they
    # were scheduled with.
    spec = {
        "new": [-1, -1, -1],
        "resumed": [-1, -1, -1],
        "running": [5, 6, 7],
        "running_all_pad": [-1, -1, -1],
    }
    num_scheduled = {"new": 4, "resumed": 4, "running": 4, "running_all_pad": 4}
    original = _scheduler_output(
        ["new"],
        ["resumed", "running", "running_all_pad"],
        ["resumed"],
        num_scheduled,
        spec,
    )

    out = QaicModelRunnerAoT._strip_waiting_spec_padding(original)

    assert out.num_scheduled_tokens == {
        "new": 1,
        "resumed": 1,
        "running": 4,
        "running_all_pad": 4,
    }
    assert out.total_num_scheduled_tokens == 10
    assert out.scheduled_spec_decode_tokens == {
        "running": [5, 6, 7],
        "running_all_pad": [-1, -1, -1],
    }
    # The scheduler's own object is untouched (shared in-process); it rolls
    # back the three padded drafts as rejected.
    assert original.num_scheduled_tokens == num_scheduled
    assert original.total_num_scheduled_tokens == 16
    assert set(original.scheduled_spec_decode_tokens) == set(spec)

    # Without padding the scheduler output is passed through as is.
    unpadded = _scheduler_output(
        ["new"], ["running"], [], {"new": 25, "running": 4}, {"running": [1, 2, 3]}
    )
    assert QaicModelRunnerAoT._strip_waiting_spec_padding(unpadded) is unpadded


def test_execute_model_runs_on_stripped_scheduler_output():
    # The strip must be applied before the runner reads the scheduler output.
    import contextlib

    class _Stop(Exception):
        pass

    def _update_states(scheduler_output):
        raise _Stop(scheduler_output)

    runner = object.__new__(QaicModelRunnerAoT)
    runner.execute_model_state = None
    runner.model = SimpleNamespace(is_vision_encoder=False)
    runner.synchronize_input_prep = contextlib.nullcontext
    runner._update_states = _update_states
    padded = _scheduler_output(["new"], [], [], {"new": 4}, {"new": [-1, -1, -1]})

    with pytest.raises(_Stop) as stop:
        runner.execute_model(padded)

    (seen,) = stop.value.args
    assert seen.num_scheduled_tokens == {"new": 1}
    assert seen.scheduled_spec_decode_tokens == {}


def test_grammar_bitmask_skips_stripped_padding_rows():
    # The scheduler builds 1 + K bitmask rows for the padded request; after the
    # strip it has one logit row, so later requests must keep their own masks.
    pytest.importorskip("xgrammar")
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.structured_output.utils import apply_grammar_bitmask

    out = QaicModelRunnerAoT._strip_waiting_spec_padding(
        _scheduler_output(
            ["padded"],
            ["other"],
            [],
            {"padded": 4, "other": 2},
            {"padded": [-1, -1, -1], "other": [7]},
        )
    )
    # Each scheduler row allows a single token: padded 1..4, other 10..11.
    allowed = [1, 2, 3, 4, 10, 11]
    bitmask = np.array([[1 << t] for t in allowed], dtype=np.int32)
    grammar_output = GrammarOutput(["padded", "other"], bitmask)
    logits = torch.zeros(3, 32)

    apply_grammar_bitmask(
        out,
        QaicModelRunnerAoT._unpad_grammar_output(out, grammar_output),
        SimpleNamespace(req_ids=["padded", "other"]),
        logits,
    )

    assert torch.isfinite(logits).sum(dim=1).tolist() == [1, 1, 1]
    assert logits.argmax(dim=1).tolist() == [1, 10, 11]

    # Without stripped padding the bitmask is passed through as is.
    unpadded = _scheduler_output([], ["r"], [], {"r": 2}, {"r": [7]})
    unpadded_grammar = GrammarOutput(["r"], bitmask[:2])
    assert (
        QaicModelRunnerAoT._unpad_grammar_output(unpadded, unpadded_grammar)
        is unpadded_grammar
    )


@pytest.mark.parametrize("role", ["kv_consumer", "kv_producer"])
def test_connector_meta_ignores_waiting_spec_padding(role):
    # The padded drafts must not count as prompt tokens: otherwise the
    # consumer's first step is flagged as a partial prefill, the request stays
    # tracked and the next step reloads an already consumed KV handoff.
    from vllm_qaic.distributed.kv_transfer.kv_connector.v1.qaic_connector import (
        QaicConnector,
        ReqTrackerObj,
    )

    prompt = list(range(8))
    connector = object.__new__(QaicConnector)
    connector.is_producer = role == "kv_producer"
    connector.disable_handoff_id = True
    connector._request_tracker = {
        r: ReqTrackerObj(
            SimpleNamespace(
                prompt_token_ids=prompt, num_prompt_tokens=8, kv_transfer_params=None
            )
        )
        for r in ("padded", "chunk")
    }
    out = _scheduler_output(
        ["padded", "chunk"], [], [], {"padded": 4, "chunk": 4}, {"padded": [-1] * 3}
    )
    out.scheduled_new_reqs = [
        SimpleNamespace(
            req_id="padded",
            num_computed_tokens=7,
            prompt_token_ids=prompt,
            block_ids=[[1]],
        ),
        # A genuine prefill chunk stays partial.
        SimpleNamespace(
            req_id="chunk",
            num_computed_tokens=0,
            prompt_token_ids=prompt,
            block_ids=[[2]],
        ),
    ]

    meta = connector.build_connector_meta(out)

    padded, chunk = meta.requests
    assert not padded.is_prefill_partial
    assert padded.is_store == connector.is_producer
    assert chunk.is_prefill_partial and not chunk.is_store
    assert set(connector._request_tracker) == {"chunk"}

    # Next step: the chunked request finishes its prompt with no drafts, and a
    # padded request resumed after preemption (cached branch) is complete too.
    connector._request_tracker["resumed"] = ReqTrackerObj(
        SimpleNamespace(
            prompt_token_ids=prompt, num_prompt_tokens=8, kv_transfer_params=None
        ),
        block_id=3,
    )
    out = _scheduler_output(
        [],
        ["chunk", "resumed"],
        ["resumed"],
        {"chunk": 4, "resumed": 4},
        {"resumed": [-1] * 3},
    )
    out.scheduled_cached_reqs.num_computed_tokens = [4, 7]
    done, resumed = connector.build_connector_meta(out).requests
    assert not done.is_prefill_partial
    assert not resumed.is_prefill_partial
    assert resumed.is_store == connector.is_producer
    assert connector._request_tracker == {}


@pytest.mark.parametrize(
    ("is_kv_consumer", "num_decodes", "num_reqs", "expected"),
    [
        (False, 2, 3, 2),
        (True, 2, 3, 3),
        (True, 0, 1, 1),
    ],
)
def test_num_decode_rows_to_pad_handles_mixed_consumer_batches(
    is_kv_consumer, num_decodes, num_reqs, expected
):
    runner = SimpleNamespace(
        is_kv_consumer=is_kv_consumer,
        num_decodes=num_decodes,
    )

    assert QaicModelRunnerAoT._num_decode_rows_to_pad(runner, num_reqs) == expected


@pytest.mark.parametrize("lora_mode", [False, True])
def test_decode_batch_input_helper_builds_static_shapes(lora_mode):
    runner = SimpleNamespace(decode_bsz=2, lora_mode=lora_mode)

    inputs = QaicCausalLM._make_decode_batch_input_for_k(runner, 3)

    assert inputs["input_ids"].shape == (2, 4)
    assert inputs["position_ids"].shape == (2, 4)
    assert inputs["input_ids"].dtype == np.int64
    assert np.all(inputs["position_ids"] == -1)
    assert inputs["batch_index"].tolist() == [[0], [1]]
    assert ("lora_ids" in inputs) is lora_mode


def test_decode_ks_from_session_uses_decode_shapes():
    runner = SimpleNamespace(
        session=SimpleNamespace(
            binding_index_map={"input_ids": 0},
            allowed_shapes=[[(0, (1, 1))], [(0, (1, 4))], [(0, (1, 128))]],
        ),
        prefill_seq_len=128,
        decode_ks=[3],
    )

    assert QaicCausalLM._decode_ks_from_session(runner) == [0, 3]


def test_decode_ks_from_session_falls_back_when_shapes_are_unavailable():
    runner = SimpleNamespace(
        session=SimpleNamespace(binding_index_map={}, allowed_shapes=[]),
        prefill_seq_len=128,
        decode_ks=[3],
    )

    assert QaicCausalLM._decode_ks_from_session(runner) == [3]


@pytest.mark.parametrize(
    ("configured", "available", "messages"),
    [
        ([0, 3], [0, 3, 5], ["extra decode specializations"]),
        ([0, 3, 5], [0, 3], ["missing decode specializations"]),
        ([0, 3], [0, 3], []),
    ],
)
def test_decode_k_mismatch_warnings(configured, available, messages, caplog):
    caplog.set_level("WARNING")

    QaicCausalLM._warn_on_decode_k_mismatch(configured, available)

    for message in messages:
        assert message in caplog.text
    if not messages:
        assert caplog.text == ""


def test_compile_completion_is_deferred_when_draft_model_exists():
    spec_config = MagicMock()
    spec_config.uses_draft_model.return_value = True
    model = SimpleNamespace(
        decode_ks=[0, 3],
        disagg_serving_en=False,
        kv_cache_info=None,
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model="target"),
        speculative_config=spec_config,
        drafter=MagicMock(),
        num_spec_tokens=3,
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        lora_config=None,
        device=torch.device("cpu"),
    )

    with (
        patch(
            "vllm_qaic.worker.model_runner.set_current_vllm_config",
            return_value=nullcontext(),
        ),
        patch(
            "vllm_qaic.model_loader.qaic.load_qaic_model",
            return_value=model,
        ) as load_model,
    ):
        QaicModelRunnerAoT.load_model(runner)

    load_model.assert_called_once_with(
        runner.vllm_config,
        "target",
        raise_on_compile_complete=False,
    )
    assert runner.decode_ks == [0, 3]
    assert runner.active_k == 3
    runner.drafter.load_model.assert_called_once_with()


def test_register_connector_is_aot_only():
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    keys = ["QaicConnector", "QaicLMCacheConnectorV1"]
    saved = {
        key: KVConnectorFactory._registry.pop(key)
        for key in keys
        if key in KVConnectorFactory._registry
    }
    try:
        with patch.object(kv_connector_module, "current_platform") as platform:
            platform.is_aot_inference.return_value = False
            kv_connector_module.register_connector()
            assert not any(key in KVConnectorFactory._registry for key in keys)

            platform.is_aot_inference.return_value = True
            kv_connector_module.register_connector()
            assert all(key in KVConnectorFactory._registry for key in keys)
    finally:
        for key in keys:
            KVConnectorFactory._registry.pop(key, None)
        KVConnectorFactory._registry.update(saved)


def test_disagg_compile_does_not_default_kv_head_replication():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=None,
            max_model_len=128,
            runner_type="generate",
            is_multimodal_model=False,
            hf_config=SimpleNamespace(
                num_attention_heads=32,
                num_key_value_heads=8,
            ),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_cpu_blocks=1,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        additional_config={
            "device_group": [9],
            "override_qaic_config": {},
        },
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_role="kv_producer"),
        lora_config=None,
    )

    qaicrt = types.ModuleType("qaicrt")
    qaicrt.QStatus = SimpleNamespace(QS_SUCCESS=0)

    class FakeUtil:
        def getResourceInfo(self, _qid):
            return qaicrt.QStatus.QS_SUCCESS, SimpleNamespace(nspTotal=16)

    qaicrt.Util = FakeUtil

    with (
        patch.dict("sys.modules", {"qaicrt": qaicrt}),
        patch.object(
            qaic_model_loader,
            "QAIC_DEVICE_CONFIG",
            {"default": {}, "target": {}, "draft": {}},
        ),
    ):
        _ = qaic_model_loader._get_qaic_compile_config(config, "default")


@pytest.mark.parametrize(
    "replication_override",
    [
        {
            "replicate_kv_heads": False,
            "num_replicate_kv_heads": 2,
        },
        {
            "qaic_config": {
                "replicate_kv_heads": False,
                "num_replicate_kv_heads": 2,
            }
        },
    ],
)
def test_disagg_compile_preserves_explicit_qaic_replication_config(
    replication_override,
):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=None,
            max_model_len=128,
            runner_type="generate",
            is_multimodal_model=False,
            hf_config=SimpleNamespace(
                num_attention_heads=32,
                num_key_value_heads=8,
            ),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_cpu_blocks=1,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        additional_config={
            "device_group": [9],
            "override_qaic_config": replication_override,
        },
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        lora_config=None,
    )

    qaicrt = types.ModuleType("qaicrt")
    qaicrt.QStatus = SimpleNamespace(QS_SUCCESS=0)

    class FakeUtil:
        def getResourceInfo(self, _qid):
            return qaicrt.QStatus.QS_SUCCESS, SimpleNamespace(nspTotal=16)

    qaicrt.Util = FakeUtil

    with (
        patch.dict("sys.modules", {"qaicrt": qaicrt}),
        patch.object(
            qaic_model_loader,
            "QAIC_DEVICE_CONFIG",
            {"default": {}, "target": {}, "draft": {}},
        ),
    ):
        compiled = qaic_model_loader._get_qaic_compile_config(config, "default")

    assert compiled.qaic_config == {
        "replicate_kv_heads": False,
        "num_replicate_kv_heads": 2,
    }


@pytest.mark.parametrize(
    ("max_model_len", "user_qaic_config"),
    [
        (
            14336,
            {
                "blocking_mode": "kv_batch_fold",
                "num_kv_blocks": 64,
                "gdn_num_head_blocks": 8,
                "skip_kv": True,
                "ctx_len": 14336,
                "moe_config": {
                    "flavour": "expert_parallel",
                    "cores_per_expert": 1,
                    "tree_reduce": False,
                    "expert_parallel_chunk_size": 256,
                },
            },
        ),
        (
            10240,
            {
                "blocking_mode": "kv_batch_fold",
                "num_kv_blocks": 4,
                "ctx_len": 10240,
                "skip_kv": True,
                "moe_config": {
                    "flavour": "expert_parallel",
                    "cores_per_expert": 2,
                    "tree_reduce": False,
                    "expert_parallel_chunk_size": 256,
                },
            },
        ),
    ],
)
def test_disagg_compile_forwards_nested_qaic_config_to_qeff(
    max_model_len, user_qaic_config
):
    """Keep blocking/MoE options in the kwargs sent to QEfficient.compile()."""
    override = {"qaic_config": deepcopy(user_qaic_config)}
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=None,
            max_model_len=max_model_len,
            runner_type="generate",
            is_multimodal_model=False,
            hf_config=SimpleNamespace(
                num_attention_heads=32,
                num_key_value_heads=8,
            ),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_cpu_blocks=1,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        additional_config={
            "device_group": [9],
            "override_qaic_config": override,
        },
        speculative_config=None,
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        lora_config=None,
    )

    qaicrt = types.ModuleType("qaicrt")
    qaicrt.QStatus = SimpleNamespace(QS_SUCCESS=0)

    class FakeUtil:
        def getResourceInfo(self, _qid):
            return qaicrt.QStatus.QS_SUCCESS, SimpleNamespace(nspTotal=16)

    qaicrt.Util = FakeUtil

    with (
        patch.dict("sys.modules", {"qaicrt": qaicrt}),
        patch.object(
            qaic_model_loader,
            "QAIC_DEVICE_CONFIG",
            {"default": {}, "target": {}, "draft": {}},
        ),
    ):
        compiled = qaic_model_loader._get_qaic_compile_config(config, "default")

    assert compiled.qaic_config == user_qaic_config
    assert compiled.cfg["qaic_config"] == user_qaic_config
    assert config.additional_config["override_qaic_config"] == override


def test_disagg_launcher_forwards_nested_qaic_config():
    run_args_vllm_serve = pytest.importorskip("qaic_disagg.utils").run_args_vllm_serve
    override = {
        "qaic_config": {
            "replicate_kv_heads": True,
            "num_replicate_kv_heads": 2,
        }
    }
    args = SimpleNamespace(
        model="test-model",
        port=8000,
        device_group=[9],
        override_qaic_config=override,
    )

    command = run_args_vllm_serve(
        args,
        "prefill",
        kv_connector="",
        skip_kv_connector=True,
    )

    additional_config = json.loads(command[command.index("--additional-config") + 1])
    assert additional_config == {
        "device_group": [9],
        "override_qaic_config": override,
    }


def test_same_device_draft_and_target_split_nsp_cores():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=None,
            max_model_len=128,
            runner_type="generate",
            is_multimodal_model=False,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_cpu_blocks=1,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        additional_config={
            "device_group": [9],
            "override_qaic_config": {},
        },
        speculative_config=SimpleNamespace(
            method="draft_model",
            num_speculative_tokens=3,
        ),
        kv_transfer_config=None,
        lora_config=None,
    )
    qaicrt = types.ModuleType("qaicrt")
    qaicrt.QStatus = SimpleNamespace(QS_SUCCESS=0)

    class FakeUtil:
        def getResourceInfo(self, _qid):
            return qaicrt.QStatus.QS_SUCCESS, SimpleNamespace(nspTotal=16)

    qaicrt.Util = FakeUtil
    device_config = {
        "target": {"num_cores": None},
        "draft": {"num_cores": None},
    }

    with (
        patch.dict("sys.modules", {"qaicrt": qaicrt}),
        patch.object(qaic_model_loader, "QAIC_DEVICE_CONFIG", device_config),
    ):
        target = qaic_model_loader._get_qaic_compile_config(
            config,
            "target",
            speculative_config=config.speculative_config,
        )
        draft = qaic_model_loader._get_qaic_compile_config(
            config,
            "draft",
            speculative_config=config.speculative_config,
        )

    assert target.cfg["num_cores"] == 8
    assert draft.cfg["num_cores"] == 8


def test_explicit_core_override_bypasses_automatic_draft_split():
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=None,
            max_model_len=128,
            runner_type="generate",
            is_multimodal_model=False,
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            num_cpu_blocks=1,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        additional_config={
            "device_group": [9],
            "override_qaic_config": {"num_cores": 5},
        },
        speculative_config=SimpleNamespace(
            method="draft_model",
            num_speculative_tokens=3,
        ),
        kv_transfer_config=None,
        lora_config=None,
    )

    with patch.object(
        qaic_model_loader,
        "QAIC_DEVICE_CONFIG",
        {"target": {"num_cores": None}, "draft": {"num_cores": None}},
    ):
        compiled = qaic_model_loader._get_qaic_compile_config(
            config,
            "draft",
            speculative_config=config.speculative_config,
        )

    assert compiled.cfg["num_cores"] == 5


def test_draft_model_compile_preserves_target_speculative_config():
    speculative_config = SimpleNamespace(
        method="draft_model",
        num_speculative_tokens=3,
    )
    kv_transfer_config = SimpleNamespace(kv_role="kv_consumer")
    config = SimpleNamespace(
        model_config=SimpleNamespace(is_multimodal_model=False),
        speculative_config=speculative_config,
        kv_transfer_config=kv_transfer_config,
    )
    model = SimpleNamespace(sampler=SimpleNamespace(include_gpu_probs_tensor=False))
    captured = {}

    class CompileConfigReached(Exception):
        pass

    def capture_compile_config(
        compile_config, speculative_model_type, speculative_config=None
    ):
        captured["config"] = compile_config
        captured["type"] = speculative_model_type
        captured["speculative_config"] = speculative_config
        raise CompileConfigReached

    with (
        patch.object(qaic_model_loader, "QaicCausalLM", return_value=model),
        patch.object(
            qaic_model_loader,
            "_get_qaic_compile_config",
            side_effect=capture_compile_config,
        ),
        pytest.raises(CompileConfigReached),
    ):
        qaic_model_loader.load_qaic_model(config, "draft")

    assert captured["config"].speculative_config is None
    assert captured["config"].kv_transfer_config is None
    assert captured["type"] == "draft"
    assert captured["speculative_config"] is speculative_config
