# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""DFlash uses the completed-prefill handoff, without a streaming connector."""

from contextlib import nullcontext
import json
from pathlib import Path
from queue import Queue
import runpy
import sys
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm_qaic.model_loader.qaic import QaicCausalLM
from vllm_qaic.model_loader.qaic import QaicCompilationComplete
from vllm_qaic.platform_base import QaicPlatform
from vllm_qaic.spec_decode.dflash_draft_model import QaicDFlashProposer
from vllm_qaic.utils.qaic_utils import KVCacheKind
from vllm_qaic.worker.model_runner import QaicModelRunnerAoT


@pytest.fixture
def pipeline():
    available, pending = Queue(), {}
    for slot in range(2):
        available.put(slot)
    handoffs, output_ids = [], set()

    def submit(inputs, last_chunk, kv_cache_buffers):
        index = available.get_nowait()
        pending[index] = (inputs, inputs["position_ids"].copy())
        assert id(inputs["hidden_states"]) not in output_ids
        output_ids.add(id(inputs["hidden_states"]))
        if last_chunk:
            handoffs.append(kv_cache_buffers)
        else:
            assert kv_cache_buffers is None
        return index

    def complete(index, is_prefill):
        assert is_prefill
        inputs, saved_positions = pending.pop(index)
        np.testing.assert_array_equal(inputs["position_ids"], saved_positions)
        slot = int(inputs["batch_index"].item())
        inputs["hidden_states"][:] = (saved_positions + 100 * slot)[..., None]
        inputs["logits"][:] = saved_positions.max()
        output_ids.remove(id(inputs["hidden_states"]))
        available.put(index)

    model = SimpleNamespace(
        last_decode=False,
        dflash_disagg=True,
        disagg_producer_en=True,
        _dflash_pipeline_buffers=[],
        _dflash_pipeline_pending={},
        _dflash_prefill_hidden={},
        _dflash_prefill_hidden_scratch=np.zeros((1, 4, 3), dtype=np.float32),
        config=SimpleNamespace(get_text_config=lambda: SimpleNamespace(hidden_size=3)),
        prefill_num_logits_buffer={},
        prefill_seq_len=4,
        uses_mrope=False,
        pad=np.full(4, -1, dtype=np.int64),
        comp_ctx_lengths_prefill=None,
        use_async_scheduling=False,
        session=SimpleNamespace(
            prefill_num_execObj=2,
            input_names=[],
            prefill_available_exec_objs=available,
            np_run_pipeline=submit,
            complete_inf=complete,
        ),
    )
    model.complete_inf = MethodType(QaicCausalLM.complete_inf, model)
    return model, handoffs, pending, available


def run_prefill(model, slot, start, end, length, buffers):
    logits = np.zeros((1, 1, 2), dtype=np.float32)
    QaicCausalLM._run_pipeline_prefill(
        model,
        np.arange(start, end),
        np.arange(start, end),
        np.array([slot]),
        np.array([end - start]),
        Queue(),
        buffers,
        [end < length],
        logits=logits,
        num_prompt_tokens_prefill=np.array([length]),
    )
    return logits


@pytest.mark.parametrize("length", [1, 4, 5, 10, 17, 33])
def test_pipeline_handoff_contains_whole_prompt_and_excludes_padding(pipeline, length):
    model, handoffs, pending, available = pipeline
    kv = np.ones((1, 1, length, 2))
    hidden = np.full((1, 1, length, 3), -999, dtype=np.float32)
    logits = run_prefill(model, 1, 0, length, length, [[], [kv, hidden]])
    assert len(handoffs) == 1 and handoffs[0][0] is kv and len(handoffs[0]) == 1
    np.testing.assert_array_equal(hidden[0, 0, :, 0], np.arange(length) + 100)
    assert np.all(logits == length - 1)
    assert available.qsize() == 2 and not pending
    assert not model._dflash_prefill_hidden
    assert not model._dflash_pipeline_pending
    assert len(model._dflash_pipeline_buffers) == 2


def test_failed_completion_preserves_pending_features_and_buffers(
    pipeline, monkeypatch
):
    model, _, pending, available = pipeline
    hidden = np.full((1, 1, 3, 3), -999, dtype=np.float32)
    complete = model.session.complete_inf
    monkeypatch.setattr(
        model.session,
        "complete_inf",
        Mock(side_effect=RuntimeError("inference failed")),
    )
    with pytest.raises(RuntimeError, match="inference failed"):
        run_prefill(model, 0, 0, 3, 3, [[np.zeros((1, 1, 3, 2)), hidden]])
    assert len(model._dflash_pipeline_pending) == len(pending) == 1
    assert len(model._dflash_pipeline_buffers) == available.qsize() == 1
    assert np.all(hidden == -999)
    monkeypatch.setattr(model.session, "complete_inf", complete)
    model.complete_inf(next(iter(pending)), True)
    np.testing.assert_array_equal(hidden[0, 0, :, 0], np.arange(3))
    assert not model._dflash_pipeline_pending and not pending
    assert len(model._dflash_pipeline_buffers) == available.qsize() == 2


@pytest.mark.parametrize("dflash,is_prefill", [(False, True), (True, False)])
def test_other_completions_do_not_consume_pipeline_features(
    pipeline, monkeypatch, dflash, is_prefill
):
    model, _, _, _ = pipeline
    model.dflash_disagg = dflash
    marker = object()
    model._dflash_pipeline_pending[0] = marker
    complete = Mock()
    monkeypatch.setattr(model.session, "complete_inf", complete)
    model.complete_inf(0, is_prefill)
    complete.assert_called_once_with(0, is_prefill)
    assert model._dflash_pipeline_pending[0] is marker


def test_prefill_completion_without_pipeline_features(pipeline, monkeypatch):
    model, _, _, _ = pipeline
    complete = Mock()
    monkeypatch.setattr(model.session, "complete_inf", complete)
    model.complete_inf(0, True)
    complete.assert_called_once_with(0, True)
    assert not model._dflash_pipeline_pending
    assert not model._dflash_pipeline_buffers


def test_partial_scheduler_steps_do_not_publish_or_lose_features(pipeline):
    model, handoffs, _, _ = pipeline
    run_prefill(model, 0, 0, 4, 11, [[]])
    run_prefill(model, 0, 4, 8, 11, [[]])
    assert not handoffs
    np.testing.assert_array_equal(model._dflash_prefill_hidden[0][:8, 0], np.arange(8))
    hidden = np.full((1, 1, 11, 3), -999, dtype=np.float32)
    run_prefill(model, 0, 8, 11, 11, [[np.zeros((1, 1, 11, 2)), hidden]])
    np.testing.assert_array_equal(hidden[0, 0, :, 0], np.arange(11))
    assert not model._dflash_prefill_hidden


def test_reused_slot_replaces_cancelled_prompt_features(pipeline):
    model, _, _, _ = pipeline
    run_prefill(model, 0, 0, 4, 17, [[]])
    old = model._dflash_prefill_hidden[0]
    run_prefill(model, 0, 0, 4, 9, [[]])
    assert model._dflash_prefill_hidden[0] is not old
    assert model._dflash_prefill_hidden[0].shape == (9, 3)


def test_interleaved_request_slots_keep_independent_prompt_features(pipeline):
    model, _, _, _ = pipeline
    run_prefill(model, 1, 0, 4, 7, [[], []])
    run_prefill(model, 0, 0, 4, 9, [[], []])
    hidden_a = np.zeros((1, 1, 7, 3), dtype=np.float32)
    hidden_b = np.zeros((1, 1, 9, 3), dtype=np.float32)
    buffers = [[None, hidden_b], [None, hidden_a]]
    run_prefill(model, 1, 4, 7, 7, buffers)
    run_prefill(model, 0, 4, 9, 9, buffers)
    np.testing.assert_array_equal(hidden_a[0, 0, :, 0], np.arange(7) + 100)
    np.testing.assert_array_equal(hidden_b[0, 0, :, 0], np.arange(9))


@pytest.fixture
def prompt_proposer():
    proposer = QaicDFlashProposer.__new__(QaicDFlashProposer)
    proposer.hidden_size = 3
    proposer.tlm_prefill_seq_len = 8
    proposer._prefill_pending = []
    return proposer


@pytest.mark.parametrize("length", [1, 4, 5, 13, 33, 129])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_received_prompt_is_owned_before_shared_memory_release(
    prompt_proposer, length, dtype
):
    features = np.arange(length * 3, dtype=dtype).reshape(1, 1, length, 3)
    logits = np.arange(32).reshape(1, 4, 8)
    prompt_proposer.queue_prefill("r", 0, features[0, 0], length, logits[:, :1])
    pending = prompt_proposer._prefill_pending[0]
    owned = pending["target_hidden"]
    assert len(owned) == -(-length // 8)
    for chunk in owned:
        assert chunk.shape == (1, 8, 3)
        assert chunk.dtype == dtype
        assert not np.shares_memory(chunk, features)
    joined = np.concatenate(owned, axis=1)[0]
    np.testing.assert_array_equal(joined[:length], features[0, 0])
    assert np.all(joined[length:] == 0)
    features[:] = -999
    assert np.all(joined >= 0)
    np.testing.assert_array_equal(pending["prefill_cum_sum"], [length])
    np.testing.assert_array_equal(pending["prefill_positions"], np.arange(length))
    np.testing.assert_array_equal(pending["last_chunk_logits"], logits[:, :1])


def test_prefill_queue_preserves_request_and_cache_slot_order(prompt_proposer):
    lengths = [9, 5]
    features = [
        np.full((1, 1, n, 3), value, dtype=np.float32)
        for n, value in zip(lengths, [7, 11], strict=True)
    ]
    logits = np.arange(96).reshape(3, 4, 8)
    for i, (req_id, slot, length) in enumerate(
        zip(["a", "b"], [2, 0], lengths, strict=True)
    ):
        prompt_proposer.queue_prefill(
            req_id, slot, features[i][0, 0], length, logits[i + 1 : i + 2, :1]
        )
    pending = prompt_proposer._prefill_pending
    assert [p["req_id"] for p in pending] == ["a", "b"]
    assert [p["batch_indices"].item() for p in pending] == [2, 0]
    assert [len(p["target_hidden"]) for p in pending] == [2, 1]
    for i, p in enumerate(pending):
        np.testing.assert_array_equal(p["prefill_positions"], np.arange(lengths[i]))
        np.testing.assert_array_equal(p["last_chunk_logits"], logits[i + 1 : i + 2, :1])
    assert np.all(pending[0]["target_hidden"][0] == 7)
    assert np.all(pending[1]["target_hidden"][0][0, :5] == 11)


@pytest.mark.parametrize("shape", [(0, 3), (3, 3), (5, 2)])
def test_prefill_queue_rejects_invalid_prompt_features(prompt_proposer, shape):
    with pytest.raises(ValueError, match="DFlash"):
        prompt_proposer.queue_prefill("r", 0, np.zeros(shape), 5, np.zeros((1, 1, 8)))
    assert not prompt_proposer._prefill_pending


def test_finished_draft_requests_use_normal_batch_gc():
    proposer = QaicDFlashProposer.__new__(QaicDFlashProposer)
    proposer._req_state = {"decode": object(), "old": object()}
    proposer._gc_finished_reqs({"decode"})
    assert set(proposer._req_state) == {"decode"}


@pytest.mark.parametrize("dflash", [False, True])
def test_prompt_features_are_not_bound_as_hardware_kv(dflash):
    kv = np.ones((1, 2, 7, 4))
    features = np.zeros((1, 1, 7, 3))
    handoff = Mock()
    model = SimpleNamespace(
        dflash_disagg=dflash,
        kv_cache_info=[((1, 2, 16, 4), np.float32, 512)],
        session=SimpleNamespace(set_data_for_kv_handoff=handoff, decode_buff_map={}),
        decode_execObj_idx=0,
    )
    buffers = [[kv, features] if dflash else [kv]]
    QaicCausalLM._kv_handoff(model, [0], buffers)
    assert len(handoff.call_args.args[0]) == 1
    assert handoff.call_args.args[0][0] is kv
    if dflash:
        assert buffers[0][-1] is features


@pytest.fixture
def platform_config(monkeypatch):
    monkeypatch.setattr(QaicPlatform, "is_aot", True)
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash", max_num_new_slots_for_drafting=16
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="future_dense"),
            is_multimodal_model=False,
            is_moe=False,
            is_hybrid=False,
            is_attention_free=False,
            enforce_eager=False,
            max_model_len=4096,
            runner_type="generate",
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="QaicConnector", kv_role="kv_producer"
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=True,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            long_prefill_token_threshold=128,
            max_num_scheduled_tokens=999,
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        device_config=SimpleNamespace(device_type="cpu"),
        parallel_config=SimpleNamespace(worker_cls="auto", world_size=1),
        additional_config={
            "override_qaic_config": {"prefill_seq_len": 128, "stages": 2}
        },
        lora_config=None,
        compilation_config=None,
    )


@pytest.mark.parametrize(
    "section,field,value,message",
    [
        ("model_config", "is_moe", True, "dense attention"),
        ("model_config", "is_hybrid", True, "dense attention"),
        ("model_config", "is_attention_free", True, "dense attention"),
        ("scheduler_config", "async_scheduling", True, "synchronous"),
        (
            "kv_transfer_config",
            "kv_connector",
            "QaicLMCacheConnectorV1",
            "QaicConnector",
        ),
    ],
)
def test_disaggregation_rejects_unsupported_configuration(
    platform_config, section, field, value, message
):
    setattr(getattr(platform_config, section), field, value)
    with pytest.raises(ValueError, match=message):
        QaicPlatform.check_and_update_config(platform_config)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("ccl_enabled", True, "CCL"),
        ("comp_ctx_lengths_prefill", [128], "CCL"),
        ("comp_ctx_lengths_decode", [128], "CCL"),
    ],
)
def test_disaggregation_rejects_invalid_compile_options(
    platform_config, key, value, message
):
    platform_config.additional_config["override_qaic_config"][key] = value
    with pytest.raises(ValueError, match=message):
        QaicPlatform.check_and_update_config(platform_config)


@pytest.mark.parametrize("model_type", ["qwen3", "llama", "future_dense"])
@pytest.mark.parametrize("threshold", [0, 128])
@pytest.mark.parametrize(
    "role,pp,bs,ctx",
    [
        ("kv_producer", 1, 1, 4096),
        ("kv_producer", 2, 1, 4096),
        ("kv_producer", 2, 2, 4096),
        ("kv_producer", 4, 4, 256),
        ("kv_consumer", 1, 4, 4096),
        ("kv_consumer", 1, 8, 4096),
    ],
)
def test_dense_model_names_do_not_limit_pipeline_or_batching(
    platform_config, model_type, threshold, role, pp, bs, ctx
):
    cfg = platform_config
    cfg.model_config.hf_config.model_type = model_type
    cfg.model_config.max_model_len = ctx
    cfg.kv_transfer_config.kv_role = role
    cfg.scheduler_config.max_num_seqs = bs
    cfg.scheduler_config.long_prefill_token_threshold = threshold
    cfg.additional_config["override_qaic_config"]["stages"] = pp
    QaicPlatform.check_and_update_config(cfg)
    budget = (
        min(ctx, 128 * (pp + 1)) + 16 * bs
        if role == "kv_producer" and pp > 1
        else min(bs * 128, 512)
    )
    assert cfg.scheduler_config.max_num_batched_tokens == budget
    assert cfg.scheduler_config.max_num_scheduled_tokens is None
    assert cfg.scheduler_config.long_prefill_token_threshold == threshold


def test_dflash_consumer_does_not_add_pipeline_stages(platform_config):
    platform_config.kv_transfer_config.kv_role = "kv_consumer"
    cfg = platform_config.additional_config["override_qaic_config"]
    del cfg["stages"]
    QaicPlatform.check_and_update_config(platform_config)
    assert "stages" not in cfg


def test_dflash_uses_common_prefix_cache_and_multimodal_checks(platform_config):
    platform_config.cache_config.enable_prefix_caching = True
    QaicPlatform.check_and_update_config(platform_config)
    assert not platform_config.cache_config.enable_prefix_caching
    platform_config.model_config.is_multimodal_model = True
    with pytest.raises(AssertionError, match="SPD with Multi-modality"):
        QaicPlatform.check_and_update_config(platform_config)


@pytest.mark.parametrize("method", [None, "ngram", "draft_model", "suffix"])
def test_dflash_guards_do_not_restrict_other_methods(platform_config, method):
    platform_config.speculative_config = (
        SimpleNamespace(method=method) if method else None
    )
    platform_config.model_config.is_moe = True
    platform_config.model_config.is_hybrid = True
    platform_config.additional_config["override_qaic_config"]["ccl_enabled"] = True
    QaicPlatform.check_and_update_config(platform_config)


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
def test_completed_prompt_buffer_registered_only_for_dflash(
    monkeypatch, role, model_type
):
    from vllm_qaic.model_loader import qaic
    from vllm_qaic.worker import model_runner

    kv_info = [((1, 2, 128, 4), np.dtype("float16"), 2048, KVCacheKind.FA)]
    model = SimpleNamespace(
        disagg_serving_en=True,
        dflash_disagg=True,
        kv_cache_info=kv_info,
        config=SimpleNamespace(
            model_type=model_type,
            get_text_config=lambda: SimpleNamespace(hidden_size=8),
        ),
        ctx_len=128,
        decode_ks=[15],
        prefill_seq_len=32,
        get_io_shape_and_dtype=lambda *a, **kw: ([1, 32, 8], np.dtype("float16"), 512),
    )
    draft = SimpleNamespace(load_model=Mock()) if role == "kv_consumer" else None
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model="target"),
        num_spec_tokens=15,
        speculative_config=SimpleNamespace(
            use_dflash=lambda: True, uses_draft_model=lambda: False
        ),
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_connector="QaicConnector")
        ),
        drafter=draft,
        lora_config=None,
        _setup_dflash_hidden_capture=Mock(),
    )
    load_model = Mock(return_value=model)
    monkeypatch.setattr(qaic, "load_qaic_model", load_model)
    monkeypatch.setattr(
        model_runner, "set_current_vllm_config", lambda _: nullcontext()
    )
    QaicModelRunnerAoT.load_model(runner)
    load_model.assert_called_once_with(
        runner.vllm_config, "target", raise_on_compile_complete=role == "kv_producer"
    )
    assert len(kv_info) == 1 and len(runner.kv_cache_info) == 2
    assert runner.kv_cache_info[-1] == (
        (1, 1, 128, 8),
        np.dtype("float16"),
        2048,
        KVCacheKind.FA,
    )
    assert runner._setup_dflash_hidden_capture.call_count == (role == "kv_consumer")
    if draft:
        draft.load_model.assert_called_once()
        assert draft.tlm_prefill_seq_len == model.prefill_seq_len


@pytest.mark.parametrize("method", ["dflash", "draft_model"])
@pytest.mark.parametrize("draft_error", [QaicCompilationComplete, ValueError])
def test_compile_only_finishes_dlm_before_exit(monkeypatch, method, draft_error):
    from vllm_qaic.model_loader import qaic
    from vllm_qaic.worker import model_runner

    target = SimpleNamespace(
        disagg_serving_en=False, decode_ks=[15], prefill_seq_len=128
    )
    load_model = Mock(return_value=target)
    monkeypatch.setattr(qaic, "load_qaic_model", load_model)
    monkeypatch.setattr(
        model_runner, "set_current_vllm_config", lambda _: nullcontext()
    )
    draft = SimpleNamespace(load_model=Mock(side_effect=draft_error))
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model="target"),
        num_spec_tokens=15,
        speculative_config=SimpleNamespace(
            use_dflash=lambda: method == "dflash",
            uses_draft_model=lambda: method == "draft_model",
        ),
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        drafter=draft,
        lora_config=None,
        _setup_dflash_hidden_capture=Mock(),
    )
    with pytest.raises(draft_error):
        QaicModelRunnerAoT.load_model(runner)
    load_model.assert_called_once_with(
        runner.vllm_config, "target", raise_on_compile_complete=False
    )
    draft.load_model.assert_called_once()
    if method == "dflash":
        assert draft.tlm_prefill_seq_len == target.prefill_seq_len
    runner._setup_dflash_hidden_capture.assert_not_called()


@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("pp,bs", [(1, 1), (1, 4), (2, 4)])
def test_example_uses_unchanged_launcher(monkeypatch, reuse, pp, bs):
    launcher = pytest.importorskip("qaic_disagg.__main__")
    from qaic_disagg.utils import run_args_vllm_serve
    from transformers import AutoConfig

    script = Path(__file__).resolve().parents[3] / "examples/qaic_spd_dflash_disagg.py"
    args = [
        str(script),
        "--prefill-pipeline-parallel-size",
        str(pp),
        "--decode-max-num-seqs",
        str(bs),
    ]
    if reuse:
        args += [
            "--prefill-qpc",
            "/prefill/qpc",
            "--decode-qpc",
            "/decode/qpc",
            "--dlm-qpc",
            "/draft/qpc",
        ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        AutoConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(block_size=16)
    )
    captured = []
    monkeypatch.setattr(
        "os.execv", lambda executable, command: captured.extend(command)
    )
    runpy.run_path(str(script), run_name="__main__")
    manager = object.__new__(launcher.QaicDisaggJobManager)
    manager.started_exit = True
    manager.skip_prefill = False
    manager.args = launcher.get_parser().parse_args(captured[3:])
    for role in ("prefill", "decode"):
        worker_args = manager.get_args(0, role)
        command = run_args_vllm_serve(
            worker_args,
            role,
            "QaicConnector",
            False,
            kv_rank=0,
            kv_ip="127.0.0.1",
            kv_port=9224,
        )
        spec = json.loads(command[command.index("--speculative-config") + 1])
        extra = json.loads(command[command.index("--additional-config") + 1])
        assert command.count("--additional-config") == 1
        assert spec == {
            "method": "dflash",
            "model": "z-lab/Qwen3-8B-DFlash-b16",
            "num_speculative_tokens": 15,
        }
        assert extra["override_qaic_config"]["prefill_only"] == (role == "prefill")
        assert extra["override_qaic_config"]["split_model_io"] is True
        assert "split_retained_state_io" not in extra["override_qaic_config"]
        assert worker_args.max_num_seqs == (1 if role == "prefill" else bs)
        if role == "decode":
            draft = extra["draft_override_qaic_config"]
            assert draft["device_group"] == [8, 9, 10, 11]
            assert draft["prefill_seq_len"] == 16
            assert draft.get("qpc_path") == ("/draft/qpc" if reuse else None)
        else:
            assert "draft_override_qaic_config" not in extra
            assert extra["override_qaic_config"]["stages"] == pp


@pytest.mark.parametrize("model_type", ["qwen3", "llama"])
@pytest.mark.parametrize(
    "role,stages", [("kv_producer", 1), ("kv_producer", 2), ("kv_consumer", 1)]
)
@pytest.mark.parametrize(
    "io_config",
    [
        {},
        {"split_model_io": True},
        {"split_retained_state_io": True},
        {"split_model_io": False, "split_retained_state_io": False},
    ],
)
def test_compile_preserves_dflash_features_and_launcher_io(
    monkeypatch, model_type, role, stages, io_config
):
    from vllm_qaic.model_loader import qaic

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization=None,
            max_model_len=4096,
            runner_type="generate",
            is_multimodal_model=False,
            hf_config=SimpleNamespace(
                model_type=model_type, num_attention_heads=32, num_key_value_heads=8
            ),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto", num_cpu_blocks=4, enable_prefix_caching=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        additional_config={
            "device_group": [0, 1, 2, 3],
            "override_qaic_config": {
                "num_cores": 8,
                "stages": stages,
                **io_config,
            },
            "dflash_cfg": {
                "block_size": 16,
                "target_layer_ids": [1, 3],
                "dlm_repo": "draft",
                "tlm_repo": "target",
            },
        },
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=15),
        kv_transfer_config=SimpleNamespace(kv_role=role),
        lora_config=None,
    )
    monkeypatch.setattr(
        qaic, "QAIC_DEVICE_CONFIG", {"default": {}, "target": {}, "draft": {}}
    )
    compiled = qaic._get_qaic_compile_config(config, "target")
    assert compiled.qaic_config["target_layer_ids"] == [1, 3]
    assert compiled.qaic_config["dflash_dlm_repo"] == "draft"
    assert compiled.qaic_config["replicate_kv_heads"]
    assert compiled.cfg["prefill_only"] == (role == "kv_producer")
    assert compiled.num_logits_to_keep == (1 if role == "kv_producer" else 16)
    assert {
        key: compiled.cfg[key]
        for key in ("split_model_io", "split_retained_state_io")
        if key in compiled.cfg
    } == io_config
