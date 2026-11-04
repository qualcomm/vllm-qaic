# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm_qaic.model_loader.qaic import (
    QaicCausalLM,
    QaicCompilationComplete,
    get_hf_model,
)
from vllm_qaic.worker.model_runner import QaicModelRunnerAoT


@pytest.mark.parametrize("dflash", [False, True])
def test_prefill_image_counter_reuses_storage_and_preserves_request(dflash):
    seen = []
    pending = []
    initial = np.array([[0]], dtype=np.int64)
    output = np.array([[0]], dtype=np.int64)

    def run(inputs, is_prefill):
        assert is_prefill
        cursor = inputs["image_idx"]
        assert not np.shares_memory(cursor, inputs["image_idx_output"])
        seen.append((cursor, cursor.item()))
        pending.append(inputs["image_idx_output"])
        return len(seen) - 1

    def complete(index, wait):
        assert wait
        pending[index][...] = seen[index][1] + 4

    model = SimpleNamespace(
        prefill_seq_len=4,
        uses_mrope=False,
        pad=np.full(4, -1, dtype=np.int64),
        config=SimpleNamespace(model_type="qwen3_5"),
        ignore_batch_index=True,
        is_dflash_vlm=dflash,
        comp_ctx_lengths_prefill=None,
        _prefill_hidden_scratch=None,
        use_async_scheduling=False,
        session=SimpleNamespace(
            input_names=[],
            prefill_available_exec_objs=Mock(empty=lambda: False),
            np_run=run,
            complete_inf=complete,
        ),
    )
    # Two requests share the default MM buffers; neither may inherit the
    # previous request's final cursor. Each request has three prefill chunks.
    QaicCausalLM._run_prefill(
        model,
        input_ids=np.zeros(24, dtype=np.int64),
        positions=np.tile(np.arange(12), 2),
        batch_indices=np.arange(2),
        prefill_cum_sum=np.array([12, 24]),
        pending_exec_queue=Queue(),
        logits=None,
        mm_kwargs_list=[{"image_idx": initial, "image_idx_output": output}] * 2,
    )
    assert [value for _, value in seen] == ([0, 4, 8] * 2 if dflash else [0] * 6)
    assert seen[0][0] is seen[1][0] is seen[2][0]
    assert seen[3][0] is seen[4][0] is seen[5][0]
    assert initial.item() == 0
    if dflash:
        assert seen[0][0] is not seen[3][0]
        assert not np.shares_memory(seen[0][0], initial)


@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize("draft_error", [None, QaicCompilationComplete, ValueError])
def test_compile_only_builds_draft_before_exiting(monkeypatch, dflash, draft_error):
    from vllm_qaic.model_loader import qaic
    from vllm_qaic.worker import model_runner

    load = Mock(side_effect=QaicCompilationComplete())
    monkeypatch.setattr(qaic, "load_qaic_model", load)
    monkeypatch.setattr(
        model_runner, "set_current_vllm_config", lambda _: nullcontext()
    )
    draft = SimpleNamespace(
        tlm_prefill_seq_len=16,
        load_model=Mock(side_effect=draft_error),
    )
    runner = SimpleNamespace(
        model_config=SimpleNamespace(model="target"),
        vllm_config=SimpleNamespace(
            additional_config={"override_qaic_config": {"prefill_seq_len": 128}}
        ),
        speculative_config=SimpleNamespace(use_dflash=lambda: dflash)
        if dflash
        else None,
        drafter=draft,
        num_spec_tokens=15 if dflash else 0,
    )
    error = (
        ValueError if dflash and draft_error is ValueError else QaicCompilationComplete
    )
    with pytest.raises(error):
        QaicModelRunnerAoT.load_model(runner)
    load.assert_called_once_with(runner.vllm_config, "target" if dflash else "default")
    assert not hasattr(runner, "model")
    assert draft.load_model.call_count == int(dflash)
    assert draft.tlm_prefill_seq_len == (128 if dflash else 16)


@pytest.mark.parametrize(
    "model_type,quantizer_object",
    [
        ("gemma4", False),
        ("qwen3_vl", False),
        ("qwen3_5", False),
        ("qwen3_5_moe", False),
        ("gpt_oss", False),
        ("gpt_oss", True),
        ("qwen3", False),
        ("llama", False),
    ],
)
@pytest.mark.parametrize("metadata", ["none", "layers_only", "repo_only", "dflash"])
def test_hf_loading_defaults_require_dflash_metadata(
    monkeypatch, model_type, metadata, quantizer_object
):
    import QEfficient
    from vllm_qaic.model_loader import qaic

    vlm = model_type in {"gemma4", "qwen3_vl", "qwen3_5", "qwen3_5_moe"}
    hf_config = SimpleNamespace(model_type=model_type)
    if model_type == "gpt_oss":
        hf_config.architectures = ["GptOssForCausalLM"]
        hf_config.quantization_config = (
            SimpleNamespace(quant_method="gpt_oss_mxfp4")
            if quantizer_object
            else {"quant_method": "gpt_oss_mxfp4"}
        )
    config = SimpleNamespace(
        hf_config=hf_config,
        model="target",
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        is_multimodal_model=vlm,
        runner_type="generate",
    )
    load = Mock(return_value=object())
    for cls in (
        QEfficient.QEFFAutoModelForCausalLM,
        QEfficient.QEFFAutoModelForImageTextToText,
    ):
        monkeypatch.setattr(cls, "from_pretrained", load)
    monkeypatch.setattr(qaic, "is_json_serializable", lambda _: True)
    monkeypatch.setattr(qaic, "_CONFIG_REGISTRY", {})
    qaic_config = {}
    if metadata in {"layers_only", "dflash"}:
        qaic_config["target_layer_ids"] = [1, 3]
    if metadata in {"repo_only", "dflash"}:
        qaic_config["dflash_dlm_repo"] = "draft"
    get_hf_model(config, qaic_config=qaic_config, kv_offload=vlm)
    args = load.call_args.kwargs
    specialized = metadata == "dflash" and (vlm or model_type == "gpt_oss")
    assert args.get("dtype") == ("float32" if specialized else None)
    if vlm:
        assert args["continuous_batching"] is (metadata != "dflash")
    if model_type == "gpt_oss":
        original_quantizer = hf_config.quantization_config
        loaded_quantizer = args["config"].quantization_config
        if quantizer_object:
            original_quantizer = vars(original_quantizer)
            loaded_quantizer = vars(loaded_quantizer)
        assert loaded_quantizer["quant_method"] == "mxfp4"
        assert original_quantizer["quant_method"] == (
            "gpt_oss_mxfp4" if specialized else "mxfp4"
        )
        if specialized:
            assert args["config"] is not hf_config


def test_dflash_registry_separates_vlm_from_text_architectures():
    from vllm_qaic.utils.qaic_utils import DFLASH_MODEL_TYPES, DFLASH_VLM_MODEL_TYPES

    assert {"gemma4", "qwen3_vl", "qwen3_5", "qwen3_5_moe"} == DFLASH_VLM_MODEL_TYPES
    assert {"qwen3", "llama", "gpt_oss"} == DFLASH_MODEL_TYPES - DFLASH_VLM_MODEL_TYPES
