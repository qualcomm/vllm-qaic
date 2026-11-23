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
    _derive_dflash_config,
    get_hf_model,
    load_qaic_model,
)
from vllm_qaic.worker.model_runner import QaicModelRunnerAoT


@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize("initial_value", [0, 7])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("increments", [(0,), (4,), (4, 0), (0, 4, 2), (4, 0, 2, 0)])
def test_prefill_image_counter_reuses_storage_and_preserves_request(
    dflash, initial_value, increments, hybrid
):
    seen = []
    pending = []
    initial = np.array([[initial_value]], dtype=np.int64)
    output = np.array([[0]], dtype=np.int64)
    n_chunks = len(increments)
    n_tokens = 4 * n_chunks

    def run(inputs, is_prefill):
        assert is_prefill
        cursor = inputs["image_idx"]
        assert not np.shares_memory(cursor, inputs["image_idx_output"])
        seen.append((cursor, cursor.item()))
        pending.append(inputs["image_idx_output"])
        return len(seen) - 1

    def complete(index, wait):
        assert wait
        pending[index][...] = seen[index][1] + increments[index % n_chunks]

    model = SimpleNamespace(
        prefill_seq_len=4,
        uses_mrope=False,
        pad=np.full(4, -1, dtype=np.int64),
        config=SimpleNamespace(model_type="qwen3_5"),
        ignore_batch_index=True,
        is_dflash_vlm=dflash,
        dflash_hybrid=hybrid,
        _dflash_prefix_positions=np.full((4, 4, 16), 17, dtype=np.int64),
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
    # previous request's final cursor, for either odd or even chunk counts.
    QaicCausalLM._run_prefill(
        model,
        input_ids=np.zeros(2 * n_tokens, dtype=np.int64),
        positions=np.tile(np.arange(n_tokens), 2),
        batch_indices=np.array([2, 0]),
        prefill_cum_sum=np.array([n_tokens, 2 * n_tokens]),
        pending_exec_queue=Queue(),
        logits=None,
        mm_kwargs_list=[{"image_idx": initial, "image_idx_output": output}] * 2,
    )
    expected = (
        np.cumsum([initial_value, *increments[:-1]]).tolist()
        if dflash
        else [initial_value] * n_chunks
    )
    assert [value for _, value in seen] == expected * 2
    assert initial.item() == initial_value
    assert np.all(model._dflash_prefix_positions[:, [1, 3]] == 17)
    assert np.all(model._dflash_prefix_positions[:, [0, 2]] == (-1 if hybrid else 17))
    if dflash:
        assert seen[0][0] is not seen[n_chunks][0]
        for start in (0, n_chunks):
            assert not np.shares_memory(seen[start][0], initial)
            for index in range(start + 1, start + n_chunks):
                assert seen[index][0] is pending[index - 1]
                assert pending[index] is seen[index - 1][0]
    else:
        assert all(cursor is initial for cursor, _ in seen)
        assert all(buffer is output for buffer in pending)


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
        speculative_config=SimpleNamespace(
            use_dflash=lambda: dflash, uses_draft_model=lambda: False
        )
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
    load.assert_called_once_with(
        runner.vllm_config,
        "target" if dflash else "default",
        raise_on_compile_complete=True,
    )
    assert not hasattr(runner, "model")
    assert draft.load_model.call_count == int(dflash)
    assert draft.tlm_prefill_seq_len == (128 if dflash else 16)


@pytest.mark.parametrize(
    "model_type",
    ["gemma4", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "qwen3", "llama", "gpt_oss"],
)
@pytest.mark.parametrize("metadata", ["none", "layers_only", "repo_only", "dflash"])
@pytest.mark.parametrize("continuous_batching", [None, False, True])
@pytest.mark.parametrize("nested_override", [False, True])
@pytest.mark.parametrize(
    "dtype_args", [{}, {"dtype": "float16"}, {"torch_dtype": "float16"}]
)
def test_hf_loading_preserves_dtype_and_batching_options(
    monkeypatch,
    model_type,
    metadata,
    continuous_batching,
    nested_override,
    dtype_args,
):
    import QEfficient
    from vllm_qaic.model_loader import qaic

    vlm = model_type in {"gemma4", "qwen3_vl", "qwen3_5", "qwen3_5_moe"}
    hf_config = SimpleNamespace(model_type=model_type)
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
    overrides = {"pretrained_extra_args": dtype_args.copy()} if dtype_args else {}
    if continuous_batching is not None:
        overrides["continuous_batching"] = continuous_batching
        if nested_override:
            overrides["continuous_batching"] = not continuous_batching
            overrides.setdefault("pretrained_extra_args", {})["continuous_batching"] = (
                continuous_batching
            )
    get_hf_model(
        config,
        qaic_config=qaic_config,
        kv_offload=vlm,
        additional_config={"override_qaic_config": overrides} if overrides else None,
    )
    args = load.call_args.kwargs
    assert {k: args[k] for k in ("dtype", "torch_dtype") if k in args} == dtype_args
    assert args["continuous_batching"] is (
        True if continuous_batching is None else continuous_batching
    )


@pytest.mark.parametrize("continuous_batching", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_loader_preserves_explicit_batching_override_when_building(
    monkeypatch, continuous_batching, cached
):
    from vllm_qaic.model_loader import qaic

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False, runner_type="generate", quantization=None
        ),
        speculative_config=None,
        lora_config=None,
        additional_config={
            "override_qaic_config": {
                "pretrained_extra_args": {"continuous_batching": continuous_batching}
            }
        },
    )
    compiled = SimpleNamespace(
        qpc_path="cached" if cached else None,
        qaic_config=None,
        kv_offload=False,
        cfg={},
        compile_only=True,
    )
    load = Mock(side_effect=QaicCompilationComplete())
    monkeypatch.setattr(qaic, "QaicCausalLM", Mock())
    monkeypatch.setattr(qaic, "_get_qaic_compile_config", Mock(return_value=compiled))
    monkeypatch.setattr(qaic, "check_qpc_exists", lambda _: True)
    monkeypatch.setattr(qaic, "get_hf_model", load)
    with pytest.raises(QaicCompilationComplete):
        load_qaic_model(config)
    if cached:
        load.assert_not_called()
    else:
        load.assert_called_once_with(
            config.model_config, None, None, False, config.additional_config
        )


@pytest.mark.parametrize("runner_type", ["pooling", "generate"])
@pytest.mark.parametrize("moe", [False, True])
def test_qwen35_loader_restores_native_config(monkeypatch, tmp_path, runner_type, moe):
    import QEfficient
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from vllm.transformers_utils.config import HFConfigParser
    from vllm.transformers_utils.configs.qwen3_5 import (
        Qwen3_5Config as VllmQwen35Config,
    )

    if moe:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig as Qwen3_5Config,
        )
        from vllm.transformers_utils.configs.qwen3_5_moe import (
            Qwen3_5MoeConfig as VllmQwen35Config,
        )

    native = Qwen3_5Config(tie_word_embeddings=False)
    native.save_pretrained(tmp_path)
    # Exercise vLLM's actual registration boundary before the generic loader.
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, native.model_type, Qwen3_5Config)
    _, compatibility = HFConfigParser().parse(tmp_path, trust_remote_code=False)
    assert type(compatibility) is VllmQwen35Config
    compatibility.text_config.max_position_embeddings = 2048
    compatibility.tie_word_embeddings = True
    original = compatibility.to_dict()
    load = Mock(return_value=object())
    monkeypatch.setattr(
        QEfficient.QEFFAutoModelForImageTextToText, "from_pretrained", load
    )
    config = SimpleNamespace(
        hf_config=compatibility,
        model=str(tmp_path),
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        is_multimodal_model=True,
        runner_type=runner_type,
    )
    get_hf_model(config, kv_offload=True)
    restored = load.call_args.kwargs["config"]
    assert type(restored) is Qwen3_5Config
    assert type(restored.vision_config) is Qwen3_5Config.sub_configs["vision_config"]
    assert (
        restored.vision_config.rope_parameters == native.vision_config.rope_parameters
    )
    assert restored.text_config.max_position_embeddings == 2048
    assert restored.tie_word_embeddings is False
    assert compatibility.to_dict() == original
    assert CONFIG_MAPPING[native.model_type] is VllmQwen35Config


@pytest.mark.parametrize("vllm_override", [False, True])
@pytest.mark.parametrize("native_available", [False, True])
def test_native_config_conversion_preserves_unrelated_registrations(
    monkeypatch, tmp_path, vllm_override, native_available
):
    import QEfficient
    from transformers import LlamaConfig
    from transformers.models.auto.configuration_auto import (
        CONFIG_MAPPING,
        CONFIG_MAPPING_NAMES,
    )
    from vllm_qaic.model_loader import qaic

    class CompatibilityConfig(LlamaConfig):
        pass

    LlamaConfig(tie_word_embeddings=False).save_pretrained(tmp_path)
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "llama", CompatibilityConfig)
    monkeypatch.setattr(
        qaic,
        "_CONFIG_REGISTRY",
        {"llama": CompatibilityConfig} if vllm_override else {},
    )
    if not native_available:
        monkeypatch.delitem(CONFIG_MAPPING_NAMES, "llama")
    compatibility = CompatibilityConfig.from_pretrained(tmp_path)
    compatibility.max_position_embeddings = 2048
    compatibility.tie_word_embeddings = True
    original = compatibility.to_dict()
    load = Mock(return_value=object())
    monkeypatch.setattr(QEfficient.QEFFAutoModelForCausalLM, "from_pretrained", load)
    config = SimpleNamespace(
        hf_config=compatibility,
        model=str(tmp_path),
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        is_multimodal_model=False,
        runner_type="generate",
    )
    get_hf_model(config)
    restored = load.call_args.kwargs["config"]
    expected = (
        LlamaConfig if vllm_override and native_available else CompatibilityConfig
    )
    assert type(restored) is expected
    assert restored.max_position_embeddings == 2048
    assert restored.tie_word_embeddings is False
    assert compatibility.to_dict() == original
    assert CONFIG_MAPPING["llama"] is CompatibilityConfig


@pytest.mark.parametrize("architecture", ["DFlashDraftModel", "DFlash2DraftModel"])
def test_shared_loader_preserves_dflash_architecture(
    monkeypatch, tmp_path, architecture
):
    import QEfficient
    from transformers import Qwen3Config
    from vllm.transformers_utils.configs.eagle import EAGLEConfig

    native = Qwen3Config(architectures=[architecture])
    native.save_pretrained(tmp_path)
    wrapped = EAGLEConfig(model=native, method="dflash", model_type="eagle")
    load = Mock(return_value=object())
    monkeypatch.setattr(QEfficient.QEFFAutoModelForCausalLM, "from_pretrained", load)
    config = SimpleNamespace(
        hf_config=wrapped,
        model=str(tmp_path),
        trust_remote_code=False,
        revision=None,
        code_revision=None,
        is_multimodal_model=False,
        runner_type="generate",
    )
    get_hf_model(config)
    restored = load.call_args.kwargs["config"]
    assert type(restored) is Qwen3Config
    assert restored.architectures == [architecture]
    assert wrapped.architectures == [architecture]


def test_dflash_registry_separates_vlm_from_text_architectures():
    from vllm_qaic.utils.qaic_utils import DFLASH_MODEL_TYPES, DFLASH_VLM_MODEL_TYPES

    assert {"gemma4", "qwen3_vl", "qwen3_5", "qwen3_5_moe"} == DFLASH_VLM_MODEL_TYPES
    assert {"qwen3", "llama", "gpt_oss"} == DFLASH_MODEL_TYPES - DFLASH_VLM_MODEL_TYPES


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("role", ["target", "draft"])
def test_dflash_rejects_unsupported_target_before_loading(cached, role):
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="dflash",
            target_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="unsupported")
            ),
        ),
        # Cached metadata must not bypass validation of the actual target.
        additional_config={"dflash_cfg": {"tlm_model_type": "qwen3"}}
        if cached
        else None,
    )
    with pytest.raises(
        ValueError, match="not supported for target model type 'unsupported'"
    ):
        load_qaic_model(config, role)


@pytest.mark.parametrize(
    "model_type",
    ["qwen3", "llama", "gemma4", "qwen3_vl", "qwen3_5", "qwen3_5_moe", "gpt_oss"],
)
@pytest.mark.parametrize("dflash2", [False, True])
def test_dflash_derivation_validates_target_not_draft(model_type, dflash2):
    draft = SimpleNamespace(
        model="draft",
        hf_config=SimpleNamespace(
            model_type="dflash",
            architectures=["DFlash2DraftModel" if dflash2 else "DFlashDraftModel"],
            dflash_config={
                "block_size": 16,
                "target_layer_ids": [0, 2],
                "mask_token_id": 7,
            },
        ),
    )
    config = SimpleNamespace(
        model_config=draft,
        speculative_config=SimpleNamespace(
            target_model_config=SimpleNamespace(
                model="target", hf_config=SimpleNamespace(model_type=model_type)
            ),
            draft_model_config=draft,
            num_speculative_tokens=15,
        ),
        additional_config={"override_qaic_config": {"prefill_seq_len": 128}},
    )
    _derive_dflash_config(config)
    derived = config.additional_config["dflash_cfg"]
    assert derived["tlm_model_type"] == model_type
    assert derived["target_layer_ids"] == ([0, 2] if dflash2 else [1, 3])
    assert derived.get("is_dflash2", False) == dflash2
    _derive_dflash_config(config)
    assert config.additional_config["dflash_cfg"] is derived
