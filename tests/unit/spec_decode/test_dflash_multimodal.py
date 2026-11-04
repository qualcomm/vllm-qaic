# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from vllm_qaic.model_loader.qaic import QaicCausalLM
from vllm_qaic.model_loader.qaic_multimodal import QaicMultiModal
from vllm_qaic.platform_base import QaicPlatform


@pytest.mark.parametrize("nested_block_size", [False, True])
@pytest.mark.parametrize("fail_generate", [False, True])
def test_simple_vlm_example(monkeypatch, nested_block_size, fail_generate):
    from PIL import Image

    path = Path(__file__).resolve().parents[3] / "examples/qaic_spd_dflash_vlm.py"
    main = runpy.run_path(str(path))["main"]
    config = SimpleNamespace(
        **(
            {"dflash_config": {"block_size": 16}}
            if nested_block_size
            else {"block_size": 16}
        )
    )
    grid = np.array([[1, 28, 28]])
    processor = Mock(return_value={"image_grid_thw": grid})
    processor.apply_chat_template.return_value = "formatted image prompt"
    embeddings = np.zeros((1, 196, 32), dtype=np.float32)
    vision, target = Mock(), Mock()
    vision.encode.return_value = [
        SimpleNamespace(outputs=SimpleNamespace(data=embeddings))
    ]
    target.generate.return_value = [
        SimpleNamespace(outputs=[SimpleNamespace(text="response", token_ids=[1, 2])])
    ]
    llm = Mock(side_effect=[vision, target])
    monkeypatch.setitem(main.__globals__, "LLM", llm)
    monkeypatch.setitem(
        main.__globals__, "AutoConfig", Mock(from_pretrained=Mock(return_value=config))
    )
    monkeypatch.setitem(
        main.__globals__,
        "AutoProcessor",
        Mock(from_pretrained=Mock(return_value=processor)),
    )
    monkeypatch.setitem(
        main.__globals__, "fetch_image", lambda _: Image.new("RGB", (448, 448))
    )
    if fail_generate:
        target.generate.side_effect = RuntimeError("generation failed")
        with pytest.raises(RuntimeError, match="generation failed"):
            main()
    else:
        main()
    vision_config, target_config = [call.kwargs for call in llm.call_args_list]
    assert vision_config["runner"] == "pooling"
    assert target_config["model"] == "Qwen/Qwen3.5-4B"
    assert target_config["max_num_seqs"] == 1
    assert target_config["speculative_config"]["num_speculative_tokens"] == 15
    assert "enable_thinking" not in processor.apply_chat_template.call_args.kwargs
    assert processor.apply_chat_template.call_args.kwargs["add_generation_prompt"]
    request = target.generate.call_args.args[0][0]
    assert request["prompt"] == "formatted image prompt"
    assert request["multi_modal_data"]["image"]["image_embeds"] is embeddings
    assert request["multi_modal_data"]["image"]["image_grid_thw"] is grid
    vision.llm_engine.engine_core.shutdown.assert_called_once()
    target.llm_engine.engine_core.shutdown.assert_called_once()


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
def test_dflash_rejects_kv_transfer_before_config_mutation(role):
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_role=role),
        speculative_config=SimpleNamespace(method="dflash"),
    )
    with pytest.raises(ValueError, match="without kv_transfer_config"):
        QaicPlatform.check_and_update_config(config)


def _vlm_model(model_type="qwen3_vl", missing_phase=None):
    hybrid = model_type in {"qwen3_5", "qwen3_5_moe"}
    width = 32 if hybrid else 16
    shapes = {
        "input_ids": [
            [1, n]
            for phase, n in [("prefill", 128), ("decode", width)]
            if phase != missing_phase
        ],
        "logits": [[1, 1, 32], [1, 16, 32]],
    }
    inputs = {}
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type=model_type,
            text_config=SimpleNamespace(layer_types=[], sliding_window=1024),
        ),
        session=SimpleNamespace(
            input_names=["image_idx", "mm_token_type_ids", "cache_commit"],
            cluster_id=None,
            binding_index_map={"input_ids": 0, "logits": 1},
            allowed_shapes=[
                [(None, shape), (None, [1, 1 if shape[1] == 128 else 16, 32])]
                for shape in shapes["input_ids"]
            ],
            get_bindings_shapes=lambda names: {n: shapes.get(n, []) for n in names},
        ),
        get_io_shape_and_dtype=lambda *a, **kw: None,
        mm_input_info={},
        mm_output_info={},
        decode_bsz=1,
        full_batch_size=1,
        decode_batch_inputs=inputs,
        decode_batch_inputs_by_k={15: inputs},
        is_dflash_vlm=True,
        dflash_hybrid=hybrid,
        vocab_size=32,
        ctx_len=4096,
        prefill_seq_len=128,
        num_spec_tokens=15,
        tlm_hidden_size=8,
        _prefill_hidden_scratch=np.zeros((1, 128, 8), dtype=np.float32),
    )
    model._get_binding_shapes = MethodType(QaicCausalLM._get_binding_shapes, model)
    model._load_multimodal_dflash_hybrid = MethodType(
        QaicMultiModal._load_multimodal_dflash_hybrid, model
    )
    return model


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_vl", "gemma4"])
@pytest.mark.parametrize("missing_phase", [None, "prefill", "decode"])
@pytest.mark.parametrize("token_type_mask", [False, True])
def test_vlm_requires_both_language_specializations(
    model_type, missing_phase, token_type_mask
):
    model = _vlm_model(model_type, missing_phase)
    if not token_type_mask:
        model.session.input_names.remove("mm_token_type_ids")
    if missing_phase:
        with pytest.raises(ValueError, match="input lengths"):
            QaicMultiModal._load_multimodal(model)
        return
    QaicMultiModal._load_multimodal(model)
    inputs = model.decode_batch_inputs
    width = 32 if model.dflash_hybrid else 16
    if token_type_mask:
        assert inputs["mm_token_type_ids"].shape == (1, width)
    else:
        assert "mm_token_type_ids" not in inputs
    if model.dflash_hybrid:
        assert inputs["input_ids"].shape == (1, 32)
        assert inputs["position_ids"].shape == (4, 1, 32)
        assert np.all(inputs["position_ids"] == -1)
        assert inputs["cache_commit"].item() == 0
        assert model.default_mm_kwargs["cache_commit"].item() == 1
        assert model._dflash_prefix_ids.shape == (1, 16)
        assert model._dflash_prefix_positions.shape == (4, 1, 16)
        assert np.all(model._dflash_prefix_positions == -1)
        assert model._dflash_decode_hidden.shape == (1, 32, 8)
        assert model._dflash_decode_hidden.dtype == model._prefill_hidden_scratch.dtype
    else:
        assert "cache_commit" not in inputs
        assert not hasattr(model, "_dflash_prefix_ids")


@pytest.mark.parametrize("is_input", [True, False])
@pytest.mark.parametrize("shapes", [[], [[1, 16, 8], [1, 128, 8]]])
def test_dflash_binding_shapes_prefer_specializations_then_fallback(is_input, shapes):
    model = _vlm_model()
    model.session.get_bindings_shapes = lambda names: {names[0]: shapes}
    calls = []

    def binding(name, is_input=True):
        calls.append((name, is_input))
        return ([2048], np.float32, 8192)

    model.get_io_shape_and_dtype = binding
    assert model._get_binding_shapes("features", is_input) == (shapes or [[2048]])
    assert calls == ([] if shapes else [("features", is_input)])
    model.get_io_shape_and_dtype = lambda *a, **kw: None
    assert model._get_binding_shapes("missing", is_input) == shapes


@pytest.mark.parametrize("shapes", [[], [[8]], [[1, 16, 8], [1, 128, 9]], [[1, 16, 8]]])
def test_dflash_hidden_width_validation(shapes):
    model = _vlm_model()
    model.tlm_hidden_states_name = "features"
    model.session.get_bindings_shapes = lambda names: {"features": shapes}
    if shapes == [[1, 16, 8]]:
        assert QaicCausalLM._get_tlm_hidden_size(model) == 8
    else:
        with pytest.raises(ValueError, match="hidden states"):
            QaicCausalLM._get_tlm_hidden_size(model)


@pytest.mark.parametrize("shapes", [[], [[1, 16]], [[1, 16, 1]], [[1, 16, 31]]])
def test_vlm_requires_vocabulary_logits(shapes):
    model = _vlm_model()
    original = model.session.get_bindings_shapes
    model.session.get_bindings_shapes = lambda names: (
        {"logits": shapes} if names == ["logits"] else original(names)
    )
    with pytest.raises(ValueError, match="vocabulary logits"):
        QaicMultiModal._load_multimodal(model)


@pytest.mark.parametrize("name", ["past_key.0", "past_key.0_RetainedState"])
@pytest.mark.parametrize("ctx_len", [512, 4096])
@pytest.mark.parametrize("headroom", [-1, 0])
def test_gemma4_sliding_cache_capacity(name, ctx_len, headroom):
    model = _vlm_model("gemma4")
    model.ctx_len = ctx_len
    model.config.text_config.layer_types = ["sliding_attention"]
    model.session.input_names.append(name)
    capacity = min(ctx_len, 1024 + 128 - 1)
    model.get_io_shape_and_dtype = lambda n, is_input=True: (
        ([1, 2, capacity + headroom, 8], np.float32, 0) if n == name else None
    )
    if headroom < 0:
        with pytest.raises(ValueError, match="capacity"):
            QaicMultiModal._load_multimodal(model)
    else:
        QaicMultiModal._load_multimodal(model)


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize(
    "missing", ["cache_commit", "image_idx", "hidden-state buffer"]
)
def test_qwen35_requires_state_bindings_and_hidden_buffer(model_type, missing):
    model = _vlm_model(model_type)
    model.default_mm_kwargs = None
    if missing == "hidden-state buffer":
        model._prefill_hidden_scratch = None
    else:
        model.session.input_names.remove(missing)
    with pytest.raises(ValueError, match=missing):
        QaicMultiModal._load_multimodal(model)


@pytest.mark.parametrize("length", [[128], (128,)])
def test_vlm_rejects_multiple_prefill_lengths(length):
    model = _vlm_model()
    model.prefill_seq_len = length
    with pytest.raises(ValueError, match="single integer prefill_seq_len"):
        QaicMultiModal._load_multimodal(model)


@pytest.mark.parametrize("accepted", [1, 7, 16])
@pytest.mark.parametrize("valid_tokens", [1, 9, 16])
def test_concatenated_decode_commits_prefix_and_returns_suffix(
    monkeypatch, accepted, valid_tokens
):
    model = QaicMultiModal.__new__(QaicMultiModal)
    model.dflash_hybrid = True
    model.num_spec_tokens = 15
    model.active_k = 15
    model._dflash_prefix_ids = np.arange(16).reshape(1, 16)
    model._dflash_prefix_positions = np.full((4, 1, 16), -1, dtype=np.int64)
    model._dflash_prefix_positions[:, :, :accepted] = np.arange(accepted)
    model._dflash_decode_hidden = np.zeros((1, 32, 8), dtype=np.float32)
    current_ids = np.arange(100, 116)
    positions = np.tile(np.arange(accepted, accepted + 16), (4, 1))
    positions[:, valid_tokens:] = -1
    calls = []

    def run(self, ids, pids, slots, logits, lora, callback, hidden):
        calls.append((ids.copy(), pids.copy()))
        self.decode_batch_inputs_by_k = {
            15: {
                "input_ids": ids.reshape(1, 32),
                "position_ids": pids.reshape(4, 1, 32),
            }
        }
        hidden[:, :16] = 1
        hidden[:, 16:] = 2
        logits[:] = np.arange(16).reshape(1, 16, 1)

    monkeypatch.setattr(QaicCausalLM, "_run_decode", run)
    logits = np.empty((1, 16, 32), dtype=np.float32)
    hidden = np.empty((1, 16, 8), dtype=np.float32)
    model._run_decode(
        current_ids, positions, np.array([0]), logits, dflash_decode_hidden_buf=hidden
    )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][16:], current_ids)
    np.testing.assert_array_equal(
        calls[0][1][:, :16], model._dflash_prefix_positions[:, 0]
    )
    assert np.all(hidden == 2)
    np.testing.assert_array_equal(
        logits[0, :valid_tokens, 0], np.arange(16 - valid_tokens, 16)
    )
    model.queue_dflash_prefix([list(range(valid_tokens))])
    np.testing.assert_array_equal(model._dflash_prefix_ids[0], current_ids)
    np.testing.assert_array_equal(model._dflash_prefix_positions[:, 0], positions)
    assert model._dflash_decode_slots is None
