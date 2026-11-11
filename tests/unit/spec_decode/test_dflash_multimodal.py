# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import runpy
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from vllm_qaic.model_loader.qaic import UINT16_MAX, QaicCausalLM
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


def _vlm_model(model_type="qwen3_vl", missing_phase=None, batch_size=1):
    hybrid = model_type in {"qwen3_5", "qwen3_5_moe"}
    width = 32 if hybrid else 16
    shapes = {
        "input_ids": [
            [1 if phase == "prefill" else batch_size, n]
            for phase, n in [("prefill", 128), ("decode", width)]
            if phase != missing_phase
        ],
        "logits": [[1, 1, 32], [batch_size, 16, 32]],
    }
    inputs = {}
    model = SimpleNamespace(
        config=SimpleNamespace(
            model_type=model_type,
            text_config=SimpleNamespace(layer_types=[], sliding_window=1024),
        ),
        session=SimpleNamespace(
            input_names=[
                "image_idx",
                "mm_token_type_ids",
                "cache_commit",
                "batch_index",
            ],
            cluster_id=None,
            binding_index_map={"input_ids": 0, "logits": 1},
            allowed_shapes=[
                [(None, shape), (None, [shape[0], 1 if shape[1] == 128 else 16, 32])]
                for shape in shapes["input_ids"]
            ],
            get_bindings_shapes=lambda names: {n: shapes.get(n, []) for n in names},
        ),
        get_io_shape_and_dtype=lambda *a, **kw: None,
        mm_input_info={},
        mm_output_info={},
        decode_bsz=batch_size,
        full_batch_size=batch_size,
        decode_batch_inputs=inputs,
        decode_batch_inputs_by_k={15: inputs},
        is_dflash_vlm=True,
        dflash_hybrid=hybrid,
        vocab_size=32,
        ctx_len=4096,
        prefill_seq_len=128,
        num_spec_tokens=15,
        max_decode_tokens=16,
        tlm_hidden_size=8,
        _prefill_hidden_scratch=np.zeros((1, 128, 8), dtype=np.float32),
    )
    model._get_binding_shapes = MethodType(QaicCausalLM._get_binding_shapes, model)
    model._load_multimodal_dflash_vlm = MethodType(
        QaicMultiModal._load_multimodal_dflash_vlm, model
    )
    model._load_multimodal_dflash_hybrid = MethodType(
        QaicMultiModal._load_multimodal_dflash_hybrid, model
    )
    return model


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_vl", "gemma4"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_dflash_decode_k_excludes_hybrid_replay(model_type, batch_size):
    model = _vlm_model(model_type, batch_size=batch_size)
    model.is_dflash_target = True
    model.decode_ks = [model.num_spec_tokens]

    assert QaicCausalLM._decode_ks_from_session(model) == [15]


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_vl", "gemma4"])
@pytest.mark.parametrize("missing_phase", [None, "prefill", "decode"])
@pytest.mark.parametrize("token_type_mask", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_vlm_requires_both_language_specializations(
    model_type, missing_phase, token_type_mask, batch_size
):
    model = _vlm_model(model_type, missing_phase, batch_size)
    if not token_type_mask:
        model.session.input_names.remove("mm_token_type_ids")
    if missing_phase:
        with pytest.raises(ValueError, match="input shapes"):
            QaicMultiModal._load_multimodal(model)
        return
    QaicMultiModal._load_multimodal(model)
    inputs = model.decode_batch_inputs
    width = 32 if model.dflash_hybrid else 16
    if token_type_mask:
        assert inputs["mm_token_type_ids"].shape == (batch_size, width)
    else:
        assert "mm_token_type_ids" not in inputs
    if model.dflash_hybrid:
        assert inputs["input_ids"].shape == (batch_size, 32)
        assert inputs["position_ids"].shape == (4, batch_size, 32)
        assert np.all(inputs["position_ids"] == -1)
        assert inputs["cache_commit"].item() == 0
        assert model.default_mm_kwargs["cache_commit"].item() == 1
        assert model._dflash_prefix_ids.shape == (batch_size, 16)
        assert model._dflash_prefix_positions.shape == (4, batch_size, 16)
        assert np.all(model._dflash_prefix_positions == -1)
        assert model._dflash_decode_hidden.shape == (batch_size, 32, 8)
        assert model._dflash_decode_hidden.dtype == model._prefill_hidden_scratch.dtype
    else:
        assert "cache_commit" not in inputs
        assert not hasattr(model, "_dflash_prefix_ids")


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe", "qwen3_vl", "gemma4"])
def test_vlm_batched_target_requires_cache_slot_input(model_type):
    model = _vlm_model(model_type, batch_size=2)
    model.session.input_names.remove("batch_index")
    with pytest.raises(ValueError, match="requires batch_index"):
        QaicMultiModal._load_multimodal(model)


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_batched_hybrid_decode_keeps_request_slots_and_masks_idle_rows(model_type):
    model = _vlm_model(model_type, batch_size=4)
    QaicMultiModal._load_multimodal(model)
    model.active_k = 15
    model.uses_mrope = True
    model.ignore_batch_index = False
    model.is_spec_decode_target_model = True
    model.comp_ctx_lengths_decode = None
    model.use_async_scheduling = False
    model.tlm_hidden_states_name = "hidden_states"
    model.decode_num_logits_buffer_by_k = {}
    model.session.np_run = Mock(return_value=0)
    model.session.complete_inf = Mock()
    feeds = model.decode_batch_inputs_by_k[15]
    feeds["batch_index"] = np.zeros((4, 1), dtype=np.int64)
    model._dflash_prefix_ids[:] = np.arange(64).reshape(4, 16)
    model._dflash_prefix_positions[:, 2, :2] = [18, 19]
    model._dflash_prefix_positions[:, 0, :5] = np.arange(55, 60)
    previous = model._dflash_prefix_positions.copy()
    ids = np.arange(100, 132).reshape(2, 16)
    positions = np.tile(np.stack([np.arange(20, 36), np.arange(60, 76)]), (4, 1, 1))
    buffers = {name: feeds[name] for name in ("input_ids", "position_ids")}

    def run_decode(ids, positions, slots):
        with patch(
            "numpy.concatenate", side_effect=AssertionError("Use decode buffers")
        ):
            QaicCausalLM._run_decode(
                model,
                ids.reshape(-1),
                positions.reshape(4, -1),
                np.array(slots),
                np.zeros((4, 16, 32), dtype=np.float32),
            )
        for name, buffer in buffers.items():
            assert feeds[name] is buffer

    run_decode(ids, positions, [2, 0])
    model.session.np_run.assert_called_once()
    np.testing.assert_array_equal(
        feeds["batch_index"].ravel(), [2, 0, UINT16_MAX, UINT16_MAX]
    )
    np.testing.assert_array_equal(feeds["input_ids"][:2, -16:], ids)
    np.testing.assert_array_equal(
        feeds["position_ids"][:, :2, :16], previous[:, [2, 0]]
    )
    assert np.all(feeds["position_ids"][:, 2:] == -1)
    assert feeds["cache_commit"].item() == 0
    QaicMultiModal.queue_dflash_prefix(model, [[1] * 3, [1] * 7])
    for row, (slot, count) in enumerate([(2, 3), (0, 7)]):
        np.testing.assert_array_equal(model._dflash_prefix_ids[slot], ids[row])
        np.testing.assert_array_equal(
            model._dflash_prefix_positions[:, slot, :count], positions[:, row, :count]
        )
        assert np.all(model._dflash_prefix_positions[:, slot, count:] == -1)
    np.testing.assert_array_equal(
        model._dflash_prefix_positions[:, [1, 3]], previous[:, [1, 3]]
    )
    # Reuse the buffers with fewer rows and a different first cache slot.
    run_decode(ids[:1], positions[:, :1], [0])
    np.testing.assert_array_equal(feeds["input_ids"][0, :16], ids[1])
    np.testing.assert_array_equal(feeds["input_ids"][0, 16:], ids[0])
    np.testing.assert_array_equal(
        feeds["position_ids"][:, 0, :16], model._dflash_prefix_positions[:, 0]
    )
    np.testing.assert_array_equal(feeds["position_ids"][:, 0, 16:], positions[:, 0])
    assert np.all(feeds["input_ids"][1:] == -1)
    assert np.all(feeds["position_ids"][:, 1:] == -1)
    np.testing.assert_array_equal(
        feeds["batch_index"].ravel(), [0, UINT16_MAX, UINT16_MAX, UINT16_MAX]
    )


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


@pytest.mark.parametrize("accepted", [0, 1, 7, 16])
@pytest.mark.parametrize("valid_tokens", [0, 1, 9, 16])
@pytest.mark.parametrize("slot", [0, 2])
@pytest.mark.parametrize("capture_hidden", [False, True])
@pytest.mark.parametrize("has_num_logits_input", [False, True])
def test_concatenated_decode_commits_prefix_and_returns_suffix(
    accepted, valid_tokens, slot, capture_hidden, has_num_logits_input
):
    model = QaicMultiModal.__new__(QaicMultiModal)
    model.dflash_hybrid = True
    model.num_spec_tokens = 15
    model.max_decode_tokens = 16
    model.active_k = 15
    model.decode_bsz = 1
    model.uses_mrope = True
    model.ignore_batch_index = False
    model.is_spec_decode_target_model = True
    model.comp_ctx_lengths_decode = [8, 32]
    model.list_of_comp_ctx_lengths = {n: np.empty(n) for n in (8, 32)}
    model.use_async_scheduling = False
    model.tlm_hidden_states_name = "tlm_hidden_states"
    model.decode_batch_inputs_by_k = {
        15: {
            "input_ids": np.zeros((1, 32), dtype=np.int64),
            "position_ids": np.full((4, 1, 32), -1, dtype=np.int64),
            "batch_index": np.zeros((1, 1), dtype=np.int64),
            "cache_commit": np.zeros(1, dtype=np.int64),
        }
    }
    model.decode_num_logits_buffer_by_k = {
        15: {"num_logits_to_keep": np.zeros((16, 1), dtype=np.int64)}
    }
    model._dflash_prefix_ids = np.arange(48).reshape(3, 16)
    model._dflash_prefix_positions = np.full((4, 3, 16), -1, dtype=np.int64)
    model._dflash_prefix_positions[:, slot, :accepted] = np.arange(accepted)
    model._dflash_decode_hidden = np.zeros((1, 32, 8), dtype=np.float32)
    current_ids = np.arange(100, 116)
    positions = np.tile(np.arange(accepted, accepted + 16), (4, 1))
    positions[:, valid_tokens:] = -1
    callback = Mock()

    def complete(index, is_prefill):
        callback.assert_called_once_with()
        inputs = model.session.np_run.call_args.args[0]
        inputs["tlm_hidden_states"][:, :16] = 1
        inputs["tlm_hidden_states"][:, 16:] = 2
        inputs["logits"][:] = np.arange(16).reshape(1, 16, 1)

    model.session = SimpleNamespace(
        input_names=["num_logits_to_keep"] if has_num_logits_input else [],
        np_run=Mock(return_value=17),
        complete_inf=Mock(side_effect=complete),
    )
    logits = np.empty((1, 16, 32), dtype=np.float32)
    hidden = np.empty((1, 16, 8), dtype=np.float32) if capture_hidden else None
    batch_indices = np.array([slot])
    model._run_decode(
        current_ids,
        positions,
        batch_indices,
        logits,
        callback=callback,
        dflash_decode_hidden_buf=hidden,
    )
    assert QaicMultiModal._run_decode is QaicCausalLM._run_decode
    model.session.np_run.assert_called_once()
    assert model.session.np_run.call_args.kwargs == {"is_prefill": False}
    model.session.complete_inf.assert_called_once_with(17, is_prefill=False)
    inputs = model.session.np_run.call_args.args[0]
    np.testing.assert_array_equal(
        inputs["input_ids"][0, :16], model._dflash_prefix_ids[slot]
    )
    np.testing.assert_array_equal(inputs["input_ids"][0, 16:], current_ids)
    np.testing.assert_array_equal(
        inputs["position_ids"][:, 0, :16], model._dflash_prefix_positions[:, slot]
    )
    np.testing.assert_array_equal(inputs["position_ids"][:, 0, 16:], positions)
    assert inputs["batch_index"].item() == slot
    assert inputs["cache_commit"].item() == 0
    max_position = accepted + valid_tokens - 1
    assert inputs["comp_ctx_lengths"].size == (8 if max_position < 8 else 32)
    assert ("num_logits_to_keep" in inputs) == has_num_logits_input
    if has_num_logits_input:
        assert inputs["num_logits_to_keep"].shape == (16, 1)
    if capture_hidden:
        assert np.all(hidden == 2)
    np.testing.assert_array_equal(
        logits[0, :valid_tokens, 0], np.arange(16 - valid_tokens, 16)
    )
    batch_indices[:] = -1
    np.testing.assert_array_equal(model._dflash_decode_slots, [slot])
    model.queue_dflash_prefix([list(range(valid_tokens))])
    np.testing.assert_array_equal(model._dflash_prefix_ids[slot], current_ids)
    np.testing.assert_array_equal(model._dflash_prefix_positions[:, slot], positions)
    assert model._dflash_decode_slots is None


@pytest.mark.parametrize("mrope", [False, True])
@pytest.mark.parametrize("k", [0, 15])
@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("capture_hidden", [False, True])
@pytest.mark.parametrize("has_num_logits_input", [False, True])
def test_shared_decode_without_hybrid(
    mrope, k, async_scheduling, capture_hidden, has_num_logits_input
):
    model_cls = QaicMultiModal if mrope else QaicCausalLM
    model = model_cls.__new__(model_cls)
    if mrope:
        model.dflash_hybrid = False
    model.active_k = k
    model.decode_bsz = 2
    model.uses_mrope = mrope
    model.ignore_batch_index = False
    model.is_spec_decode_target_model = k > 0
    model.comp_ctx_lengths_decode = None
    model.use_async_scheduling = async_scheduling
    model.tlm_hidden_states_name = "tlm_hidden_states"
    width = k + 1
    inputs = {
        "input_ids": np.zeros((2, width), dtype=np.int64),
        "position_ids": np.zeros(
            (4, 2, width) if mrope else (2, width), dtype=np.int64
        ),
        "batch_index": np.zeros((2, 1), dtype=np.int64),
    }
    model.decode_batch_inputs_by_k = {k: inputs}
    model.decode_num_logits_buffer_by_k = {
        k: {"num_logits_to_keep": np.zeros((width, 1), dtype=np.int64)}
    }
    model.session = SimpleNamespace(
        input_names=["num_logits_to_keep"] if has_num_logits_input else [],
        np_run=Mock(return_value=17),
        complete_inf=Mock(),
    )
    ids = np.arange(width)
    positions = np.arange(20, 20 + width)
    if mrope:
        positions = np.tile(positions, (4, 1))
    logits = np.empty((2, width, 32), dtype=np.float32)
    hidden = np.empty((2, width, 8), dtype=np.float32) if capture_hidden else None
    callback = Mock()
    model._run_decode(
        ids,
        positions,
        np.array([3]),
        logits,
        callback=callback,
        dflash_decode_hidden_buf=hidden,
    )
    model.session.np_run.assert_called_once()
    assert model.session.np_run.call_args.args[0] is inputs
    assert ("num_logits_to_keep" in inputs) == (k > 0 and has_num_logits_input)
    callback.assert_called_once_with()
    if async_scheduling:
        model.session.complete_inf.assert_not_called()
    else:
        model.session.complete_inf.assert_called_once_with(17, is_prefill=False)
    np.testing.assert_array_equal(inputs["input_ids"][0], ids)
    np.testing.assert_array_equal(inputs["position_ids"][..., 0, :], positions)
    np.testing.assert_array_equal(inputs["batch_index"], [[3], [UINT16_MAX]])
    assert np.all(inputs["input_ids"][1] == -1)
    assert np.all(inputs["position_ids"][..., 1, :] == -1)
    assert inputs["logits"] is logits
    if capture_hidden:
        assert inputs["tlm_hidden_states"] is hidden
    else:
        assert "tlm_hidden_states" not in inputs
    assert not hasattr(model, "_dflash_decode_slots")
