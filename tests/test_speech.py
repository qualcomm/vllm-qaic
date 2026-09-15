# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import importlib.metadata

import numpy as np
import torch
from transformers import BatchFeature


def test_general_plugin_remains_independent_of_cohere_asr():
    plugins = {
        entry_point.name: entry_point.value
        for entry_point in importlib.metadata.entry_points(group="vllm.general_plugins")
    }
    assert plugins["qaic_kv_connector"] == "vllm_qaic:register_connector"
    assert "qaic" not in plugins


def test_cohere_asr_processor_extracts_audio_once(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.models.cohere_asr import CohereASRMultiModalProcessor
    from vllm_qaic.model_loader.qaic_cohere_asr_processor import (
        QaicCohereASRMultiModalProcessor,
    )

    calls = {"vllm": 0, "native": 0}

    def tokenize_only(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        calls["vllm"] += 1
        assert mm_data == {}
        return BatchFeature({"input_ids": torch.tensor([[7]])})

    class NativeFeatureExtractor:
        sampling_rate = 16_000
        max_audio_clip_s = 35.0
        overlap_chunk_second = 5.0

        def __call__(self, audios, **kwargs):
            calls["native"] += 1
            assert len(audios) == 1
            return BatchFeature(
                {
                    "input_features": torch.ones((1, 4, 128)),
                    "attention_mask": torch.ones((1, 4), dtype=torch.int64),
                }
            )

    monkeypatch.setattr(
        CohereASRMultiModalProcessor, "_call_hf_processor", tokenize_only
    )
    processor = object.__new__(QaicCohereASRMultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_config=lambda: SimpleNamespace(max_audio_clip_s=35.0)
    )
    feature_extractor = NativeFeatureExtractor()
    processor.__dict__["_qaic_hf_processor"] = SimpleNamespace(
        feature_extractor=feature_extractor
    )

    outputs = processor._call_hf_processor(
        "prompt",
        {"audios": [np.zeros(16_000, dtype=np.float32)]},
        {},
        {},
    )

    assert calls == {"vllm": 1, "native": 1}
    assert outputs["input_features"].shape == (1, 128, 4)
    np.testing.assert_array_equal(outputs["length"], [4])
    assert feature_extractor.overlap_chunk_second == 5.0
