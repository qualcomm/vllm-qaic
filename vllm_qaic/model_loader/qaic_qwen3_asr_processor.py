# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC processor for Qwen3-ASR raw-feature QPCs."""

from collections.abc import Mapping

import torch

from transformers import BatchFeature

from vllm.model_executor.models.qwen3_asr import (
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalFieldConfig


class QaicQwen3ASRMultiModalProcessor(Qwen3ASRMultiModalProcessor):
    """Keep native Qwen mel features instead of flattened embeddings."""

    def _call_hf_processor(self, *args, **kwargs):
        hf_inputs = super()._call_hf_processor(*args, **kwargs)
        if "input_audio_features" in hf_inputs and "input_features" not in hf_inputs:
            hf_inputs["input_features"] = hf_inputs.pop("input_audio_features")
        if "feature_attention_mask" in hf_inputs:
            mask = hf_inputs["feature_attention_mask"]
            if isinstance(mask, torch.Tensor) and mask.dtype != torch.int32:
                hf_inputs["feature_attention_mask"] = mask.to(torch.int32)
            hf_inputs["input_features_mask"] = hf_inputs["feature_attention_mask"]
        return hf_inputs

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        fields = {
            "input_features": MultiModalFieldConfig.batched("audio"),
            "feature_attention_mask": MultiModalFieldConfig.batched("audio"),
            "input_features_mask": MultiModalFieldConfig.batched("audio"),
        }
        if "audio_feature_lengths" in hf_inputs:
            fields["audio_feature_lengths"] = MultiModalFieldConfig.batched("audio")
        return fields


QAIC_QWEN3_ASR_PROCESSOR = (
    QaicQwen3ASRMultiModalProcessor,
    Qwen3ASRProcessingInfo,
    Qwen3ASRDummyInputsBuilder,
    Qwen3ASRForConditionalGeneration,
)
