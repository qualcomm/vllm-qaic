# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from functools import cached_property

from transformers import AutoProcessor, BatchFeature

from vllm.model_executor.models.cohere_asr import (
    CohereASRDummyInputsBuilder,
    CohereASRMultiModalProcessor,
    CohereASRProcessingInfo,
    CohereAsrForConditionalGeneration,
)
from vllm.multimodal.parse import MultiModalDataItems


class QaicCohereASRMultiModalProcessor(CohereASRMultiModalProcessor):
    """Use Cohere's HF processor so QPC inputs match QEff export semantics."""

    @cached_property
    def _qaic_hf_processor(self):
        model_config = self.info.ctx.model_config
        return AutoProcessor.from_pretrained(
            model_config.model,
            revision=model_config.revision,
            trust_remote_code=False,
        )

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        valid_mm_items = mm_items.select(
            {key for key, count in mm_items.get_all_counts().items() if count > 0}
        )
        processor_data, passthrough_data = self._get_hf_mm_data(valid_mm_items)
        if not processor_data:
            return BatchFeature(dict(passthrough_data))

        processor_data, hf_processor_mm_kwargs = self._preprocess_hf_mm_data(
            processor_data, hf_processor_mm_kwargs
        )
        prompt = self._get_hf_processor_text(mm_items.get_all_counts())

        # vLLM builds the request-specific decoder control prefix separately.
        # Run its processor only for prompt tokenization, then extract audio
        # once with the native HF feature extractor used by QEff.
        processed_outputs = self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**hf_processor_mm_kwargs),
            {} if prompt is None else {"text": prompt},
            hf_processor_mm_kwargs,
        )

        feature_extractor = self._qaic_hf_processor.feature_extractor
        feature_extractor.max_audio_clip_s = self.info.get_hf_config().max_audio_clip_s
        audio_outputs = feature_extractor(
            processor_data["audio"],
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        processed_outputs["input_features"] = audio_outputs[
            "input_features"
        ].transpose(1, 2).contiguous()
        processed_outputs["length"] = audio_outputs["attention_mask"].sum(dim=-1)
        processed_outputs.update(passthrough_data)
        return self._postprocess_hf_mm_data(
            processor_data,
            hf_processor_mm_kwargs,
            processed_outputs,
        )


QAIC_COHERE_ASR_PROCESSOR = (
    QaicCohereASRMultiModalProcessor,
    CohereASRProcessingInfo,
    CohereASRDummyInputsBuilder,
    CohereAsrForConditionalGeneration,
)
