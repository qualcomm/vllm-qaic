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

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            return super()._call_hf_processor(prompt, mm_data, mm_kwargs, tok_kwargs)

        # vLLM builds the request-specific decoder control prefix separately.
        # Run its processor only for prompt tokenization, then extract audio
        # once with the native HF feature extractor used by QEff.
        processed_outputs = super()._call_hf_processor(
            prompt, {}, mm_kwargs, tok_kwargs
        )

        feature_extractor = self._qaic_hf_processor.feature_extractor
        feature_extractor.max_audio_clip_s = self.info.get_hf_config().max_audio_clip_s
        audio_outputs = feature_extractor(
            mm_data["audios"],
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        processed_outputs["input_features"] = audio_outputs[
            "input_features"
        ].transpose(1, 2).contiguous()
        processed_outputs["length"] = audio_outputs["attention_mask"].sum(dim=-1)
        return processed_outputs


QAIC_COHERE_ASR_PROCESSOR = (
    QaicCohereASRMultiModalProcessor,
    CohereASRProcessingInfo,
    CohereASRDummyInputsBuilder,
    CohereAsrForConditionalGeneration,
)
