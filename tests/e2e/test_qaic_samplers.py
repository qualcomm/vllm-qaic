# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import json
import random

import pytest
from vllm import SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from vllm_qaic.utils.qaic_utils import _clean_config


@pytest.mark.qaic_test_config(
    {
        "aot": dict(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ctx_len=256,
            dtype="mxfp6",
            kv_dtype="mxint8",
        ),
        "eager": dict(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ctx_len=256,
        ),
    }
)
class TestQaicSamplers:
    def test_presence_penalty(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=None, presence_penalty=0.2
        )

        random.shuffle(prompt)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_frequency_penalty(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=None, frequency_penalty=0.1
        )

        random.shuffle(prompt)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_repetition_penalty(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=None, repetition_penalty=1.3
        )

        random.shuffle(prompt)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_topp(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = SamplingParams(temperature=0.0, max_tokens=None, top_p=0.95)

        random.shuffle(prompt)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_topk(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = SamplingParams(temperature=0.0, max_tokens=None, top_k=5)

        random.shuffle(prompt)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_minp(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = SamplingParams(temperature=0.0, max_tokens=None, min_p=0.4)

        random.shuffle(prompt)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_random_samplers(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:6]
        outputDict = {p: [] for p in prompt}

        sampling_params = [
            SamplingParams(temperature=0.0, max_tokens=None, top_p=0.95),
            SamplingParams(temperature=0.0, max_tokens=None, min_p=0.4),
            SamplingParams(temperature=0.0, max_tokens=None, top_k=5),
            SamplingParams(temperature=0.0, max_tokens=None, repetition_penalty=1.3),
            SamplingParams(temperature=0.0, max_tokens=None, frequency_penalty=0.1),
            SamplingParams(temperature=0.0, max_tokens=None, presence_penalty=0.2),
        ]

        random.shuffle(prompt)
        random.shuffle(sampling_params)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"

    def test_mixed_samplers(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:4]
        outputDict = {p: [] for p in prompt}

        sampling_params = [
            SamplingParams(
                temperature=0.0, max_tokens=None, top_p=0.95, frequency_penalty=0.1
            ),
            SamplingParams(
                temperature=0.0, max_tokens=None, top_k=5, presence_penalty=0.2
            ),
            SamplingParams(
                temperature=0.0, max_tokens=None, top_p=0.95, repetition_penalty=1.3
            ),
            SamplingParams(
                temperature=0.0, max_tokens=None, min_p=0.3, frequency_penalty=0.1
            ),
        ]

        random.shuffle(prompt)
        random.shuffle(sampling_params)
        output = qaic_model.generate(prompt, sampling_params)
        for p, (_, texts) in zip(prompt, output, strict=False):
            outputDict[p].append(p + texts[0])

        for key in outputDict:
            assert outputDict[key] is not None, "Output text not generated!!!"


@pytest.mark.qaic_test_config(
    {
        "aot": dict(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            ctx_len=256,
            dtype="mxfp6",
            kv_dtype="mxint8",
        ),
    }
)
class TestQaicGuidedDecoding:
    """Guided decoding with host sampling or an ODS QPC compiled with masks."""

    @pytest.fixture(autouse=True)
    def _require_guided_override(self, override_qaic_config):
        config = _clean_config(dict(override_qaic_config or {}))
        if config.get("aic_include_sampler") and not config.get(
            "aic_include_guided_decoding"
        ):
            pytest.skip(
                "ODS needs aic_include_guided_decoding in --override-qaic-config"
            )

    @pytest.mark.parametrize("temperature", [0.0, 1.0])
    def test_choice_per_request(self, qaic_model, sample_prompts, temperature):
        prompts = sample_prompts[:4]
        # Different choices per request catch slot-to-row mapping mistakes.
        choices = [
            ["yes", "no"],
            ["red", "green", "blue"],
            ["Paris", "London"],
            ["0", "1"],
        ]
        params = [
            SamplingParams(
                temperature=temperature,
                max_tokens=8,
                structured_outputs=StructuredOutputsParams(choice=allowed),
            )
            for allowed in choices
        ]
        outputs = qaic_model.generate(prompts, params)
        # generate() returns prompt + completion; compare only the completion.
        for prompt, (_, texts), allowed in zip(prompts, outputs, choices, strict=True):
            assert texts[0][len(prompt) :] in allowed

    def test_json_schema(self, qaic_model, sample_prompts):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        params = SamplingParams(
            temperature=1.0,
            max_tokens=64,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
        prompt = sample_prompts[0]
        [(_, texts)] = qaic_model.generate([prompt], [params])
        completion = texts[0][len(prompt) :]
        assert json.loads(completion) in ({"answer": "yes"}, {"answer": "no"})
