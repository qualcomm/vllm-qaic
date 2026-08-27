# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import random

import pytest
from vllm import SamplingParams


class TestQaicSamplers:
    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
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
