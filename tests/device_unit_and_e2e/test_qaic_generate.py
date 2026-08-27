# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import random

import pytest


class TestQaicGenerate:
    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
    def test_generate(self, qaic_model, sampling_params, sample_prompts):
        prompts = list(sample_prompts)
        output_dict: dict[str, list[str]] = {p: [] for p in prompts}

        for _ in range(5):
            random.shuffle(prompts)
            output = qaic_model.generate(prompts, sampling_params)
            assert len(output) == len(prompts), (
                "Number of generated outputs does not match the number of prompts!!"
            )
            for prompt, (_, texts) in zip(prompts, output, strict=False):
                output_dict[prompt].append(prompt + texts[0])

        for prompt, generations in output_dict.items():
            assert len(set(generations)) == 1, (
                "Outputs from different slots for same prompt does not match!!"
            )
