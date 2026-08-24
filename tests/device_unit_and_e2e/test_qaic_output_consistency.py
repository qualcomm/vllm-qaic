# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import random

import pytest


class TestQaicOutputConsistency:
    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
    def test_output_consistency(self, qaic_model, sampling_params, sharegpt_prompts):
        prompts = sharegpt_prompts(1)
        output = qaic_model.generate(prompts, sampling_params)
        assert len(output) == len(prompts), (
            "Number of generated outputs does not match the number of valid inputs!!"
        )
        check_output = [texts[0] for _, texts in output]
        assert len(set(check_output)) == 1, (
            "Outputs from different slots for same prompt does not match!!"
        )

    @pytest.mark.qaic_test_config(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ctx_len=256,
        dtype="mxfp6",
        kv_dtype="mxint8",
    )
    def test_generate(self, qaic_model, sampling_params, sharegpt_prompts):
        prompts = sharegpt_prompts(10)
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
                "Same prompts have different outputs generated! Run-to-run "
                "generation variation"
            )
