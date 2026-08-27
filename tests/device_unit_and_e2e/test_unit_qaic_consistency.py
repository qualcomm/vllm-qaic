# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device unit tests for QAIC output consistency (determinism).

The full consistency suite (test_qaic_output_consistency.py) generates over
ShareGPT prompts across repeated shuffled batches to stress-test run-to-run
stability at scale. That heavier, larger-N scenario is appropriately
e2e-only.

This file is the missing on-device-unit layer: confirms the same guarantee
— greedy decode of an identical prompt must be bit-for-bit identical
regardless of what else shares its batch slot — holds for the smallest
possible case, using the lightweight sample_prompts fixture instead of a
ShareGPT/tokenizer download.

Coverage areas
--------------
1. The same prompt generates identical output when run twice
2. The same prompt generates identical output regardless of its position
   in the batch
"""

import pytest
from vllm import SamplingParams


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="mxfp6",
    kv_dtype="mxint8",
    decode_bsz=4,
)
class TestOutputConsistencyOnDevice:
    def test_repeated_generation_is_identical(self, qaic_model, sample_prompts):
        prompt = sample_prompts[:1]
        params = SamplingParams(temperature=0.0, max_tokens=8)
        first = qaic_model.generate(prompt, params)[0][1][0]
        second = qaic_model.generate(prompt, params)[0][1][0]
        assert first == second

    def test_output_independent_of_batch_position(self, qaic_model, sample_prompts):
        prompts = sample_prompts[:4]
        params = SamplingParams(temperature=0.0, max_tokens=8)

        forward = qaic_model.generate(prompts, params)
        reversed_prompts = list(reversed(prompts))
        backward = qaic_model.generate(reversed_prompts, params)

        forward_texts = {p: out[1][0] for p, out in zip(prompts, forward, strict=False)}
        backward_texts = {
            p: out[1][0] for p, out in zip(reversed_prompts, backward, strict=False)
        }
        assert forward_texts == backward_texts, (
            "Same prompt must produce the same output regardless of batch position"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
