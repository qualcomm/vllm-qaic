# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device unit tests for QAIC accuracy-relevant inference.

The full accuracy suite (test_accuracy.py) streams the MLPerf Llama2 dataset
and scores generations against reference answers with ROUGE (via the
optional `evaluate`/`nltk` dependencies) — a real accuracy-eval harness, not
a lightweight check. Pure-Python coverage of the quantization config that
feeds accuracy runs (mxfp6/mxint8 normalisation, QaicQuantConfig) lives in
unit/accuracy/test_accuracy_config.py.

This file is the missing on-device-unit layer in between: confirms that the
exact quantization settings the accuracy CI job runs with (mxfp6 matmul,
mxint8 kv-cache) actually produce a correct, deterministic answer for a
simple factual prompt on real hardware — a fast proxy for "accuracy didn't
silently regress" without requiring the MLPerf dataset or rouge/nltk.

Coverage areas
--------------
1. A simple factual prompt produces the expected answer under mxfp6/mxint8
2. That same prompt is deterministic (temperature=0) across repeated calls
"""

import pytest
from vllm import SamplingParams


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestAccuracyOnDevice:
    def test_factual_prompt_produces_expected_answer(self, qaic_model):
        out = qaic_model.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0.0, max_tokens=5),
        )
        _, texts = out[0]
        assert "Paris" in texts[0], (
            f"Expected 'Paris' in output under mxfp6/mxint8, got: {texts[0]!r}"
        )

    def test_factual_prompt_is_deterministic(self, qaic_model):
        prompt = ["The capital of France is"]
        params = SamplingParams(temperature=0.0, max_tokens=5)
        first = qaic_model.generate(prompt, params)[0][1][0]
        second = qaic_model.generate(prompt, params)[0][1][0]
        assert first == second, (
            "Greedy decode of the same prompt must be deterministic"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
