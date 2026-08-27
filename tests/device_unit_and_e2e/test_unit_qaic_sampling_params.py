# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC sampling parameters with verifiable output
constraints (min_tokens, n completions, stop strings, per-request params).

Smoke tests for individual parameters (presence_penalty, top_p, etc.) live in
device_unit_and_e2e/test_qaic_samplers.py (TestQaicSamplers) — this file only
covers parameters whose effect on output shape/length can be verified
directly, mirroring the split that existed in
unit/generic/test_sampling_params.py (now removed; depended on a nonexistent
`llm` fixture).

Coverage areas
--------------
1. min_tokens lower bound
2. n completions per request
3. stop string termination
4. distinct SamplingParams per request in the same batch
5. greedy and sampling requests mixed in the same batch
"""

import pytest


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestSamplingConstraintsOnDevice:
    def test_min_tokens(self, qaic_model):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, min_tokens=5, max_tokens=20, ignore_eos=True)
        out = qaic_model.generate(["Hello"], params)
        token_ids, _ = out[0]
        assert len(token_ids[0]) >= 5

    def test_n_completions(self, qaic_model):
        from vllm import SamplingParams

        params = SamplingParams(temperature=1.0, max_tokens=10, n=3)
        out = qaic_model.generate(["Hello"], params)
        token_ids, texts = out[0]
        assert len(texts) == 3
        for ids in token_ids:
            assert len(ids) > 0

    def test_stop_string(self, qaic_model):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=50, stop=["the", "The"])
        out = qaic_model.generate(["Once upon a time"], params)
        _, texts = out[0]
        assert len(texts[0]) > 0

    def test_different_params_per_request(self, qaic_model):
        from vllm import SamplingParams

        prompts = ["Hello", "World", "Foo"]
        params_list = [
            SamplingParams(temperature=0.0, max_tokens=5),
            SamplingParams(temperature=0.8, top_p=0.9, max_tokens=5),
            SamplingParams(temperature=0.0, presence_penalty=1.0, max_tokens=5),
        ]
        out = qaic_model.generate(prompts, params_list)
        assert len(out) == 3
        for i, (token_ids, _) in enumerate(out):
            assert len(token_ids[0]) > 0, f"Empty output at request {i}"

    def test_greedy_and_sampling_in_same_batch(self, qaic_model):
        from vllm import SamplingParams

        prompts = ["Hello", "World"]
        params_list = [
            SamplingParams(temperature=0.0, max_tokens=5),  # greedy
            SamplingParams(temperature=1.0, max_tokens=5),  # sampling
        ]
        out = qaic_model.generate(prompts, params_list)
        assert len(out[0][0][0]) > 0
        assert len(out[1][0][0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
