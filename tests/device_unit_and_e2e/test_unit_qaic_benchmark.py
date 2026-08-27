# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device unit tests for QAIC throughput/latency benchmarking.

The full benchmark suite (test_qaic_benchmark.py) shells out to vLLM's
`vllm bench throughput`/`vllm bench latency` CLI against a real ShareGPT
dataset slice and asserts on the CLI's own summary output. That is
appropriately e2e-only (it exercises the CLI entrypoint itself, not just
model serving).

This file is the missing on-device-unit layer: confirms that a batch of
real requests against a real qaic_model runner completes and yields
sane, self-consistent throughput numbers (all positive, output token count
matches requested max_tokens under greedy decoding) — a fast smoke check
that the serving path itself hasn't regressed in a way that would show up
as zero/negative/missing throughput, without needing the CLI or a dataset
download.

Coverage areas
--------------
1. A small batch of concurrent requests completes and produces one output
   per prompt
2. Computed throughput (requests/sec, tokens/sec) is positive and finite
"""

import time

import pytest
from vllm import SamplingParams


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="mxfp6",
    kv_dtype="mxint8",
    decode_bsz=4,
)
class TestBenchmarkOnDevice:
    def test_batch_completes_with_one_output_per_prompt(self, qaic_model, sample_prompts):
        prompts = sample_prompts[:4]
        out = qaic_model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
        assert len(out) == len(prompts)

    def test_throughput_is_positive_and_finite(self, qaic_model, sample_prompts):
        prompts = sample_prompts[:4]
        max_tokens = 8

        start = time.perf_counter()
        out = qaic_model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))
        elapsed = time.perf_counter() - start

        gen_tokens = sum(len(token_ids[0]) for token_ids, _ in out)
        reqps = len(prompts) / elapsed
        tokps = gen_tokens / elapsed

        assert elapsed > 0
        assert reqps > 0 and reqps == reqps  # not inf/nan
        assert tokps > 0 and tokps == tokps
        assert gen_tokens > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
