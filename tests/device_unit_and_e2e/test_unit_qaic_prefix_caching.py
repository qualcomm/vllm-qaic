# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device correctness tests for QAIC prefix caching.

Pure-Python coverage of the AOT-mode disable logic (enable_prefix_caching
forced False, mamba_block_size/mamba_cache_mode reset, block_size formula)
and the disaggregated-serving constraints lives in
unit/prefixcaching/test_prefix_caching.py. Those confirm prefix caching is
correctly disabled — this file documents the intended ENABLED behaviour on
real hardware, once supported.

Prefix caching is not yet supported in vllm-qaic AOT mode — marked
xfail(strict=False) so this documents intended behaviour without failing CI
until support lands.

Coverage areas
--------------
1. Repeated shared-prefix prompts reuse cache and produce correct output
2. Prefix-cache hit does not change greedy output vs. no caching
"""

import pytest

pytestmark = pytest.mark.xfail(
    reason="Prefix caching is not yet supported in vllm-qaic AOT mode",
    strict=False,
)


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
)
class TestPrefixCachingOnDevice:
    def test_shared_prefix_prompts_produce_output(self, device_group, make_runner):
        from vllm import SamplingParams

        shared_prefix = "The quick brown fox jumps over the lazy dog. " * 4
        prompts = [shared_prefix + "Tell me a story.", shared_prefix + "What is 2+2?"]
        with make_runner(False, device_group, enable_prefix_caching=True) as model:
            out = model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=10))
        for i, (ids, _) in enumerate(out):
            assert len(ids[0]) > 0, f"Empty output at prompt {i}"

    def test_prefix_cache_hit_matches_no_cache(self, device_groups, make_runner):
        shared_prefix = "The quick brown fox jumps over the lazy dog. " * 4
        prompt = [shared_prefix + "Tell me a story."]
        from vllm import SamplingParams

        greedy = SamplingParams(temperature=0.0, max_tokens=10)

        with make_runner(False, device_groups[0], enable_prefix_caching=False) as m:
            out_no_cache = m.generate(prompt, greedy)

        with make_runner(False, device_groups[1], enable_prefix_caching=True) as m:
            m.generate(prompt, greedy)  # warm the cache
            out_cached = m.generate(prompt, greedy)

        assert out_no_cache[0][0] == out_cached[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
