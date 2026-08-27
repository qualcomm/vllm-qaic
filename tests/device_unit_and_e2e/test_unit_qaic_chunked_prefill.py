# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC chunked prefill.

Pure-Python coverage of the prefill_seq_len / max_num_batched_tokens formula
(driven directly against QaicPlatform.check_and_update_config()) lives in
unit/generic/test_chunked_prefill.py. These tests confirm the formula's
real-world effect: a real runner actually chunks a long prompt, produces
correct output for mixed short/long batches, and that chunked vs. non-chunked
prefill produce identical greedy output.

Coverage areas
--------------
1. Long prompt (> prefill_seq_len) produces valid output under chunking
2. Mixed short/long batch all produce valid output
3. Chunked prefill output matches non-chunked prefill output (same greedy seq)
"""

import pytest


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=64,
    ctx_len=512,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_long_prompt_produces_output(device_group, make_runner, seq_len):
    """A prompt longer than prefill_seq_len must trigger chunking and still
    produce valid output."""
    from vllm import SamplingParams

    words = "The quick brown fox jumps over the lazy dog. " * (seq_len // 4)
    with make_runner(
        False, device_group, enable_chunked_prefill=True, max_num_batched_tokens=seq_len,
    ) as model:
        out = model.generate([words], SamplingParams(temperature=0.0, max_tokens=5))
    assert len(out[0][0][0]) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=64,
    ctx_len=512,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_mixed_short_long_batch(device_group, make_runner, seq_len):
    """A batch mixing short and long (chunked) prompts must all produce
    valid output."""
    from vllm import SamplingParams

    long_prompt = "word " * (seq_len + seq_len // 2)
    prompts = ["Hi", long_prompt, "What is the meaning of life?", "A"]
    with make_runner(
        False, device_group, enable_chunked_prefill=True, max_num_batched_tokens=seq_len,
    ) as model:
        out = model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=5))
    assert len(out) == len(prompts)
    for i, (ids, _) in enumerate(out):
        assert len(ids[0]) > 0, f"Empty output at prompt {i}"


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=64,
    ctx_len=512,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
)
class TestChunkedPrefillMatchesBaseline:
    """Isolated in its own class (own qaic_test_config marker) so
    ci_scripts/collect_jobs.py's job sizing - which resolves num_devices from
    a single representative item per _item_scope() (nodeid.rsplit("::", 1)[0])
    - sees num_device_groups=2 for this test independently of the other
    bare functions above (which only need 1 device group). Sharing a module-
    level scope with them would size this job for 1 device and raise
    DevicePoolError at runtime."""

    def test_chunked_matches_non_chunked(self, device_groups, make_runner, seq_len):
        """Chunked prefill must produce the same greedy output as non-chunked
        prefill for the same long prompt."""
        from vllm import SamplingParams

        prompt = ["word " * (seq_len + seq_len // 2)]
        greedy = SamplingParams(temperature=0.0, max_tokens=10)

        with make_runner(False, device_groups[0]) as plain_model:
            out_plain = plain_model.generate(prompt, greedy)

        with make_runner(
            False, device_groups[1], enable_chunked_prefill=True, max_num_batched_tokens=seq_len,
        ) as chunked_model:
            out_chunked = chunked_model.generate(prompt, greedy)

        assert out_plain[0][0] == out_chunked[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
