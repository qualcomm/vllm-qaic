# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import random

import pytest
from vllm import SamplingParams

_DFLASH_BLOCK_SIZE = 16
_DFLASH_SEQ_LEN = 128

# Short max_tokens keeps these correctness checks fast; DFlash's own
# acceptance-rate/throughput behavior is covered by examples/benchmarks, not here.
_SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=16)


@pytest.mark.qaic_test_config(
    model_name="Qwen/Qwen3-4B",
    ctx_len=4096,
    seq_len=_DFLASH_SEQ_LEN,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=4,
    override_qaic_config={
        "num_cores": 8,
        "prefill_seq_len": _DFLASH_SEQ_LEN,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
    },
    draft_override_qaic_config={
        "num_cores": 8,
        "prefill_seq_len": _DFLASH_BLOCK_SIZE,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
    },
    speculative_config={
        "method": "dflash",
        "model": "z-lab/Qwen3-4B-DFlash-b16",
        "num_speculative_tokens": _DFLASH_BLOCK_SIZE,
    },
)
class TestQaicDFlash:
    """DFlash (block-diffusion draft model) speculative decoding checks."""

    def test_generate(self, qaic_model, sample_prompts):
        prompts = list(sample_prompts)[:4]
        output = qaic_model.generate(prompts, _SAMPLING_PARAMS)
        assert len(output) == len(prompts), (
            "Number of generated outputs does not match the number of prompts!!"
        )
        for _, texts in output:
            assert texts[0], "DFlash produced an empty generation"

    def test_prefill_chunking(self, qaic_model, sharegpt_prompts, seq_len):
        """Prompts longer than the TLM's prefill_seq_len (`seq_len`) force the
        DFlash runner's multi-chunk prefill path (non-final chunks flagged via
        `prefill_is_partial`, per-chunk TLM hidden-state capture for the DLM).

        `min_len` guarantees at least 2 chunks are actually exercised (prompt
        strictly longer than `seq_len`); `seed` makes the ShareGPT sample
        selection deterministic across runs."""
        prompts = sharegpt_prompts(
            2, in_len=seq_len * 3, min_len=seq_len + 1, seed=0
        )
        assert len(prompts) > 0, "no ShareGPT prompts long enough to force chunking"
        output = qaic_model.generate(prompts, _SAMPLING_PARAMS)
        assert len(output) == len(prompts), (
            "Number of generated outputs does not match the number of prompts!!"
        )
        for _, texts in output:
            assert texts[0], "DFlash produced an empty generation for a chunked prefill"

    def test_mixed_decode_prefill_batch(self, qaic_model, sample_prompts, decode_bsz):
        """More concurrent prompts than `decode_bsz` forces continuous batching to
        schedule new prefills alongside in-flight decodes within the same step."""
        prompts = list(sample_prompts)[: decode_bsz + 2]
        assert len(prompts) > decode_bsz, (
            "need more prompts than decode_bsz to exercise mixed decode+prefill"
        )
        output = qaic_model.generate(prompts, _SAMPLING_PARAMS)
        assert len(output) == len(prompts), (
            "Number of generated outputs does not match the number of prompts!!"
        )
        for _, texts in output:
            assert texts[0], "DFlash produced an empty generation"

    def test_output_consistency(self, qaic_model, sample_prompts):
        prompts = list(sample_prompts)[:4]
        output_dict: dict[str, list[str]] = {p: [] for p in prompts}

        for _ in range(2):
            random.shuffle(prompts)
            output = qaic_model.generate(prompts, _SAMPLING_PARAMS)
            assert len(output) == len(prompts), (
                "Number of generated outputs does not match the number of prompts!!"
            )
            for prompt, (_, texts) in zip(prompts, output, strict=False):
                output_dict[prompt].append(prompt + texts[0])

        for prompt, generations in output_dict.items():
            assert len(set(generations)) == 1, (
                "Outputs from different slots for same prompt does not match!!"
            )
