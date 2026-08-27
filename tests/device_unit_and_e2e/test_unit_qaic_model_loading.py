# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC model loading.

Confirms an LLM initialises correctly under various configurations and that
the resulting model is actually usable (tokenizer access, repeated
generate() calls, minimal-config construction, async_scheduling construction
in AOT mode).

Coverage areas
--------------
1. LLM initialises without error and exposes a working tokenizer
2. Multiple sequential generate() calls on the same LLM instance
3. LLM loads with a minimal config (no override_qaic_config)
4. LLM loads with async_scheduling=True (AOT only)
"""

import pytest

from vllm_qaic.platform_base import QaicPlatform


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestModelLoadingOnDevice:
    def test_llm_loads_successfully(self, qaic_model):
        assert qaic_model is not None

    def test_llm_has_tokenizer(self, qaic_model):
        tokenizer = qaic_model.llm.get_tokenizer()
        assert tokenizer is not None

    def test_llm_tokenizer_round_trip(self, qaic_model):
        tokenizer = qaic_model.llm.get_tokenizer()
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        recovered = tokenizer.decode(tokens, skip_special_tokens=True)
        assert text.strip() in recovered or recovered.strip() in text

    def test_second_generate_after_first(self, qaic_model, sampling_params):
        out1 = qaic_model.generate(["Hello"], sampling_params)
        out2 = qaic_model.generate(["World"], sampling_params)
        assert len(out1[0][0][0]) > 0
        assert len(out2[0][0][0]) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
)
def test_loads_with_minimal_config(device_group, vllm_runner, model_name, ctx_len, decode_bsz):
    """LLM must load with a minimal config (no dtype/kv_dtype/override_qaic_config)."""
    with vllm_runner(
        model_name,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        additional_config={"device_group": device_group},
    ) as model:
        assert model is not None


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_loads_with_async_scheduling(device_group, make_runner):
    """LLM must load with async_scheduling=True (AOT only)."""
    if not QaicPlatform.is_aot:
        pytest.skip("async_scheduling=True is AOT-only")
    with make_runner(True, device_group) as model:
        assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
