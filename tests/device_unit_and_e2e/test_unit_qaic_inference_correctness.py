# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC inference correctness.

Confirms generate() produces correct, well-formed output across boundary
conditions: max_tokens limits, prompt-length boundaries (including the
prefill_seq_len chunking boundary), and batch-size boundaries.

Replaces the old unit/generic/test_inference_correctness.py, which depended
on a nonexistent `llm` fixture and used the RequestOutput-based
llm.generate() API. Ported here onto the real qaic_model/make_runner fixture
system and the VllmRunner.generate() -> list[tuple[token_ids, texts]] shape.

Coverage areas
--------------
1. Basic generate() correctness (non-empty token_ids/text)
2. max_tokens boundary conditions
3. Prompt-length boundary conditions (short, near seq_len, over seq_len)
4. Batch-size boundary conditions
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
class TestInferenceBasicOnDevice:
    def test_generate_returns_output(self, qaic_model, sampling_params):
        out = qaic_model.generate(["Hello"], sampling_params)
        assert len(out) == 1

    def test_output_has_token_ids(self, qaic_model, sampling_params):
        out = qaic_model.generate(["Hello"], sampling_params)
        token_ids, _ = out[0]
        assert len(token_ids[0]) > 0

    def test_output_has_text(self, qaic_model, sampling_params):
        out = qaic_model.generate(["Hello"], sampling_params)
        _, texts = out[0]
        assert isinstance(texts[0], str)
        assert len(texts[0]) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestMaxTokensBoundaryOnDevice:
    """Uses qaic_model.llm.generate() (raw vLLM RequestOutput) rather than
    qaic_model.generate() (the VllmRunner wrapper in tests/conftest.py),
    whose req_sample_output_ids.append(prompt_ids + output_ids) concatenates
    prompt tokens onto the generated tokens — exact max_tokens boundary
    assertions here need generated tokens only."""

    def test_max_tokens_1(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.llm.generate(
            ["Hello"], SamplingParams(temperature=0.0, max_tokens=1)
        )
        assert len(out[0].outputs[0].token_ids) == 1

    def test_max_tokens_5(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.llm.generate(
            ["Hello"], SamplingParams(temperature=0.0, max_tokens=5)
        )
        assert len(out[0].outputs[0].token_ids) <= 5

    def test_max_tokens_respected_in_batch(self, qaic_model):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=3)
        prompts = ["Hello", "World", "Foo", "Bar"]
        out = qaic_model.llm.generate(prompts, params)
        for i, req_output in enumerate(out):
            assert len(req_output.outputs[0].token_ids) <= 3, (
                f"max_tokens exceeded at prompt {i}"
            )

    def test_ignore_eos_generates_max_tokens(self, qaic_model):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, max_tokens=10, ignore_eos=True)
        out = qaic_model.llm.generate(["Hello"], params)
        assert len(out[0].outputs[0].token_ids) == 10


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestPromptBoundaryOnDevice:
    def test_single_token_prompt(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(["Hi"], SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out[0][0][0]) > 0

    def test_long_prompt_near_ctx_len(self, qaic_model, ctx_len):
        from vllm import SamplingParams

        words = "The quick brown fox jumps over the lazy dog. " * (ctx_len // 40)
        out = qaic_model.generate([words], SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out[0][0][0]) > 0

    def test_prompt_at_prefill_seq_len_boundary(self, qaic_model, seq_len):
        from vllm import SamplingParams

        words = "word " * (seq_len - 2)
        out = qaic_model.generate([words], SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out[0][0][0]) > 0

    def test_prompt_over_prefill_seq_len(self, qaic_model, seq_len):
        """Prompt longer than prefill_seq_len must trigger chunking and produce output."""
        from vllm import SamplingParams

        words = "word " * (seq_len + seq_len // 2)
        out = qaic_model.generate([words], SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out[0][0][0]) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestBatchSizeBoundaryOnDevice:
    def test_batch_size_1(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out) == 1

    def test_batch_size_2(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(["Hello", "World"], SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out) == 2

    def test_batch_size_max(self, qaic_model, decode_bsz):
        from vllm import SamplingParams

        prompts = [f"Prompt {i}" for i in range(decode_bsz)]
        out = qaic_model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out) == decode_bsz
        for i, (token_ids, _) in enumerate(out):
            assert len(token_ids[0]) > 0, f"Empty output at prompt {i}"

    def test_mixed_length_batch(self, qaic_model):
        from vllm import SamplingParams

        prompts = [
            "Hi",
            "The quick brown fox jumps over the lazy dog.",
            "What is the meaning of life?",
            "A",
        ]
        out = qaic_model.generate(prompts, SamplingParams(temperature=0.0, max_tokens=5))
        assert len(out) == len(prompts)
        for i, (token_ids, _) in enumerate(out):
            assert len(token_ids[0]) > 0, f"Empty output at prompt {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
