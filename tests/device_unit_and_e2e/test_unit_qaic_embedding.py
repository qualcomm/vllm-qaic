# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC embedding model support.

Pure-Python coverage of the _check_vector() helper, model/task lists, and
override_qaic_config pooling-key normalisation (via the real _clean_config())
lives in unit/embedding/test_embedding_utils.py. These tests confirm the
real on-device pipeline: load a pooling model (override_qaic_config carries
pooling_device/pooling_method/normalize per
vllm/model_executor/model_loader/qaic.py::load_qaic_model) and embed real
prompts.

Coverage areas
--------------
1. embed() returns one output per prompt, non-empty vector
2. Unit-norm output when normalize=True (BGE-style pooling config)
3. Same prompt embeds identically across two calls (determinism)
4. Self-similarity (cosine(v, v)) is ~1.0

NOTE: uses make_runner (not the qaic_model fixture) so that
pooler_config=PoolerConfig(task="embed") can be passed through to LLM(), per
the project convention in examples/qaic_embed.py. Without it, vLLM's own
pooling-task inference (vllm.config.model.ModelConfig.get_pooling_task) picks
"embed&token_classify" as this model's default task ahead of "embed" in its
priority list, and .embed()'s hardcoded pooling_task="embed" then mismatches,
raising ValueError. override_qaic_config["task"] only affects QAIC/QEfficient
compile-time pooling behavior, not vLLM's own pooler_config.

NOTE: _POOLING_OVERRIDE is also passed explicitly as an override_qaic_config
kwarg to make_runner(), not just via the qaic_test_config marker. Unlike the
qaic_model fixture, make_runner's override_qaic_config parameter defaults to
the session-scoped override_qaic_config fixture (conftest.py), which only
reads the --override-qaic-config CLI option, not this test's marker — so the
marker's override_qaic_config would otherwise be silently ignored.
"""

import math

import pytest
from vllm.config import PoolerConfig

_POOLING_OVERRIDE = {
    "pooling_device": "qaic",
    "pooling_method": "mean",
    "normalize": True,
    "softmax": False,
    "task": "embed",
}


@pytest.mark.qaic_test_config(
    model_name="BAAI/bge-base-en-v1.5",
    ctx_len=512,
    decode_bsz=4,
    override_qaic_config=_POOLING_OVERRIDE,
)
class TestEmbeddingOnDevice:
    @pytest.fixture
    def qaic_embed_model(self, device_group, make_runner):
        with make_runner(
            False,
            device_group,
            override_qaic_config=_POOLING_OVERRIDE,
            pooler_config=PoolerConfig(task="embed"),
        ) as model:
            yield model

    def test_embed_returns_output_per_prompt(self, qaic_embed_model):
        prompts = ["Hello world", "The capital of France is Paris"]
        out = qaic_embed_model.embed(prompts)
        assert len(out) == len(prompts)

    def test_embedding_is_non_empty_vector(self, qaic_embed_model):
        out = qaic_embed_model.embed(["Hello world"])
        assert len(out[0]) > 0

    def test_unit_norm_when_normalized(self, qaic_embed_model):
        """override_qaic_config normalize=True must yield unit L2 norm."""
        out = qaic_embed_model.embed(["Hello world"])
        norm = math.sqrt(sum(x**2 for x in out[0]))
        assert abs(norm - 1.0) < 1e-2

    def test_embedding_deterministic(self, qaic_embed_model):
        prompt = ["Hello world"]
        out1 = qaic_embed_model.embed(prompt)
        out2 = qaic_embed_model.embed(prompt)
        assert out1[0] == out2[0]

    def test_self_similarity_near_one(self, qaic_embed_model):
        """cos(v, v) must be ~1.0 regardless of normalization."""
        out = qaic_embed_model.embed(["The quick brown fox"] * 2)
        emb1, emb2 = out[0], out[1]
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = math.sqrt(sum(x**2 for x in emb1))
        norm2 = math.sqrt(sum(x**2 for x in emb2))
        cosine = dot / (norm1 * norm2 + 1e-8)
        assert cosine > 0.99, f"Self-similarity should be ~1.0, got {cosine:.4f}"

    def test_different_prompts_produce_different_embeddings(self, qaic_embed_model):
        out = qaic_embed_model.embed(["Hello world", "Goodbye world"])
        assert out[0] != out[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
