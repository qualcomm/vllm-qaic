# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC embedding model support (pure Python, no hardware).

On-device embedding tests (output shape, unit norm, determinism, similarity)
have moved to
vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_embedding.py,
which uses the device/ fixture system (qaic_model, make_runner) instead of
the old embed_llm fixture.

Usage:
  pytest test_embedding_utils.py -v

Coverage areas
--------------
1. _check_vector() — normalize, softmax, raw, edge cases
2. EMBEDDING_MODELS / CROSS_ENCODER_MODELS lists
3. SEQWISE_TASKS / TOKWISE_TASKS classification
4. override_qaic_config pooling keys (pooling_device, pooling_method,
   normalize, softmax, task, embed_seq_len) via _clean_config()
5. prefix_caching_enabled logic
"""

import math
import os
import types

import pytest

from vllm_qaic.utils.qaic_utils import _clean_config


# ---------------------------------------------------------------------------
# Re-implement the helper functions from test_qaic_embed_offline_online.py
#
# On-device embedding coverage now lives in
# vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_embedding.py (uses the
# device/ fixture system directly, not these re-implemented helpers).
# ---------------------------------------------------------------------------

def _check_vector(vec, normalize: bool, softmax: bool, label: str = ""):
    """
    Core check for a single embedding vector based on normalize/softmax flags.
    Mirrors the logic in test_qaic_embed_offline_online.py::_check_vector().
    """
    norm = math.sqrt(sum(x**2 for x in vec))
    total = sum(vec)

    if normalize and not softmax:
        assert abs(norm - 1.0) < 1e-3, (
            f"{label}: expected unit norm (normalize=True, softmax=False), got {norm:.4f}"
        )
    elif softmax:
        assert abs(total - 1.0) < 1e-3, (
            f"{label}: expected sum~1.0 (softmax=True), got {total:.4f}"
        )
        assert all(v >= 0 for v in vec), (
            f"{label}: expected non-negative values (softmax=True)"
        )
    else:
        assert norm > 0, f"{label}: zero vector (all values are zero)"

    return norm, total


# QAIC-supported embedding model lists (from test_qaic_embed_offline_online.py)
EMBEDDING_MODELS = [
    "ibm-granite/granite-embedding-30m-english",
    "ibm-granite/granite-embedding-125m-english",
    "ibm-granite/granite-embedding-107m-multilingual",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "nomic-ai/nomic-embed-text-v1.5",
    "NovaSearch/stella_en_1.5B_v5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-large-v2",
    "intfloat/e5-mistral-7b-instruct",
    "sentence-transformers/multi-qa-mpnet-base-cos-v1",
    "jinaai/jina-embeddings-v2-base-code",
]

CROSS_ENCODER_MODELS = [
    "BAAI/bge-reranker-v2-m3",
]

SEQWISE_TASKS = ["embed", "encode_embed", "classify", "encode_classify", "score"]
TOKWISE_TASKS = ["reward", "token_embed", "token_classify"]


def _compute_prefix_caching_enabled():
    """Mirrors the prefix_caching_enabled logic from test_qaic_embed_offline_online.py."""
    return not (
        os.environ.get("VLLM_USE_V1") != "0"
        and os.environ.get("DISABLE_PREFIX_CACHING") == "1"
    )


# ===========================================================================
# 1. _check_vector() — normalize=True, softmax=False (unit norm)
# ===========================================================================

class TestCheckVectorNormalized:
    """normalize=True, softmax=False → unit L2 norm."""

    def test_unit_vector_passes(self):
        vec = [1.0, 0.0, 0.0]
        norm, _ = _check_vector(vec, normalize=True, softmax=False)
        assert abs(norm - 1.0) < 1e-3

    def test_normalized_vector_passes(self):
        raw = [3.0, 4.0]
        magnitude = math.sqrt(3**2 + 4**2)
        vec = [x / magnitude for x in raw]
        norm, _ = _check_vector(vec, normalize=True, softmax=False)
        assert abs(norm - 1.0) < 1e-3

    def test_non_unit_vector_fails(self):
        vec = [1.0, 1.0, 1.0]
        with pytest.raises(AssertionError, match="unit norm"):
            _check_vector(vec, normalize=True, softmax=False)

    def test_high_dim_normalized_vector(self):
        dim = 768
        vec = [1.0 / math.sqrt(dim)] * dim
        norm, _ = _check_vector(vec, normalize=True, softmax=False)
        assert abs(norm - 1.0) < 1e-3

    def test_negative_values_allowed_when_normalized(self):
        vec = [1.0 / math.sqrt(2), -1.0 / math.sqrt(2)]
        norm, _ = _check_vector(vec, normalize=True, softmax=False)
        assert abs(norm - 1.0) < 1e-3


# ===========================================================================
# 2. _check_vector() — softmax=True (probabilities)
# ===========================================================================

class TestCheckVectorSoftmax:
    """softmax=True → all positive, sum ≈ 1.0."""

    def test_valid_probability_distribution(self):
        vec = [0.1, 0.3, 0.6]
        _, total = _check_vector(vec, normalize=False, softmax=True)
        assert abs(total - 1.0) < 1e-3

    def test_uniform_distribution(self):
        n = 10
        vec = [1.0 / n] * n
        _, total = _check_vector(vec, normalize=False, softmax=True)
        assert abs(total - 1.0) < 1e-3

    def test_negative_values_fail_softmax(self):
        vec = [0.6, 0.5, -0.1]
        with pytest.raises(AssertionError, match="non-negative"):
            _check_vector(vec, normalize=False, softmax=True)

    def test_sum_not_one_fails_softmax(self):
        vec = [0.3, 0.3, 0.3]
        with pytest.raises(AssertionError, match="sum~1.0"):
            _check_vector(vec, normalize=False, softmax=True)

    def test_softmax_overrides_normalize(self):
        vec = [0.25, 0.25, 0.25, 0.25]
        _, total = _check_vector(vec, normalize=True, softmax=True)
        assert abs(total - 1.0) < 1e-3


# ===========================================================================
# 3. _check_vector() — raw values (no normalize, no softmax)
# ===========================================================================

class TestCheckVectorRaw:
    """normalize=False, softmax=False → raw values, non-zero."""

    def test_raw_positive_vector_passes(self):
        vec = [0.5, 1.2, -0.3, 0.8]
        norm, _ = _check_vector(vec, normalize=False, softmax=False)
        assert norm > 0

    def test_raw_negative_vector_passes(self):
        vec = [-1.0, -2.0, -3.0]
        norm, _ = _check_vector(vec, normalize=False, softmax=False)
        assert norm > 0

    def test_zero_vector_fails(self):
        vec = [0.0, 0.0, 0.0]
        with pytest.raises(AssertionError, match="zero vector"):
            _check_vector(vec, normalize=False, softmax=False)

    def test_single_nonzero_element_passes(self):
        vec = [0.0, 0.0, 1.0]
        norm, _ = _check_vector(vec, normalize=False, softmax=False)
        assert norm > 0

    def test_label_appears_in_error_message(self):
        vec = [1.0, 1.0, 1.0]
        with pytest.raises(AssertionError, match="MY_LABEL"):
            _check_vector(vec, normalize=True, softmax=False, label="MY_LABEL")


# ===========================================================================
# 4. EMBEDDING_MODELS / CROSS_ENCODER_MODELS lists
# ===========================================================================

class TestEmbeddingModelLists:
    def test_embedding_models_non_empty(self):
        assert len(EMBEDDING_MODELS) > 0

    def test_cross_encoder_models_non_empty(self):
        assert len(CROSS_ENCODER_MODELS) > 0

    def test_bge_base_in_embedding_models(self):
        assert "BAAI/bge-base-en-v1.5" in EMBEDDING_MODELS

    def test_bge_reranker_in_cross_encoder_models(self):
        assert "BAAI/bge-reranker-v2-m3" in CROSS_ENCODER_MODELS

    def test_granite_embedding_in_models(self):
        granite_models = [m for m in EMBEDDING_MODELS if "granite" in m]
        assert len(granite_models) > 0

    def test_no_overlap_between_lists(self):
        overlap = set(EMBEDDING_MODELS) & set(CROSS_ENCODER_MODELS)
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_models_have_org_prefix(self):
        for model in EMBEDDING_MODELS + CROSS_ENCODER_MODELS:
            assert "/" in model, f"Model {model!r} missing org prefix"


# ===========================================================================
# 5. SEQWISE_TASKS / TOKWISE_TASKS classification
# ===========================================================================

class TestTaskClassification:
    def test_seqwise_tasks_non_empty(self):
        assert len(SEQWISE_TASKS) > 0

    def test_tokwise_tasks_non_empty(self):
        assert len(TOKWISE_TASKS) > 0

    def test_embed_is_seqwise(self):
        assert "embed" in SEQWISE_TASKS

    def test_classify_is_seqwise(self):
        assert "classify" in SEQWISE_TASKS

    def test_score_is_seqwise(self):
        assert "score" in SEQWISE_TASKS

    def test_reward_is_tokwise(self):
        assert "reward" in TOKWISE_TASKS

    def test_token_embed_is_tokwise(self):
        assert "token_embed" in TOKWISE_TASKS

    def test_no_overlap_between_task_lists(self):
        overlap = set(SEQWISE_TASKS) & set(TOKWISE_TASKS)
        assert len(overlap) == 0, f"Task overlap: {overlap}"

    def test_all_tasks_are_strings(self):
        for task in SEQWISE_TASKS + TOKWISE_TASKS:
            assert isinstance(task, str)


# ===========================================================================
# 6. override_qaic_config pooling keys, via the real _clean_config()
#
# QAIC pooling models are configured through override_qaic_config, not
# vllm.config.PoolerConfig directly (see vllm/model_executor/model_loader/qaic.py
# ::load_qaic_model, which asserts pooling_device/pooling_method are present in
# override_qaic_config for runner_type == "pooling"). These tests exercise the
# real _clean_config() normalization for those keys — the QAIC-specific surface
# a pooling deployment actually configures.
# ===========================================================================

def _make_vllm_config(max_model_len: int = 2048):
    model_config = types.SimpleNamespace(max_model_len=max_model_len)
    return types.SimpleNamespace(model_config=model_config)


class TestOverrideQaicConfigPooling:
    def test_pooling_device_qaic_lowercased(self):
        result = _clean_config({"pooling_device": "QAIC"})
        assert result["pooling_device"] == "qaic"

    def test_pooling_method_passthrough(self):
        result = _clean_config({"pooling_method": "mean"})
        assert result["pooling_method"] == "mean"

    def test_normalize_true_string_becomes_bool(self):
        result = _clean_config({"normalize": "true"})
        assert result["normalize"] is True

    def test_normalize_false_string_becomes_bool(self):
        result = _clean_config({"normalize": "false"})
        assert result["normalize"] is False

    def test_softmax_true_string_becomes_bool(self):
        result = _clean_config({"softmax": "true"})
        assert result["softmax"] is True

    def test_task_embed_passthrough(self):
        result = _clean_config({"task": "embed"})
        assert result["task"] == "embed"

    def test_task_classify_passthrough(self):
        result = _clean_config({"task": "classify"})
        assert result["task"] == "classify"

    def test_full_pooling_override_config(self):
        """All pooling-related override_qaic_config keys together, as a
        pooling deployment would pass them."""
        result = _clean_config({
            "pooling_device": "qaic",
            "pooling_method": "mean",
            "normalize": "true",
            "softmax": "false",
            "task": "embed",
        })
        assert result == {
            "pooling_device": "qaic",
            "pooling_method": "mean",
            "normalize": True,
            "softmax": False,
            "task": "embed",
        }

    def test_embed_seq_len_resolves_to_prefill_seq_len(self):
        """embed_seq_len (the CLI-facing key) must include max_model_len and
        resolve to prefill_seq_len — the key load_qaic_model's compile config
        actually reads."""
        cfg = _make_vllm_config(max_model_len=512)
        result = _clean_config({"embed_seq_len": "128,256,512"}, cfg)
        assert result["prefill_seq_len"] == [128, 256, 512]

    def test_embed_seq_len_missing_max_model_len_raises(self):
        cfg = _make_vllm_config(max_model_len=1024)
        with pytest.raises(AssertionError):
            _clean_config({"embed_seq_len": "128,256,512"}, cfg)


# ===========================================================================
# 7. prefix_caching_enabled logic
# ===========================================================================

class TestPrefixCachingEnabled:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("VLLM_USE_V1", raising=False)
        monkeypatch.delenv("DISABLE_PREFIX_CACHING", raising=False)
        assert _compute_prefix_caching_enabled() is True

    def test_disabled_when_both_flags_set(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_V1", "1")
        monkeypatch.setenv("DISABLE_PREFIX_CACHING", "1")
        assert _compute_prefix_caching_enabled() is False

    def test_enabled_when_vllm_use_v1_is_zero(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_V1", "0")
        monkeypatch.setenv("DISABLE_PREFIX_CACHING", "1")
        assert _compute_prefix_caching_enabled() is True

    def test_enabled_when_disable_prefix_caching_not_one(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_V1", "1")
        monkeypatch.setenv("DISABLE_PREFIX_CACHING", "0")
        assert _compute_prefix_caching_enabled() is True

    def test_enabled_when_disable_prefix_caching_unset(self, monkeypatch):
        monkeypatch.setenv("VLLM_USE_V1", "1")
        monkeypatch.delenv("DISABLE_PREFIX_CACHING", raising=False)
        assert _compute_prefix_caching_enabled() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
