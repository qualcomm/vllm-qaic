# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC-specific multimodal helper functions.

These functions are defined in test_multimodal.py and used to determine
which QAIC-specific configuration to apply for different VLM architectures.
They are extracted here as pure unit tests that require no hardware.

Functions tested
----------------
- is_internvl(name)   — True for InternVL models
- is_llama4(name)     — True for Llama-4 Scout
- is_qwenvl(name)     — True for Qwen VL models
- is_gemma(name)      — True for Gemma models
- is_granite(name)    — True for Granite models
- update_qaic_config(base_cfg, **updates) — merges QAIC config with VLM-specific overrides

Coverage areas
--------------
1. Model name detection functions — positive and negative cases
2. update_qaic_config() — merges base config with updates
3. update_qaic_config() — Qwen VL models get height/width injected
4. update_qaic_config() — None base config treated as empty dict
"""

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Import the real QAIC-specific model-detection helpers from
# tests/device_unit_and_e2e/multimodal/conftest.py instead of re-implementing
# them here.
#
# These functions cannot be imported with a plain
# `from tests.device_unit_and_e2e.multimodal.conftest import ...` because
# pytest's default import mode does not add the vllm-qaic/ repo root (the
# parent of the top-level `tests` package) to sys.path when collecting a
# test file that lives inside a *different* subpackage
# (tests.unit.multimodal vs. tests.device_unit_and_e2e.multimodal) —
# empirically verified: the plain dotted import raises
# `ModuleNotFoundError: No module named 'tests'` when this file is
# collected by pytest (it only works via ad-hoc `python3 -c` from the
# vllm-qaic/ cwd, which is not representative of real test runs). Inserting
# the repo root explicitly makes the import work under real pytest
# execution regardless of invocation cwd.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.device_unit_and_e2e.multimodal.conftest import (  # noqa: E402
    QWEN_IMAGE_HEIGHT,
    QWEN_IMAGE_WIDTH,
    is_gemma,
    is_granite,
    is_internvl,
    is_llama4,
    is_qwenvl,
)
from tests.device_unit_and_e2e.multimodal.conftest import (  # noqa: E402
    update_qaic_config as _real_update_qaic_config,
)


def update_qaic_config(base_cfg, model_name: str = "", **updates) -> dict:
    """
    Merge base_cfg with updates, injecting Qwen VL image dimensions if needed.

    Thin wrapper around the real
    tests.device_unit_and_e2e.multimodal.conftest.update_qaic_config, which
    takes (model_name, base_cfg, **updates) — the opposite argument order
    from this module's historical (base_cfg, model_name, **updates) signature.
    Kept so existing call sites below (which pass base_cfg positionally and
    model_name as a keyword) do not need to change.
    """
    return _real_update_qaic_config(model_name, base_cfg, **updates)


# ===========================================================================
# 1. is_internvl()
# ===========================================================================

class TestIsInternVL:
    def test_internvl_model_detected(self):
        assert is_internvl("OpenGVLab/InternVL2_5-1B") is True

    def test_internvl_case_sensitive(self):
        """Detection is case-sensitive — 'internvl' (lowercase) must not match."""
        assert is_internvl("internvl2-1b") is False

    def test_llama_not_internvl(self):
        assert is_internvl("meta-llama/Llama-3-8B") is False

    def test_qwen_not_internvl(self):
        assert is_internvl("Qwen/Qwen2.5-VL-7B-Instruct") is False

    def test_empty_string_not_internvl(self):
        assert is_internvl("") is False

    def test_internvl_in_path(self):
        assert is_internvl("OpenGVLab/InternVL2-8B") is True


# ===========================================================================
# 2. is_llama4()
# ===========================================================================

class TestIsLlama4:
    def test_llama4_scout_detected(self):
        assert is_llama4("meta-llama/Llama-4-Scout-17B-16E-Instruct") is True

    def test_llama3_not_llama4(self):
        assert is_llama4("meta-llama/Llama-3-8B") is False

    def test_partial_match_not_llama4(self):
        """Partial match must not trigger — exact string required."""
        assert is_llama4("meta-llama/Llama-4-Scout-17B") is False

    def test_empty_string_not_llama4(self):
        assert is_llama4("") is False


# ===========================================================================
# 3. is_qwenvl()
# ===========================================================================

class TestIsQwenVL:
    def test_qwen_vl_detected(self):
        assert is_qwenvl("Qwen/Qwen2.5-VL-7B-Instruct") is True

    def test_qwen3_vl_detected(self):
        assert is_qwenvl("Qwen/Qwen3-VL-7B") is True

    def test_qwen_text_only_detected(self):
        """Any Qwen model (including text-only) matches — detection is by name prefix."""
        assert is_qwenvl("Qwen/Qwen2.5-7B-Instruct") is True

    def test_llama_not_qwenvl(self):
        assert is_qwenvl("meta-llama/Llama-3-8B") is False

    def test_case_sensitive_lowercase_qwen(self):
        """'qwen' (lowercase) must not match."""
        assert is_qwenvl("qwen/qwen2-7b") is False

    def test_empty_string_not_qwenvl(self):
        assert is_qwenvl("") is False


# ===========================================================================
# 4. is_gemma()
# ===========================================================================

class TestIsGemma:
    def test_gemma_detected(self):
        assert is_gemma("google/gemma-4-E2B-it") is True

    def test_gemma2_detected(self):
        assert is_gemma("google/gemma-2-9b-it") is True

    def test_llama_not_gemma(self):
        assert is_gemma("meta-llama/Llama-3-8B") is False

    def test_empty_string_not_gemma(self):
        assert is_gemma("") is False

    def test_gemma_case_sensitive(self):
        """'Gemma' (capital G) must not match."""
        assert is_gemma("Google/Gemma-2-9B") is False


# ===========================================================================
# 5. is_granite()
# ===========================================================================

class TestIsGranite:
    def test_granite_detected(self):
        assert is_granite("ibm-granite/granite-3.3-8b-instruct") is True

    def test_granite_embedding_detected(self):
        assert is_granite("ibm-granite/granite-embedding-30m-english") is True

    def test_llama_not_granite(self):
        assert is_granite("meta-llama/Llama-3-8B") is False

    def test_empty_string_not_granite(self):
        assert is_granite("") is False

    def test_granite_case_sensitive(self):
        """'Granite' (capital G) must not match."""
        assert is_granite("IBM-Granite/Granite-3.3-8B") is False


# ===========================================================================
# 6. update_qaic_config()
# ===========================================================================

class TestUpdateQaicConfig:
    def test_none_base_cfg_treated_as_empty(self):
        """None base_cfg must be treated as an empty dict."""
        result = update_qaic_config(None, model_name="meta-llama/Llama-3-8B")
        assert isinstance(result, dict)

    def test_base_cfg_preserved(self):
        """Keys in base_cfg must be preserved."""
        result = update_qaic_config({"num_cores": 8}, model_name="meta-llama/Llama-3-8B")
        assert result["num_cores"] == 8

    def test_updates_applied(self):
        """Keyword updates must be applied to the config."""
        result = update_qaic_config({}, model_name="meta-llama/Llama-3-8B", mos=2)
        assert result["mos"] == 2

    def test_none_updates_ignored(self):
        """Updates with None value must be ignored."""
        result = update_qaic_config({}, model_name="meta-llama/Llama-3-8B", mos=None)
        assert "mos" not in result

    def test_qwenvl_gets_height_width(self):
        """Qwen VL models must get height and width injected."""
        result = update_qaic_config({}, model_name="Qwen/Qwen2.5-VL-7B-Instruct")
        assert result["height"] == QWEN_IMAGE_HEIGHT
        assert result["width"] == QWEN_IMAGE_WIDTH

    def test_non_qwenvl_no_height_width(self):
        """Non-Qwen models must NOT get height/width injected."""
        result = update_qaic_config({}, model_name="meta-llama/Llama-3-8B")
        assert "height" not in result
        assert "width" not in result

    def test_base_cfg_not_mutated(self):
        """The original base_cfg dict must not be mutated."""
        base = {"num_cores": 8}
        update_qaic_config(base, model_name="meta-llama/Llama-3-8B", mos=2)
        assert "mos" not in base

    def test_updates_override_base(self):
        """Updates must override base_cfg values."""
        result = update_qaic_config({"num_cores": 8}, model_name="", num_cores=16)
        assert result["num_cores"] == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
