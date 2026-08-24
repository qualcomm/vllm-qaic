# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.platform_base.QaicPlatform.

Tests the pure-Python methods of QaicPlatform that do not require QAIC
hardware, model loading, or network access.

Coverage areas
--------------
1. Basic platform properties (device name, AOT mode, dtype support)
2. additional_config parsing (JSON string, Python literal, dict, invalid)
3. _apply_dynamic_resolution_config() — Qwen2.5VL/Qwen3VL pixel bounds
4. DYNAMIC_RESOLUTION_MODELS list
5. Disaggregated serving constraints (LoRA, SpD types, prefix caching, stages)

Removed (no real coverage or duplicates):
  - TestEagerModeConstraints: trivial tautologies (is_aot is True or False)
  - TestWhisperModelHandling: tested SimpleNamespace attrs, not platform code
  - TestIntelOpenMPTuning: manually simulated the KMP logic instead of calling
    check_and_update_config() and verifying side effects
  - TestPrefillSeqLenList: simulated the budget formula; covered by
    test_chunked_prefill.py::TestPrefillSeqLen
  - TestOnDeviceSamplingConfig: duplicate of
    samplers/test_samplers.py::TestAicIncludeSamplerParsing
"""

import json
import types

import pytest
import torch

import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
from vllm_qaic.platform_base import QaicPlatform, DYNAMIC_RESOLUTION_MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_config(
    max_model_len: int = 2048,
    model_type: str = "llama",
    is_multimodal: bool = False,
    runner_type: str = "generate",
    mm_processor_kwargs: dict | None = None,
):
    hf_config = types.SimpleNamespace(model_type=model_type)
    return types.SimpleNamespace(
        max_model_len=max_model_len,
        hf_config=hf_config,
        is_multimodal_model=is_multimodal,
        runner_type=runner_type,
        mm_processor_kwargs=mm_processor_kwargs,
        enforce_eager=False,
    )


# NOTE: the shared `make_vllm_config` fixture (conftest.py) covers the full
# vllm_config mock used by check_and_update_config() tests below. This
# module additionally needs bare `model_config` objects (no wrapping
# vllm_config) for _apply_dynamic_resolution_config(), hence _make_model_config
# stays local.


# ===========================================================================
# 1. Basic platform properties
# ===========================================================================

class TestPlatformProperties:
    def test_device_name_is_qaic(self):
        assert QaicPlatform.get_device_name() == "qaic"

    def test_is_async_output_not_supported(self):
        assert QaicPlatform.is_async_output_supported(enforce_eager=None) is False
        assert QaicPlatform.is_async_output_supported(enforce_eager=True) is False
        assert QaicPlatform.is_async_output_supported(enforce_eager=False) is False

    def test_is_pin_memory_not_available(self):
        assert QaicPlatform.is_pin_memory_available() is False

    def test_check_if_supports_float16(self):
        assert QaicPlatform.check_if_supports_dtype(torch.float16) is True

    def test_check_if_supports_float32(self):
        assert QaicPlatform.check_if_supports_dtype(torch.float32) is True

    def test_check_if_not_supports_bfloat16(self):
        """QAIC does not support bfloat16 in AOT mode."""
        assert QaicPlatform.check_if_supports_dtype(torch.bfloat16) is False

    def test_check_if_not_supports_int8(self):
        assert QaicPlatform.check_if_supports_dtype(torch.int8) is False

    def test_device_control_env_var(self):
        assert QaicPlatform.device_control_env_var == "QAIC_VISIBLE_DEVICES"

    def test_dist_backend_is_qccl(self):
        assert QaicPlatform.dist_backend == "qccl"


# ===========================================================================
# 2. additional_config parsing
# ===========================================================================

class TestAdditionalConfigParsing:
    """
    check_and_update_config() must accept additional_config as:
    - a dict (pass-through)
    - a JSON string (parsed to dict)
    - a Python literal string (parsed to dict)
    - None (treated as empty dict)
    And must raise TypeError/ValueError for invalid inputs.
    """

    def test_dict_input_accepted(self, make_vllm_config):
        """Dict input must be accepted without modification."""
        cfg = make_vllm_config(additional_config={"override_qaic_config": {}})
        try:
            QaicPlatform.check_and_update_config(cfg)
        except Exception as e:
            if "additional_config" in str(e) or "parse" in str(e).lower():
                pytest.fail(f"Dict input raised unexpected error: {e}")

    def test_json_string_parsed_to_dict(self, make_vllm_config):
        """A valid JSON string must be parsed to a dict."""
        json_str = json.dumps({"override_qaic_config": {"num_cores": "8"}})
        cfg = make_vllm_config(additional_config=json_str)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except Exception as e:
            if "additional_config" in str(e) or "parse" in str(e).lower():
                pytest.fail(f"JSON string raised unexpected error: {e}")
        assert isinstance(cfg.additional_config, dict)

    def test_python_literal_string_parsed_to_dict(self, make_vllm_config):
        """A Python literal dict string must be parsed to a dict."""
        literal_str = "{'override_qaic_config': {'num_cores': '8'}}"
        cfg = make_vllm_config(additional_config=literal_str)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except Exception as e:
            if "additional_config" in str(e) or "parse" in str(e).lower():
                pytest.fail(f"Python literal string raised unexpected error: {e}")
        assert isinstance(cfg.additional_config, dict)

    def test_invalid_string_raises_value_error(self, make_vllm_config):
        """An invalid string (not JSON, not Python literal) must raise ValueError."""
        cfg = make_vllm_config(additional_config="this is not valid json or python")
        with pytest.raises(ValueError):
            QaicPlatform.check_and_update_config(cfg)

    def test_non_dict_json_raises_type_error(self, make_vllm_config):
        """A JSON string that parses to a non-dict (e.g., list) must raise TypeError."""
        cfg = make_vllm_config(additional_config='["not", "a", "dict"]')
        with pytest.raises(TypeError):
            QaicPlatform.check_and_update_config(cfg)


# ===========================================================================
# 3. _apply_dynamic_resolution_config()
# ===========================================================================

class TestApplyDynamicResolutionConfig:
    """
    Tests for the Qwen2.5VL/Qwen3VL dynamic resolution configuration.
    This sets min_pixels/max_pixels defaults and validates height/width lists.
    """

    def test_default_min_pixels_set(self):
        """min_pixels must default to 4 * 28 * 28 = 3136."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {}
        QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)
        assert model_config.mm_processor_kwargs["min_pixels"] == 4 * 28 * 28

    def test_default_max_pixels_set(self):
        """max_pixels must default to 16384 * 28 * 28 = 12845056."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {}
        QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)
        assert model_config.mm_processor_kwargs["max_pixels"] == 16384 * 28 * 28

    def test_existing_min_pixels_not_overridden(self):
        """If min_pixels is already set, it must not be overridden."""
        model_config = _make_model_config(mm_processor_kwargs={"min_pixels": 100})
        override_cfg = {}
        QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)
        assert model_config.mm_processor_kwargs["min_pixels"] == 100

    def test_height_width_list_accepted(self):
        """Matching height/width lists must be accepted."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {"height": [140, 280], "width": [140, 280]}
        QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)

    def test_mismatched_height_width_raises(self):
        """Mismatched height/width list lengths must raise AssertionError."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {"height": [140, 280], "width": [140]}
        with pytest.raises(AssertionError):
            QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)

    def test_height_without_width_raises(self):
        """Providing height without width must raise AssertionError."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {"height": [140]}
        with pytest.raises(AssertionError):
            QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)

    def test_int_height_width_converted_to_list(self):
        """Single int height/width must be converted to a list."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {"height": 140, "width": 140}
        QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)
        assert isinstance(override_cfg["height"], list)
        assert isinstance(override_cfg["width"], list)

    def test_mm_processor_kwargs_propagated_to_override(self):
        """min_pixels/max_pixels must be propagated to override_qaic_config."""
        model_config = _make_model_config(mm_processor_kwargs=None)
        override_cfg = {}
        QaicPlatform._apply_dynamic_resolution_config(model_config, override_cfg)
        assert "mm_processor_kwargs" in override_cfg
        assert "min_pixels" in override_cfg["mm_processor_kwargs"]
        assert "max_pixels" in override_cfg["mm_processor_kwargs"]


# ===========================================================================
# 4. DYNAMIC_RESOLUTION_MODELS list
# ===========================================================================

class TestDynamicResolutionModels:
    def test_qwen2_5_vl_in_list(self):
        assert "qwen2_5_vl" in DYNAMIC_RESOLUTION_MODELS

    def test_qwen3_vl_in_list(self):
        assert "qwen3_vl" in DYNAMIC_RESOLUTION_MODELS

    def test_list_is_non_empty(self):
        assert len(DYNAMIC_RESOLUTION_MODELS) > 0

    def test_llama_not_in_list(self):
        """Standard LLM models must not be in the dynamic resolution list."""
        assert "llama" not in DYNAMIC_RESOLUTION_MODELS


# ===========================================================================
# 5. Disaggregated serving constraints
# ===========================================================================

class TestDisaggregatedServingConstraints:
    """
    check_and_update_config() must enforce constraints when
    kv_transfer_config is set (disaggregated serving mode).
    """

    def test_lora_with_disagg_raises(self, make_vllm_config, make_kv_transfer_config):
        """LoRA + disaggregated serving must raise AssertionError."""
        lora_config = types.SimpleNamespace()
        kv_cfg = make_kv_transfer_config()
        cfg = make_vllm_config(
            lora_config=lora_config,
            kv_transfer_config=kv_cfg,
        )
        with pytest.raises(AssertionError, match="LORA with Disaggregated"):
            QaicPlatform.check_and_update_config(cfg)

    def test_unsupported_spd_type_with_disagg_raises(self, make_vllm_config, make_kv_transfer_config):
        """SpD types other than ngram/draft_model must raise AssertionError."""
        spec_config = types.SimpleNamespace(method="turbo", num_speculative_tokens=3)
        kv_cfg = make_kv_transfer_config()
        cfg = make_vllm_config(
            speculative_config=spec_config,
            kv_transfer_config=kv_cfg,
        )
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_ngram_spd_with_disagg_allowed(self, make_vllm_config, make_kv_transfer_config):
        """ngram SpD must be allowed with disaggregated serving."""
        spec_config = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        kv_cfg = make_kv_transfer_config(kv_role="kv_consumer")
        cfg = make_vllm_config(
            speculative_config=spec_config,
            kv_transfer_config=kv_cfg,
        )
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "SPD Types" in str(e):
                pytest.fail(f"ngram SpD should be allowed with disagg: {e}")

    def test_draft_model_spd_with_disagg_allowed(self, make_vllm_config, make_kv_transfer_config):
        """draft_model SpD must be allowed with disaggregated serving."""
        spec_config = types.SimpleNamespace(method="draft_model", num_speculative_tokens=3)
        kv_cfg = make_kv_transfer_config(kv_role="kv_consumer")
        cfg = make_vllm_config(
            speculative_config=spec_config,
            kv_transfer_config=kv_cfg,
        )
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "SPD Types" in str(e):
                pytest.fail(f"draft_model SpD should be allowed with disagg: {e}")

    def test_prefix_caching_with_kv_consumer_disabled(self, make_vllm_config, make_kv_transfer_config):
        """Prefix caching with kv_consumer role must be silently disabled.

        Prefix caching is not supported with any disaggregated serving role.
        The platform auto-disables it (sets enable_prefix_caching = False)
        without raising an error — the user's request is silently overridden.
        """
        kv_cfg = make_kv_transfer_config(kv_role="kv_consumer")
        cfg = make_vllm_config(kv_transfer_config=kv_cfg)
        cfg.cache_config.enable_prefix_caching = True
        QaicPlatform.check_and_update_config(cfg)
        assert cfg.cache_config.enable_prefix_caching is False, (
            "Prefix caching must be auto-disabled for kv_consumer role"
        )

    def test_prefix_caching_with_kv_both_disabled(self, make_vllm_config, make_kv_transfer_config):
        """Prefix caching with kv_both role must be silently disabled.

        Same behaviour as kv_consumer — prefix caching is auto-disabled.
        """
        kv_cfg = make_kv_transfer_config(kv_role="kv_both")
        cfg = make_vllm_config(kv_transfer_config=kv_cfg)
        cfg.cache_config.enable_prefix_caching = True
        QaicPlatform.check_and_update_config(cfg)
        assert cfg.cache_config.enable_prefix_caching is False, (
            "Prefix caching must be auto-disabled for kv_both role"
        )

    def test_prefix_caching_with_kv_producer_disabled(
        self, make_vllm_config, make_kv_transfer_config, patch_qaic_executor_import_bug
    ):
        """Prefix caching with kv_producer role must also be silently disabled.

        Prefix caching is not supported with any disaggregated serving role,
        including kv_producer.  The platform auto-disables it.

        kv_role="kv_producer" also runs the "stages" pipeline-parallel check,
        which does `int(override_qaic_config.get("stages"))` with no default —
        raising TypeError when "stages" is absent, and imports
        vllm_qaic.executor.qaic_uniproc_executor, a subpackage that does not
        exist (real module: vllm_qaic.qaic_uniproc_executor). Both are source
        bugs we are not fixing here; "stages" is supplied explicitly and the
        patch_qaic_executor_import_bug fixture stubs the broken import path so
        this test can reach and verify the prefix-caching behaviour under test.
        """
        kv_cfg = make_kv_transfer_config(kv_role="kv_producer")
        cfg = make_vllm_config(kv_transfer_config=kv_cfg)
        cfg.cache_config.enable_prefix_caching = True
        cfg.additional_config = {"override_qaic_config": {"stages": "1"}}
        QaicPlatform.check_and_update_config(cfg)
        assert cfg.cache_config.enable_prefix_caching is False, (
            "Prefix caching must be auto-disabled for kv_producer role"
        )

    def test_stages_assertion_max_num_seqs_exceeds_stages(
        self, make_vllm_config, make_kv_transfer_config, patch_qaic_executor_import_bug
    ):
        """max_num_seqs > stages must raise AssertionError."""
        kv_cfg = make_kv_transfer_config(kv_role="kv_producer")
        cfg = make_vllm_config(kv_transfer_config=kv_cfg)
        cfg.scheduler_config.max_num_seqs = 8
        cfg.additional_config = {
            "override_qaic_config": {"stages": "4"}
        }
        with pytest.raises(AssertionError, match="max_num_seqs"):
            QaicPlatform.check_and_update_config(cfg)

    def test_stages_assertion_max_num_seqs_equals_stages_allowed(
        self, make_vllm_config, make_kv_transfer_config, patch_qaic_executor_import_bug
    ):
        """max_num_seqs == stages must be allowed."""
        kv_cfg = make_kv_transfer_config(kv_role="kv_producer")
        cfg = make_vllm_config(kv_transfer_config=kv_cfg)
        cfg.scheduler_config.max_num_seqs = 4
        cfg.additional_config = {
            "override_qaic_config": {"stages": "4"}
        }
        QaicPlatform.check_and_update_config(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
