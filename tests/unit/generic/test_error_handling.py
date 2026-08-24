# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC error handling and config validation (pure Python, no hardware).

Tests that the plugin fails correctly (raises the right exception, not silently
or with a crash) when given invalid inputs or unsupported configurations.

On-device runtime error handling tests (TestRuntimeErrorHandling) have
moved to vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_error_handling.py
to use the device/ fixture system.
"""

import types

import pytest

import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base


# ===========================================================================
# Pure Python: config validation errors (no hardware)
# ===========================================================================

class TestConfigValidationErrors:
    """
    Verify that check_and_update_config() raises the correct errors
    for invalid/unsupported configurations.
    """

    def test_lora_with_spd_raises(self, make_vllm_config):
        """LoRA + SpD must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        lora_config = types.SimpleNamespace()
        spec_config = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        cfg = make_vllm_config(lora_config=lora_config, speculative_config=spec_config)
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_lora_with_multimodal_raises(self, make_vllm_config):
        """LoRA + multimodal must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        lora_config = types.SimpleNamespace()
        cfg = make_vllm_config(lora_config=lora_config)
        cfg.model_config.is_multimodal_model = True
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_spd_with_multimodal_raises(self, make_vllm_config):
        """SpD + multimodal must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        spec_config = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        cfg = make_vllm_config(speculative_config=spec_config)
        cfg.model_config.is_multimodal_model = True
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_ods_with_spd_raises(self, make_vllm_config):
        """ODS + SpD must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        spec_config = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        cfg = make_vllm_config(speculative_config=spec_config)
        cfg.additional_config = {"override_qaic_config": {"aic_include_sampler": True}}
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_invalid_additional_config_string_raises(self, make_vllm_config):
        """Invalid additional_config string must raise ValueError."""
        from vllm_qaic.platform_base import QaicPlatform
        cfg = make_vllm_config()
        cfg.additional_config = "this is not valid json or python"
        with pytest.raises(ValueError):
            QaicPlatform.check_and_update_config(cfg)

    def test_non_dict_additional_config_raises(self, make_vllm_config):
        """Non-dict additional_config must raise TypeError."""
        from vllm_qaic.platform_base import QaicPlatform
        cfg = make_vllm_config()
        cfg.additional_config = '["not", "a", "dict"]'
        with pytest.raises(TypeError):
            QaicPlatform.check_and_update_config(cfg)

    def test_prefix_caching_with_kv_consumer_disabled(self, make_vllm_config):
        """Prefix caching + kv_consumer: check_and_update_config() never raises
        here — it silently disables prefix caching first (verified
        empirically; see generic/test_platform.py::
        TestDisaggregatedServingConstraints for the same behaviour).
        """
        from vllm_qaic.platform_base import QaicPlatform
        kv_cfg = types.SimpleNamespace(kv_role="kv_consumer")
        cfg = make_vllm_config(kv_transfer_config=kv_cfg)
        cfg.cache_config.enable_prefix_caching = True
        QaicPlatform.check_and_update_config(cfg)
        assert not cfg.cache_config.enable_prefix_caching

    def test_lora_with_disagg_raises(self, make_vllm_config):
        """LoRA + disaggregated serving must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        lora_config = types.SimpleNamespace()
        kv_cfg = types.SimpleNamespace(kv_role="kv_both")
        cfg = make_vllm_config(lora_config=lora_config, kv_transfer_config=kv_cfg)
        with pytest.raises(AssertionError, match="LORA with Disaggregated"):
            QaicPlatform.check_and_update_config(cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
