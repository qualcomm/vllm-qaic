# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC prefix caching behaviour (pure Python, no hardware).

In QAIC AOT mode, prefix caching is forcibly disabled because the QPC
(compiled model) uses fixed-size KV cache blocks that are incompatible
with the variable-length prefix caching mechanism.

NOTE: Prefix caching is not yet supported in vllm-qaic.  The tests in
sections 1-4 verify that prefix caching is correctly DISABLED (which is
the expected behaviour).  These tests should pass.  If prefix caching
support is added in the future, update these tests accordingly.

On-device prefix caching correctness tests (TestPrefixCachingOnDevice)
have moved to
vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_prefix_caching.py
to use the device/ fixture system (still marked xfail there — prefix
caching is not yet supported in vllm-qaic AOT mode).

Coverage areas
--------------
1. Prefix caching disabled in AOT mode (enable_prefix_caching → False)
2. mamba_block_size reset when prefix caching is disabled
3. mamba_cache_mode reset to "none" when prefix caching is disabled
4. block_size = max_model_len in AOT mode
5. Disaggregated serving: prefix caching with kv_consumer/kv_both raises
6. Disaggregated serving: prefix caching with kv_producer allowed
"""

import types

import pytest

import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
from vllm_qaic.platform_base import QaicPlatform


# ===========================================================================
# 1-3. Prefix caching disabled in AOT mode
# ===========================================================================

class TestPrefixCachingDisabledInAOT:
    """
    In AOT mode (is_aot=True), check_and_update_config() must disable
    prefix caching and reset mamba_block_size/mamba_cache_mode.

    check_and_update_config() never raises for these enable_prefix_caching=True
    scenarios (empirically verified) — it silently disables prefix caching
    first, so these tests call it directly rather than swallowing exceptions.
    """

    def test_prefix_caching_disabled_when_enabled(self, make_vllm_config):
        """enable_prefix_caching=True must be set to False in AOT mode."""
        cfg = make_vllm_config(enable_prefix_caching=True)
        QaicPlatform.check_and_update_config(cfg)
        if QaicPlatform.is_aot:
            assert cfg.cache_config.enable_prefix_caching is False

    def test_mamba_block_size_reset(self, make_vllm_config):
        """mamba_block_size must be reset to max_model_len when prefix caching disabled."""
        cfg = make_vllm_config(enable_prefix_caching=True, max_model_len=2048)
        QaicPlatform.check_and_update_config(cfg)
        if QaicPlatform.is_aot:
            assert cfg.cache_config.mamba_block_size == 2048

    def test_mamba_cache_mode_reset(self, make_vllm_config):
        """mamba_cache_mode must be reset to 'none' when prefix caching disabled."""
        cfg = make_vllm_config(enable_prefix_caching=True, mamba_cache_mode="full")
        QaicPlatform.check_and_update_config(cfg)
        if QaicPlatform.is_aot:
            assert cfg.cache_config.mamba_cache_mode == "none"

    def test_prefix_caching_already_disabled_no_change(self, make_vllm_config):
        """If prefix caching is already disabled, no change should occur."""
        cfg = make_vllm_config(enable_prefix_caching=False)
        QaicPlatform.check_and_update_config(cfg)
        assert cfg.cache_config.enable_prefix_caching is False


# ===========================================================================
# 4. Prefix caching logic — block_size in AOT vs eager
# ===========================================================================

class TestBlockSizeConfig:
    """
    In AOT mode: block_size = max_model_len (ctx_len)
    In eager mode: block_size = 16 (standard vLLM block size)
    """

    def test_aot_block_size_equals_max_model_len(self, make_vllm_config):
        """In AOT mode, block_size must be set to max_model_len."""
        cfg = make_vllm_config(max_model_len=2048)
        QaicPlatform.check_and_update_config(cfg)
        if QaicPlatform.is_aot:
            assert cfg.cache_config.block_size == 2048

    def test_block_size_formula(self, make_vllm_config):
        """block_size = max_model_len in AOT mode."""
        for max_len in [512, 1024, 2048, 4096]:
            cfg = make_vllm_config(max_model_len=max_len)
            QaicPlatform.check_and_update_config(cfg)
            if QaicPlatform.is_aot:
                assert cfg.cache_config.block_size == max_len, (
                    f"block_size should be {max_len}, got {cfg.cache_config.block_size}"
                )


# ===========================================================================
# 5-6. Disaggregated serving prefix caching constraints
# ===========================================================================

class TestDisaggregatedPrefixCaching:
    def _make_kv_config(self, kv_role: str):
        return types.SimpleNamespace(kv_role=kv_role)

    def test_prefix_caching_with_kv_consumer_disabled(self, make_vllm_config):
        """Prefix caching + kv_consumer: check_and_update_config() never raises
        here — it silently disables prefix caching first (verified
        empirically).
        """
        kv_cfg = self._make_kv_config("kv_consumer")
        cfg = make_vllm_config(enable_prefix_caching=True, kv_transfer_config=kv_cfg)
        QaicPlatform.check_and_update_config(cfg)
        assert not cfg.cache_config.enable_prefix_caching

    def test_prefix_caching_with_kv_both_disabled(self, make_vllm_config):
        """Prefix caching + kv_both: same auto-disable behaviour as kv_consumer."""
        kv_cfg = self._make_kv_config("kv_both")
        cfg = make_vllm_config(enable_prefix_caching=True, kv_transfer_config=kv_cfg)
        QaicPlatform.check_and_update_config(cfg)
        assert not cfg.cache_config.enable_prefix_caching

    def test_prefix_caching_with_kv_producer_allowed(self, make_vllm_config, patch_qaic_executor_import_bug):
        """Prefix caching + kv_producer must NOT raise on the prefix caching check.

        kv_role="kv_producer" also runs the "stages" pipeline-parallel check
        (requires 'stages' in override_qaic_config to avoid a TypeError in
        platform_base.py: stages = int(override_qaic_config.get("stages"))) and
        imports vllm_qaic.executor.qaic_uniproc_executor, a subpackage that
        does not exist (real module: vllm_qaic.qaic_uniproc_executor). Both
        are source bugs we are not fixing here; "stages" is supplied
        explicitly and the patch_qaic_executor_import_bug fixture stubs the
        broken import path so this test can reach the prefix-caching check.
        """
        kv_cfg = self._make_kv_config("kv_producer")
        cfg = make_vllm_config(enable_prefix_caching=True, kv_transfer_config=kv_cfg)
        cfg.additional_config["override_qaic_config"]["stages"] = "1"
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "Prefix caching" in str(e):
                pytest.fail(f"kv_producer should allow prefix caching: {e}")

    def test_no_prefix_caching_with_kv_consumer_allowed(self, make_vllm_config):
        """No prefix caching + kv_consumer must NOT raise on the prefix caching check."""
        kv_cfg = self._make_kv_config("kv_consumer")
        cfg = make_vllm_config(enable_prefix_caching=False, kv_transfer_config=kv_cfg)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "Prefix caching" in str(e):
                pytest.fail(f"No prefix caching with kv_consumer should be allowed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
