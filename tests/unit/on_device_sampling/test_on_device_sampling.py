# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC on-device sampling (ODS) config parsing (pure Python, no hardware).

On-device sampling (aic_include_sampler=True) moves the token sampling step
from the host CPU onto the QAIC device, reducing PCIe round-trips.

On-device ODS inference tests (TestOnDeviceSamplingOnDevice) have moved to
vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_on_device_sampling.py
to use the device/ fixture system (still marked xfail there — ODS is not
yet fully supported in vllm-qaic).

Coverage areas
--------------
1. aic_include_sampler config parsing (bool strings, bool values)
2. ODS + SpD mutual exclusion (AssertionError)
"""

import types

import pytest


# ===========================================================================
# 1. aic_include_sampler config parsing
# ===========================================================================

class TestAicIncludeSamplerParsing:
    def test_true_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config({"aic_include_sampler": "true"}).get("aic_include_sampler") is True

    def test_false_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config({"aic_include_sampler": "false"}).get("aic_include_sampler") is False

    def test_one_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config({"aic_include_sampler": "1"}).get("aic_include_sampler") is True

    def test_zero_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config({"aic_include_sampler": "0"}).get("aic_include_sampler") is False

    def test_absent_key_not_in_result(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"num_cores": "8"})
        # aic_include_sampler should not appear if not set
        assert result.get("aic_include_sampler") is None or "aic_include_sampler" not in result

    def test_ods_flag_does_not_affect_other_keys(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": "true", "num_cores": "8"})
        assert result.get("aic_include_sampler") is True
        assert result.get("num_cores") == 8


# ===========================================================================
# 2. ODS + SpD mutual exclusion
# ===========================================================================

class TestODSConstraints:
    def _make_vllm_config(self, ods_enabled: bool, spec_config=None):
        import torch
        model_config = types.SimpleNamespace(
            max_model_len=2048,
            hf_config=types.SimpleNamespace(model_type="llama"),
            is_multimodal_model=False, runner_type="generate",
            mm_processor_kwargs=None, enforce_eager=False,
        )
        device_config = types.SimpleNamespace(device=torch.device("cpu"), device_type="cpu")
        scheduler_config = types.SimpleNamespace(
            enable_chunked_prefill=True, long_prefill_token_threshold=128,
            max_num_seqs=4, max_num_batched_tokens=512, async_scheduling=False,
        )
        cache_config = types.SimpleNamespace(
            enable_prefix_caching=False, block_size=16,
            mamba_block_size=2048, mamba_cache_mode="none",
        )
        parallel_config = types.SimpleNamespace(
            worker_cls="auto", world_size=1, distributed_executor_backend=None,
        )
        return types.SimpleNamespace(
            additional_config={"override_qaic_config": {"aic_include_sampler": ods_enabled}},
            model_config=model_config, device_config=device_config,
            scheduler_config=scheduler_config, cache_config=cache_config,
            parallel_config=parallel_config, lora_config=None,
            speculative_config=spec_config, kv_transfer_config=None,
            compilation_config=None,
        )

    def test_ods_with_spd_raises(self):
        """ODS + SpD must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        spec = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        cfg = self._make_vllm_config(ods_enabled=True, spec_config=spec)
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_ods_without_spd_allowed(self):
        """ODS without SpD must not raise on the ODS check."""
        from vllm_qaic.platform_base import QaicPlatform
        cfg = self._make_vllm_config(ods_enabled=True, spec_config=None)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "On-device sampling" in str(e):
                pytest.fail(f"ODS without SpD should be allowed: {e}")

    def test_ods_disabled_with_spd_allowed(self):
        """ODS disabled + SpD must not raise on the ODS check."""
        from vllm_qaic.platform_base import QaicPlatform
        spec = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        cfg = self._make_vllm_config(ods_enabled=False, spec_config=spec)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "On-device sampling" in str(e):
                pytest.fail(f"ODS=False with SpD should be allowed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
