# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC sampling parameters (pure Python, no hardware).

On-device sampler smoke tests (presence/frequency/repetition penalty,
top-p, top-k) live in
vllm-qaic/tests/device_unit_and_e2e/test_qaic_samplers.py
(TestQaicSamplers) — this file's former TestSamplersOnDevice class covered
the same ground via a nonexistent `llm` fixture and has been removed.

Coverage areas
--------------
1. aic_include_sampler config parsing (on-device sampling flag)
2. ODS + SpD mutual exclusion assertion
"""

import types

import pytest


# ===========================================================================
# 1. aic_include_sampler config parsing
# ===========================================================================

class TestAicIncludeSamplerParsing:
    """
    aic_include_sampler enables on-device sampling (ODS).
    It must be parsed from override_qaic_config as a bool string.
    """

    def test_true_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": "true"})
        assert result.get("aic_include_sampler") is True

    def test_false_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": "false"})
        assert result.get("aic_include_sampler") is False

    def test_one_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": "1"})
        assert result.get("aic_include_sampler") is True

    def test_zero_string(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": "0"})
        assert result.get("aic_include_sampler") is False

    def test_bool_true(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": True})
        assert result.get("aic_include_sampler") is True

    def test_bool_false(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": False})
        assert result.get("aic_include_sampler") is False


# ===========================================================================
# 2. ODS + SpD mutual exclusion
# ===========================================================================

class TestODSSpDMutualExclusion:
    """
    On-device sampling (aic_include_sampler=True) and speculative decoding
    cannot be used together.  check_and_update_config() must raise.
    """

    def test_ods_with_spd_raises(self):
        """aic_include_sampler=True + speculative_config must raise AssertionError."""
        from vllm_qaic.platform_base import QaicPlatform
        import torch

        spec_config = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        model_config = types.SimpleNamespace(
            max_model_len=2048,
            hf_config=types.SimpleNamespace(model_type="llama"),
            is_multimodal_model=False,
            runner_type="generate",
            mm_processor_kwargs=None,
            enforce_eager=False,
        )
        device_config = types.SimpleNamespace(device=torch.device("cpu"), device_type="cpu")
        scheduler_config = types.SimpleNamespace(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=128,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            async_scheduling=False,
        )
        cache_config = types.SimpleNamespace(
            enable_prefix_caching=False, block_size=16,
            mamba_block_size=2048, mamba_cache_mode="none",
        )
        parallel_config = types.SimpleNamespace(
            worker_cls="auto", world_size=1, distributed_executor_backend=None,
        )
        vllm_config = types.SimpleNamespace(
            additional_config={"override_qaic_config": {"aic_include_sampler": True}},
            model_config=model_config, device_config=device_config,
            scheduler_config=scheduler_config, cache_config=cache_config,
            parallel_config=parallel_config, lora_config=None,
            speculative_config=spec_config, kv_transfer_config=None,
            compilation_config=None,
        )
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(vllm_config)

    def test_ods_without_spd_allowed(self):
        """aic_include_sampler=True without speculative_config must not raise."""
        from vllm_qaic.platform_base import QaicPlatform
        import torch

        model_config = types.SimpleNamespace(
            max_model_len=2048,
            hf_config=types.SimpleNamespace(model_type="llama"),
            is_multimodal_model=False,
            runner_type="generate",
            mm_processor_kwargs=None,
            enforce_eager=False,
        )
        device_config = types.SimpleNamespace(device=torch.device("cpu"), device_type="cpu")
        scheduler_config = types.SimpleNamespace(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=128,
            max_num_seqs=4,
            max_num_batched_tokens=512,
            async_scheduling=False,
        )
        cache_config = types.SimpleNamespace(
            enable_prefix_caching=False, block_size=16,
            mamba_block_size=2048, mamba_cache_mode="none",
        )
        parallel_config = types.SimpleNamespace(
            worker_cls="auto", world_size=1, distributed_executor_backend=None,
        )
        vllm_config = types.SimpleNamespace(
            additional_config={"override_qaic_config": {"aic_include_sampler": True}},
            model_config=model_config, device_config=device_config,
            scheduler_config=scheduler_config, cache_config=cache_config,
            parallel_config=parallel_config, lora_config=None,
            speculative_config=None, kv_transfer_config=None,
            compilation_config=None,
        )
        try:
            QaicPlatform.check_and_update_config(vllm_config)
        except AssertionError as e:
            if "On-device sampling" in str(e) or "aic_include_sampler" in str(e):
                pytest.fail(f"ODS without SpD should be allowed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
