# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.patch.patch_config.

Tests QaicCacheConfig and QaicDeviceConfig — the monkey-patched vLLM config
classes that add QAIC-specific behaviour.  No hardware required.

Coverage areas
--------------
1. QaicCacheConfig — accepts "mxint8" as a valid QAIC-specific cache_dtype
2. QaicCacheConfig — accepts standard CacheConfig dtypes unchanged
3. QaicDeviceConfig — sets device to cpu when device_type is "qaic" (AOT mode)
4. QaicDeviceConfig — handles explicit string and torch.device inputs
5. Monkey-patch verification — vllm.config.CacheConfig is QaicCacheConfig
6. Monkey-patch verification — vllm.config.DeviceConfig is QaicDeviceConfig
"""

import pytest
import torch

# Import the patch module to trigger the monkey-patching
import vllm_qaic.patch.patch_config  # noqa: F401

import vllm.config
import vllm.engine.arg_utils
from vllm_qaic.patch.patch_config import QaicCacheConfig, QaicDeviceConfig


# ===========================================================================
# 1-2. QaicCacheConfig
# ===========================================================================

class TestQaicCacheConfig:
    """QaicCacheConfig extends CacheConfig to accept 'mxint8' as a valid dtype."""

    def _make_cfg(self, cache_dtype: str) -> QaicCacheConfig:
        return QaicCacheConfig(
            block_size=16,
            gpu_memory_utilization=0.9,
            cache_dtype=cache_dtype,
            num_gpu_blocks_override=None,
        )

    def test_mxint8_accepted(self):
        """'mxint8' is a QAIC-specific KV-cache dtype and must be accepted."""
        cfg = self._make_cfg("mxint8")
        assert cfg.cache_dtype == "mxint8"

    def test_auto_accepted(self):
        """Standard 'auto' dtype must still be accepted."""
        cfg = self._make_cfg("auto")
        assert cfg.cache_dtype == "auto"

    def test_fp8_e5m2_accepted(self):
        cfg = self._make_cfg("fp8_e5m2")
        assert cfg.cache_dtype == "fp8_e5m2"

    def test_fp8_e4m3_accepted(self):
        cfg = self._make_cfg("fp8_e4m3")
        assert cfg.cache_dtype == "fp8_e4m3"

    def test_is_subclass_of_cache_config(self):
        """QaicCacheConfig must be a subclass of the original CacheConfig."""
        from vllm.config.cache import CacheConfig
        assert issubclass(QaicCacheConfig, CacheConfig)


# ===========================================================================
# 3-4. QaicDeviceConfig
# ===========================================================================

class TestQaicDeviceConfig:
    """
    QaicDeviceConfig overrides DeviceConfig so that device_type='qaic' maps
    to torch.device('cpu') — required for AOT (non-eager) inference where
    the actual QAIC device is not a torch device.
    """

    def test_qaic_device_type_sets_cpu_device(self):
        """When device='qaic', the actual torch.device must be cpu (AOT mode)."""
        cfg = QaicDeviceConfig(device="qaic")
        assert cfg.device_type == "qaic"
        assert cfg.device == torch.device("cpu")

    def test_cpu_device_type_sets_cpu_device(self):
        """When device='cpu', torch.device should be cpu."""
        cfg = QaicDeviceConfig(device="cpu")
        assert cfg.device_type == "cpu"
        assert cfg.device == torch.device("cpu")

    def test_string_device_sets_device_type(self):
        """device_type is derived from the string device name."""
        cfg = QaicDeviceConfig(device="qaic")
        assert cfg.device_type == "qaic"

    def test_torch_device_cpu_input(self):
        """Passing torch.device('cpu') should work."""
        cfg = QaicDeviceConfig(device=torch.device("cpu"))
        assert cfg.device_type == "cpu"

    def test_is_subclass_of_device_config(self):
        """QaicDeviceConfig must be a subclass of the original DeviceConfig."""
        from vllm.config.device import DeviceConfig
        assert issubclass(QaicDeviceConfig, DeviceConfig)


# ===========================================================================
# 5-6. Monkey-patch verification
# ===========================================================================

class TestMonkeyPatch:
    """
    Verify that importing vllm_qaic.patch.patch_config replaces the vLLM
    config classes with the QAIC-extended versions.
    """

    def test_vllm_config_cache_config_is_qaic(self):
        """vllm.config.CacheConfig must be replaced with QaicCacheConfig."""
        assert vllm.config.CacheConfig is QaicCacheConfig

    def test_vllm_config_device_config_is_qaic(self):
        """vllm.config.DeviceConfig must be replaced with QaicDeviceConfig."""
        assert vllm.config.DeviceConfig is QaicDeviceConfig

    def test_arg_utils_cache_config_is_qaic(self):
        """vllm.engine.arg_utils.CacheConfig must also be patched."""
        assert vllm.engine.arg_utils.CacheConfig is QaicCacheConfig

    def test_arg_utils_device_config_is_qaic(self):
        """vllm.engine.arg_utils.DeviceConfig must also be patched."""
        assert vllm.engine.arg_utils.DeviceConfig is QaicDeviceConfig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
