# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.quantization.quant_config.

Tests QaicQuantConfig — the QAIC-specific quantization configuration class
that supports mxfp6 and other QAIC-compatible quantization methods.
No hardware required.

Coverage areas
--------------
1. QAIC_QUANTIZATION_LIST — contains expected methods
2. QAIC_QUANTIZATION_METHOD — is "mxfp6"
3. QAIC_KV_CACHE_DTYPE — is "mxint8"
4. QaicQuantConfig.get_name() — returns "qaic_quant"
5. QaicQuantConfig.get_supported_act_dtypes() — returns expected dtypes
6. QaicQuantConfig.get_config_filenames() — returns empty list
7. QaicQuantConfig.from_config() — creates instance correctly
8. QaicQuantConfig.get_quant_method() — returns None (QAIC handles quant on-chip)
9. QaicQuantConfig.override_quantization_method() — logic for mxfp6 override
10. QaicQuantConfig.get_scaled_act_names() — returns empty list
"""

import pytest
import torch

from vllm_qaic.utils import (
    QAIC_KV_CACHE_DTYPE,
    QAIC_QUANTIZATION_LIST,
    QAIC_QUANTIZATION_METHOD,
)
from vllm_qaic.quantization.quant_config import QaicQuantConfig


# ===========================================================================
# 1-3. QAIC constants
# ===========================================================================

class TestQaicConstants:
    def test_quantization_method_is_mxfp6(self):
        assert QAIC_QUANTIZATION_METHOD == "mxfp6"

    def test_kv_cache_dtype_is_mxint8(self):
        assert QAIC_KV_CACHE_DTYPE == "mxint8"

    def test_quantization_list_contains_mxfp6(self):
        assert "mxfp6" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_contains_awq(self):
        assert "awq" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_contains_gptq(self):
        assert "gptq" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_contains_fp8(self):
        assert "fp8" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_contains_compressed_tensors(self):
        assert "compressed-tensors" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_contains_mxfp4(self):
        assert "mxfp4" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_is_non_empty(self):
        assert len(QAIC_QUANTIZATION_LIST) > 0

    def test_quantization_method_in_list(self):
        """The primary quantization method must be in the supported list."""
        assert QAIC_QUANTIZATION_METHOD in QAIC_QUANTIZATION_LIST


# ===========================================================================
# 4-10. QaicQuantConfig
# ===========================================================================

class TestQaicQuantConfig:
    def test_get_name(self):
        """get_name() must return 'qaic_quant'."""
        cfg = QaicQuantConfig()
        assert cfg.get_name() == "qaic_quant"

    def test_get_supported_act_dtypes(self):
        """Supported activation dtypes must include float16, bfloat16, float32."""
        dtypes = QaicQuantConfig.get_supported_act_dtypes()
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes
        assert torch.float32 in dtypes

    def test_get_config_filenames_returns_empty(self):
        """QAIC quantization does not use config files — must return []."""
        assert QaicQuantConfig.get_config_filenames() == []

    def test_from_config_with_quant_method(self):
        """from_config() must extract quantize_method from config dict."""
        cfg = QaicQuantConfig.from_config({"quant_method": "mxfp6"})
        assert cfg.quantize_method == "mxfp6"

    def test_from_config_with_quantize_method_key(self):
        """from_config() must also accept 'quantize_method' key."""
        cfg = QaicQuantConfig.from_config({"quantize_method": "awq"})
        assert cfg.quantize_method == "awq"

    def test_from_config_empty_dict(self):
        """from_config() with empty dict must return instance with None method."""
        cfg = QaicQuantConfig.from_config({})
        assert cfg.quantize_method is None

    def test_get_quant_method_returns_none(self):
        """
        QAIC handles quantization on-chip; get_quant_method() must return None
        so vLLM does not apply any software quantization layer.
        """
        cfg = QaicQuantConfig()
        result = cfg.get_quant_method(layer=None, prefix="")
        assert result is None

    def test_get_scaled_act_names_returns_empty(self):
        """QAIC does not use scaled activations — must return []."""
        cfg = QaicQuantConfig()
        assert cfg.get_scaled_act_names() == []

    def test_default_quantize_method_is_none(self):
        """Default constructor must have quantize_method=None."""
        cfg = QaicQuantConfig()
        assert cfg.quantize_method is None

    def test_repr_contains_qaic(self):
        """__repr__ must mention QaicQuantConfig."""
        cfg = QaicQuantConfig()
        assert "QaicQuantConfig" in repr(cfg)


class TestQaicQuantConfigOverride:
    """Tests for override_quantization_method() — QAIC-specific override logic."""

    def test_mxfp6_hf_config_with_mxfp6_user_quant(self):
        """
        When HF config has quant_method='mxfp6' and user requests 'mxfp6',
        the override must return 'mxfp6'.
        """
        result = QaicQuantConfig.override_quantization_method(
            hf_quant_cfg={"quant_method": "mxfp6"},
            user_quant=QAIC_QUANTIZATION_METHOD,
        )
        assert result == QAIC_QUANTIZATION_METHOD

    def test_awq_hf_config_with_mxfp6_user_quant(self):
        """
        When HF config has quant_method='awq' (in QAIC_QUANTIZATION_LIST)
        and user requests 'mxfp6', the override must return 'mxfp6'.
        """
        result = QaicQuantConfig.override_quantization_method(
            hf_quant_cfg={"quant_method": "awq"},
            user_quant=QAIC_QUANTIZATION_METHOD,
        )
        assert result == QAIC_QUANTIZATION_METHOD

    def test_unknown_hf_quant_returns_none(self):
        """
        When HF config has an unknown quant_method, override must return None
        (no override applied).
        """
        result = QaicQuantConfig.override_quantization_method(
            hf_quant_cfg={"quant_method": "some_unknown_method"},
            user_quant=QAIC_QUANTIZATION_METHOD,
        )
        assert result is None

    def test_wrong_user_quant_returns_none(self):
        """
        When user_quant is not QAIC_QUANTIZATION_METHOD, override must return None.
        """
        result = QaicQuantConfig.override_quantization_method(
            hf_quant_cfg={"quant_method": "mxfp6"},
            user_quant="bitsandbytes",
        )
        assert result is None

    @pytest.mark.parametrize("method", QAIC_QUANTIZATION_LIST)
    def test_all_supported_methods_trigger_override(self, method):
        """All methods in QAIC_QUANTIZATION_LIST must trigger the override."""
        result = QaicQuantConfig.override_quantization_method(
            hf_quant_cfg={"quant_method": method},
            user_quant=QAIC_QUANTIZATION_METHOD,
        )
        assert result == QAIC_QUANTIZATION_METHOD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
