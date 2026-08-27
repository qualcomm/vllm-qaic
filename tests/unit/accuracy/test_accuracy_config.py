# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC accuracy evaluation configuration.

The accuracy CI script (run_accuracy_test_qaic.sh) uses lm-evaluation-harness
to measure model accuracy on standard benchmarks (ARC, HellaSwag, MMLU, etc.).

These unit tests cover the QAIC-specific configuration logic that affects
accuracy results — specifically the quantization config and override_qaic_config
normalisation that is applied before the QPC is compiled.

No hardware, model loading, or network access required.

Coverage areas
--------------
1. QAIC quantization list — mxfp6 must be present (used for accuracy runs)
2. _clean_config() — accuracy-relevant keys: mxfp6, mxint8, num_cores, mos
3. QaicQuantConfig — quantization config used during accuracy runs
"""

import pytest


# ===========================================================================
# 1. QAIC quantization list
# ===========================================================================

class TestQaicQuantizationList:
    """
    Accuracy runs use mxfp6 quantization.
    Verify that mxfp6 is in the supported quantization list.
    """

    def test_mxfp6_in_quantization_list(self):
        from vllm_qaic.utils import QAIC_QUANTIZATION_LIST
        assert "mxfp6" in QAIC_QUANTIZATION_LIST

    def test_quantization_list_non_empty(self):
        from vllm_qaic.utils import QAIC_QUANTIZATION_LIST
        assert len(QAIC_QUANTIZATION_LIST) > 0

    def test_all_quantization_types_are_strings(self):
        from vllm_qaic.utils import QAIC_QUANTIZATION_LIST
        for q in QAIC_QUANTIZATION_LIST:
            assert isinstance(q, str), f"Expected string, got {type(q)} for {q!r}"

    def test_no_duplicates_in_list(self):
        """No duplicate entries, other than a known one-time re-append.

        vLLM core's ``register_quantization_config()`` decorator (applied to
        ``QaicQuantConfig``) appends ``QAIC_QUANTIZATION_METHOD`` ("mxfp6") to
        ``current_platform.supported_quantization`` the first time
        ``vllm_qaic.quantization.quant_config`` is imported in this process —
        and that list *is* ``QAIC_QUANTIZATION_LIST`` (same object, aliased in
        ``platform_base.py``). So once anything in the test session imports
        that module, "mxfp6" legitimately appears twice. Every other entry
        must still be unique.
        """
        from collections import Counter

        from vllm_qaic.utils import QAIC_QUANTIZATION_LIST, QAIC_QUANTIZATION_METHOD

        counts = Counter(QAIC_QUANTIZATION_LIST)
        for method, count in counts.items():
            allowed = 2 if method == QAIC_QUANTIZATION_METHOD else 1
            assert count <= allowed, f"Unexpected duplicate quantization method: {method!r}"


# ===========================================================================
# 2. _clean_config() — accuracy-relevant keys
# ===========================================================================

class TestCleanConfigAccuracy:
    """
    Accuracy runs pass override_qaic_config with mxfp6, mxint8, num_cores, mos.
    Incorrect normalisation silently changes model behaviour and breaks accuracy.
    """

    def test_mxfp6_true_normalised(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mxfp6": "true"})
        assert result.get("mxfp6_matmul") is True

    def test_mxfp6_false_normalised(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mxfp6": "false"})
        assert result.get("mxfp6_matmul") is False

    def test_mxint8_true_normalised(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mxint8": "true"})
        assert result.get("mxint8_kv_cache") is True

    def test_mxint8_false_normalised(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mxint8": "false"})
        assert result.get("mxint8_kv_cache") is False

    def test_num_cores_string_to_int(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"num_cores": "16"})
        assert result["num_cores"] == 16

    def test_mos_string_to_int(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mos": "2"})
        assert result["mos"] == 2

    def test_typical_accuracy_config(self):
        """Typical accuracy run: mxfp6=true, mxint8=true, num_cores=16, mos=2."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({
            "mxfp6": "true",
            "mxint8": "true",
            "num_cores": "16",
            "mos": "2",
        })
        assert result.get("mxfp6_matmul") is True
        assert result.get("mxint8_kv_cache") is True
        assert result["num_cores"] == 16
        assert result["mos"] == 2

    def test_none_config_returns_empty(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config(None) == {}

    def test_ctx_len_ignored(self):
        """ctx_len is in the ignore list and must not appear in output."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"ctx_len": "2048", "num_cores": "16"})
        assert "ctx_len" not in result
        assert result["num_cores"] == 16


# ===========================================================================
# 3. QaicQuantConfig — quantization config for accuracy runs
# ===========================================================================

class TestQaicQuantConfigAccuracy:
    """
    QaicQuantConfig is the quantization config used when running accuracy
    benchmarks with mxfp6 quantization.
    """

    def test_mxfp6_quant_config_name(self):
        from vllm_qaic.quantization.quant_config import QaicQuantConfig
        cfg = QaicQuantConfig()
        # QaicQuantConfig.get_name() returns the internal registry name "qaic_quant".
        # The "mxfp6" string is the user-facing quantization method name, not the
        # config class name.
        assert cfg.get_name() == "qaic_quant"

    def test_mxfp6_supported_act_dtypes_includes_float16(self):
        import torch
        from vllm_qaic.quantization.quant_config import QaicQuantConfig
        cfg = QaicQuantConfig()
        dtypes = cfg.get_supported_act_dtypes()
        assert torch.float16 in dtypes

    def test_mxfp6_config_filenames_is_list(self):
        from vllm_qaic.quantization.quant_config import QaicQuantConfig
        cfg = QaicQuantConfig()
        filenames = cfg.get_config_filenames()
        assert isinstance(filenames, list)

    def test_platform_supports_mxfp6(self):
        """QaicPlatform.get_supported_quantization() must include mxfp6."""
        from vllm_qaic.platform_base import QaicPlatform
        supported = QaicPlatform.get_supported_quantization()
        assert "mxfp6" in supported


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
