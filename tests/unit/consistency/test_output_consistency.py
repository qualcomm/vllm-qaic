# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC output consistency (pure Python, no hardware).

On-device greedy determinism / batch-consistency coverage lives in
vllm-qaic/tests/device_unit_and_e2e/test_qaic_output_consistency.py
(TestQaicOutputConsistency), which predates this file's extraction from the
on-device suite and already covers the same ground via the device/ fixture
system.

Coverage areas
--------------
1. _clean_config() normalisation — keys that affect output consistency
"""

import pytest


# ===========================================================================
# 1. _clean_config() keys that affect output consistency
# ===========================================================================

class TestCleanConfigConsistency:
    """
    _clean_config() normalises override_qaic_config before it is forwarded
    to the QPC compiler.  Incorrect normalisation can silently change model
    behaviour and break output consistency.
    """

    def test_mxfp6_normalised_to_bool(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mxfp6": "true"})
        assert result.get("mxfp6_matmul") is True

    def test_mxint8_normalised_to_bool(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mxint8": "true"})
        assert result.get("mxint8_kv_cache") is True

    def test_num_cores_normalised_to_int(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"num_cores": "16"})
        assert result["num_cores"] == 16

    def test_mos_normalised_to_int(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mos": "2"})
        assert result["mos"] == 2

    def test_dfs_normalised_to_bool(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"dfs": "1"})
        assert result.get("aic_enable_depth_first") is True

    def test_none_config_returns_empty(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config(None) == {}

    def test_empty_config_returns_empty(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        assert _clean_config({}) == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
