# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC benchmark configuration.

The benchmark CI scripts (run_benchmark_test_qaic.sh) run offline and online
throughput/latency benchmarks.  These unit tests cover the QAIC-specific
configuration logic — specifically the override_qaic_config normalisation
that is applied before the QPC is compiled.

No hardware, model loading, or network access required.

Coverage areas
--------------
1. _clean_config() — benchmark-specific keys: num_cores=14, mos=4, aic_num_cores alias
2. QPC path preservation — qpc_path must not be lowercased or modified
3. Device group parsing — benchmark runs specify device groups
4. Ignore-list keys — batch_size, full_batch_size must be dropped

Note: mxfp6/mxint8 normalisation and ctx_len ignore are tested in
accuracy/test_accuracy_config.py and are not duplicated here.
"""

import pytest


# ===========================================================================
# 1. _clean_config() — benchmark-specific keys
# ===========================================================================

class TestCleanConfigBenchmark:
    """
    Benchmark runs pass override_qaic_config with num_cores, mos, qpc_path.
    Verify these are correctly normalised before being forwarded to the compiler.

    Note: mxfp6/mxint8 normalisation is tested in accuracy/test_accuracy_config.py.
    This class only tests benchmark-specific values (num_cores=14, mos=4).
    """

    def test_num_cores_string_to_int(self):
        """Benchmark typically uses 14 cores (not 16 like accuracy runs)."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"num_cores": "14"})
        assert result["num_cores"] == 14

    def test_mos_string_to_int(self):
        """Benchmark typically uses mos=4."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"mos": "4"})
        assert result["mos"] == 4

    def test_aic_num_cores_alias(self):
        """aic_num_cores is an alias for num_cores — both map to the same key."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_num_cores": "16"})
        assert result["num_cores"] == 16

    def test_typical_benchmark_config(self):
        """Typical benchmark config: num_cores=14, mos=4, mxfp6=true, mxint8=true."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({
            "num_cores": "14",
            "mos": "4",
            "mxfp6": "true",
            "mxint8": "true",
        })
        assert result["num_cores"] == 14
        assert result["mos"] == 4
        assert result.get("mxfp6_matmul") is True
        assert result.get("mxint8_kv_cache") is True


# ===========================================================================
# 2. QPC path preservation
# ===========================================================================

class TestQpcPathPreservation:
    """
    Benchmark runs specify a pre-compiled QPC path.
    The path must be preserved verbatim — not lowercased or modified.
    """

    def test_qpc_path_preserved_verbatim(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        path = "/opt/QPC/MyModel_v2/qpc"
        result = _clean_config({"qpc_path": path})
        assert result["qpc_path"] == path

    def test_qpc_path_with_uppercase_preserved(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        path = "/opt/QPC/Llama-3.1-8B-Instruct/qpc"
        result = _clean_config({"qpc_path": path})
        assert result["qpc_path"] == path

    def test_mdp_load_partition_config_preserved(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        path = "/opt/MDP/partition.json"
        result = _clean_config({"mdp_load_partition_config": path})
        assert result["mdp_load_partition_config"] == path


# ===========================================================================
# 3. Device group parsing for benchmark runs
# ===========================================================================

class TestDeviceGroupBenchmark:
    """
    Benchmark runs specify device groups (e.g. [0,1,2,3] for 4-device runs).
    Verify that device_group is correctly parsed.
    """

    def test_device_group_list(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"device_group": [0, 1, 2, 3]})
        assert result["device_group"] == [0, 1, 2, 3]
        assert result["num_devices"] == 4

    def test_device_group_string_bracket(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"device_group": "[0,1,2,3]"})
        assert result["device_group"] == [0, 1, 2, 3]
        assert result["num_devices"] == 4

    def test_single_device(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"device_id": 0})
        assert result["device_group"] == [0]
        assert result["num_devices"] == 1


# ===========================================================================
# 4. Ignore-list keys must be dropped
# ===========================================================================

class TestIgnoreListBenchmark:
    """
    Keys like batch_size and full_batch_size are benchmark parameters that must
    NOT be forwarded to the QPC compiler.

    Note: ctx_len ignore is tested in accuracy/test_accuracy_config.py.
    """

    def test_batch_size_ignored(self):
        """batch_size is a benchmark param, must not reach the compiler."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"batch_size": "8", "num_cores": "14"})
        assert "batch_size" not in result
        assert result["num_cores"] == 14

    def test_full_batch_size_ignored(self):
        """full_batch_size is a benchmark param, must not reach the compiler."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"full_batch_size": "16"})
        assert "full_batch_size" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
