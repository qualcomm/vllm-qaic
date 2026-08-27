# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.utils.qaic_utils._clean_config.

_clean_config is a pure function that normalises the ``override_qaic_config``
dict coming from the CLI / additional_config before it is forwarded to the
QPC compiler.  No QAIC hardware, no model loading, and no network access are
required.

Coverage areas
--------------
1. Empty / None input
2. compiler_args string parsing
3. Key normalisation (lowercase, hyphen → underscore)
4. device_id / device_group parsing (int, list, string variants)
5. num_cores / num_devices parsing
6. mxfp6 variants → mxfp6_matmul
7. mxint8 variants → mxint8_kv_cache
8. dfs / aic_enable_depth_first mapping
9. mos / mdts_mos parsing
10. node_precision_info (bool string vs file path)
11. Ignore-list keys (ctx_len, batch_size, …) are dropped
12. Boolean string normalisation ("true"/"1" → True, "false"/"0" → False)
13. comp_ctx_lengths_prefill / comp_ctx_lengths_decode parsing
14. embed_seq_len parsing
15. Passthrough of unknown compiler args
"""

import types

import pytest

# ---------------------------------------------------------------------------
# Import the function under test
# ---------------------------------------------------------------------------
from vllm_qaic.utils.qaic_utils import _clean_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vllm_config(max_model_len: int = 2048):
    """Return a minimal mock VllmConfig with only the fields _clean_config reads."""
    model_config = types.SimpleNamespace(max_model_len=max_model_len)
    return types.SimpleNamespace(model_config=model_config)


# ===========================================================================
# 1. Empty / None input
# ===========================================================================

class TestEmptyAndNoneInput:
    def test_none_returns_empty_dict(self):
        assert _clean_config(None) == {}

    def test_empty_dict_returns_empty_dict(self):
        assert _clean_config({}) == {}

    def test_none_values_are_skipped(self):
        """Keys whose value is None must not appear in the output."""
        result = _clean_config({"num_cores": None, "mos": None})
        assert result == {}


# ===========================================================================
# 2. compiler_args string parsing
# ===========================================================================

class TestCompilerArgsParsing:
    def test_single_flag_no_value(self):
        """A bare flag like '--aic-hw' becomes '__aic_hw'=True after key normalisation
        (hyphens are replaced by underscores, so '--aic-hw' → '__aic_hw')."""
        result = _clean_config({"compiler_args": "--aic-hw"})
        assert result.get("__aic_hw") is True

    def test_key_value_pair(self):
        """'key=value' is split into key → value."""
        result = _clean_config({"compiler_args": "aic_num_cores=16"})
        assert result.get("num_cores") == 16

    def test_multiple_args_space_separated(self):
        """Multiple args separated by spaces are all parsed."""
        result = _clean_config({"compiler_args": "mos=2 aic_num_cores=8"})
        assert result.get("mos") == 2
        assert result.get("num_cores") == 8

    def test_multiple_args_pipe_separated(self):
        """Pipe-separated args are also supported."""
        result = _clean_config({"compiler_args": "mos=4|aic_num_cores=16"})
        assert result.get("mos") == 4
        assert result.get("num_cores") == 16

    def test_compiler_args_key_removed_from_output(self):
        """The 'compiler_args' key itself must not appear in the output."""
        result = _clean_config({"compiler_args": "mos=1"})
        assert "compiler_args" not in result


# ===========================================================================
# 3. Key normalisation
# ===========================================================================

class TestKeyNormalisation:
    def test_uppercase_key_lowercased(self):
        result = _clean_config({"NUM_CORES": "8"})
        assert result.get("num_cores") == 8

    def test_hyphen_in_key_replaced_by_underscore(self):
        result = _clean_config({"aic-num-cores": "4"})
        assert result.get("num_cores") == 4

    def test_mixed_case_and_hyphen(self):
        result = _clean_config({"AIC-Num-Cores": "16"})
        assert result.get("num_cores") == 16

    def test_leading_trailing_whitespace_stripped(self):
        result = _clean_config({"  num_cores  ": "8"})
        assert result.get("num_cores") == 8


# ===========================================================================
# 4. device_id / device_group parsing
# ===========================================================================

class TestDeviceGroupParsing:
    def test_single_int(self):
        result = _clean_config({"device_id": 0})
        assert result["device_group"] == [0]
        assert result["num_devices"] == 1

    def test_list_of_ints(self):
        result = _clean_config({"device_group": [0, 1, 2, 3]})
        assert result["device_group"] == [0, 1, 2, 3]
        assert result["num_devices"] == 4

    def test_string_bracket_notation(self):
        result = _clean_config({"device_group": "[0,1,2,3]"})
        assert result["device_group"] == [0, 1, 2, 3]
        assert result["num_devices"] == 4

    def test_string_comma_separated(self):
        result = _clean_config({"device_ids": "0,1"})
        assert result["device_group"] == [0, 1]
        assert result["num_devices"] == 2

    def test_string_space_separated(self):
        result = _clean_config({"device_group": "0 1 2"})
        assert result["device_group"] == [0, 1, 2]
        assert result["num_devices"] == 3

    def test_single_device_string(self):
        result = _clean_config({"device_group": "0"})
        assert result["device_group"] == [0]
        assert result["num_devices"] == 1


# ===========================================================================
# 5. num_cores / num_devices
# ===========================================================================

class TestNumCoresAndDevices:
    def test_num_cores_int(self):
        result = _clean_config({"num_cores": 16})
        assert result["num_cores"] == 16

    def test_num_cores_string(self):
        result = _clean_config({"num_cores": "8"})
        assert result["num_cores"] == 8

    def test_aic_num_cores_alias(self):
        result = _clean_config({"aic_num_cores": "14"})
        assert result["num_cores"] == 14

    def test_num_devices_int(self):
        result = _clean_config({"num_devices": 4})
        assert result["num_devices"] == 4

    def test_num_devices_string(self):
        result = _clean_config({"num_devices": "2"})
        assert result["num_devices"] == 2


# ===========================================================================
# 6. mxfp6 variants → mxfp6_matmul
# ===========================================================================

class TestMxfp6Variants:
    @pytest.mark.parametrize("key", ["mxfp6", "mxfp6_matmul", "mxfp6_en"])
    def test_mxfp6_key_true(self, key):
        result = _clean_config({key: "true"})
        assert result.get("mxfp6_matmul") is True

    @pytest.mark.parametrize("key", ["mxfp6", "mxfp6_matmul", "mxfp6_en"])
    def test_mxfp6_key_false(self, key):
        result = _clean_config({key: "false"})
        assert result.get("mxfp6_matmul") is False

    def test_mxfp6_value_as_key_name(self):
        """When the value itself is 'mxfp6', mxfp6_matmul should be True."""
        result = _clean_config({"quantization": "mxfp6"})
        assert result.get("mxfp6_matmul") is True

    def test_mxfp6_disabled_with_zero(self):
        result = _clean_config({"mxfp6": "0"})
        assert result.get("mxfp6_matmul") is False


# ===========================================================================
# 7. mxint8 variants → mxint8_kv_cache
# ===========================================================================

class TestMxint8Variants:
    @pytest.mark.parametrize("key", ["mxint8", "mxint8_en", "mxint8_kv_cache"])
    def test_mxint8_key_true(self, key):
        result = _clean_config({key: "true"})
        assert result.get("mxint8_kv_cache") is True

    @pytest.mark.parametrize("key", ["mxint8", "mxint8_en", "mxint8_kv_cache"])
    def test_mxint8_key_false(self, key):
        result = _clean_config({key: "false"})
        assert result.get("mxint8_kv_cache") is False

    def test_mxint8_value_as_key_name(self):
        result = _clean_config({"kv_cache": "mxint8"})
        assert result.get("mxint8_kv_cache") is True

    def test_mxint8_disabled_with_zero(self):
        result = _clean_config({"mxint8": "0"})
        assert result.get("mxint8_kv_cache") is False


# ===========================================================================
# 8. dfs / aic_enable_depth_first
# ===========================================================================

class TestDfsMapping:
    @pytest.mark.parametrize("key", ["dfs", "aic_enable_depth_first"])
    def test_dfs_enabled(self, key):
        result = _clean_config({key: "true"})
        assert result.get("aic_enable_depth_first") is True

    @pytest.mark.parametrize("key", ["dfs", "aic_enable_depth_first"])
    def test_dfs_disabled(self, key):
        result = _clean_config({key: "false"})
        assert result.get("aic_enable_depth_first") is False

    def test_dfs_disabled_with_zero(self):
        result = _clean_config({"dfs": "0"})
        assert result.get("aic_enable_depth_first") is False

    def test_dfs_enabled_with_one(self):
        result = _clean_config({"dfs": "1"})
        assert result.get("aic_enable_depth_first") is True


# ===========================================================================
# 9. mos / mdts_mos
# ===========================================================================

class TestMosParsing:
    def test_mos_int(self):
        result = _clean_config({"mos": 2})
        assert result["mos"] == 2

    def test_mos_string(self):
        result = _clean_config({"mos": "4"})
        assert result["mos"] == 4

    def test_mdts_mos_int(self):
        result = _clean_config({"mdts_mos": 3})
        assert result["mdts_mos"] == 3

    def test_mdts_mos_string(self):
        result = _clean_config({"mdts_mos": "2"})
        assert result["mdts_mos"] == 2


# ===========================================================================
# 10. node_precision_info
# ===========================================================================

class TestNodePrecisionInfo:
    def test_true_string(self):
        result = _clean_config({"node_precision_info": "true"})
        assert result["node_precision_info"] is True

    def test_one_string(self):
        result = _clean_config({"node_precision_info": "1"})
        assert result["node_precision_info"] is True

    def test_false_string(self):
        result = _clean_config({"node_precision_info": "false"})
        assert result["node_precision_info"] is False

    def test_zero_string(self):
        result = _clean_config({"node_precision_info": "0"})
        assert result["node_precision_info"] is False

    def test_file_path_preserved(self):
        """A file path must be passed through as-is (not converted to bool)."""
        path = "/path/to/npi.yaml"
        result = _clean_config({"node_precision_info": path})
        assert result["node_precision_info"] == path

    def test_relative_path_preserved(self):
        path = "configs/npi.yaml"
        result = _clean_config({"node_precision_info": path})
        assert result["node_precision_info"] == path


# ===========================================================================
# 11. Ignore-list keys are dropped
# ===========================================================================

class TestIgnoreList:
    @pytest.mark.parametrize("key", ["ctx_len", "batch_size", "full_batch_size", "num_speculative_tokens"])
    def test_ignored_key_not_in_output(self, key):
        result = _clean_config({key: "128"})
        assert key not in result, f"Key '{key}' should be in the ignore list and absent from output"

    def test_ignored_keys_alongside_valid_keys(self):
        result = _clean_config({"ctx_len": "512", "num_cores": "8"})
        assert "ctx_len" not in result
        assert result["num_cores"] == 8


# ===========================================================================
# 12. Boolean string normalisation
# ===========================================================================

class TestBooleanStringNormalisation:
    @pytest.mark.parametrize("val", ["true", "1", ""])
    def test_truthy_strings_become_true(self, val):
        result = _clean_config({"some_flag": val})
        assert result.get("some_flag") is True, f"Expected True for value={val!r}"

    @pytest.mark.parametrize("val", ["false", "0"])
    def test_falsy_strings_become_false(self, val):
        result = _clean_config({"some_flag": val})
        assert result.get("some_flag") is False, f"Expected False for value={val!r}"

    def test_non_bool_string_passes_through(self):
        result = _clean_config({"qpc_path": "/some/path"})
        assert result["qpc_path"] == "/some/path"


# ===========================================================================
# 13. comp_ctx_lengths_prefill / comp_ctx_lengths_decode
# ===========================================================================

class TestCompCtxLengths:
    def test_prefill_string_list(self):
        cfg = _make_vllm_config(max_model_len=2048)
        result = _clean_config({"comp_ctx_lengths_prefill": "128,256,512"}, cfg)
        assert result["comp_ctx_lengths_prefill"] == [128, 256, 512]

    def test_decode_string_list(self):
        cfg = _make_vllm_config(max_model_len=2048)
        result = _clean_config({"comp_ctx_lengths_decode": "64,128"}, cfg)
        assert result["comp_ctx_lengths_decode"] == [64, 128]

    def test_values_are_sorted(self):
        cfg = _make_vllm_config(max_model_len=2048)
        result = _clean_config({"comp_ctx_lengths_prefill": "512,128,256"}, cfg)
        assert result["comp_ctx_lengths_prefill"] == [128, 256, 512]

    def test_values_exceeding_max_model_len_are_dropped(self):
        """Values > max_model_len must not appear in the output."""
        cfg = _make_vllm_config(max_model_len=512)
        result = _clean_config({"comp_ctx_lengths_prefill": "128,256,1024"}, cfg)
        # The whole list is invalid (1024 > 512), so the key should be absent.
        assert "comp_ctx_lengths_prefill" not in result

    def test_list_input_accepted(self):
        cfg = _make_vllm_config(max_model_len=2048)
        result = _clean_config({"comp_ctx_lengths_prefill": [128, 256]}, cfg)
        assert result["comp_ctx_lengths_prefill"] == [128, 256]


# ===========================================================================
# 14. embed_seq_len parsing
# ===========================================================================

class TestEmbedSeqLen:
    def test_string_list_includes_max_model_len(self):
        cfg = _make_vllm_config(max_model_len=512)
        result = _clean_config({"embed_seq_len": "128,256,512"}, cfg)
        assert result["prefill_seq_len"] == [128, 256, 512]

    def test_int_equal_to_max_model_len_raises(self):
        """When embed_seq_len is passed as a bare int the implementation
        asserts ``value == max_model_len`` but then tries
        ``max_model_len in value`` (int), which raises TypeError because
        an int is not iterable.  This test documents that known limitation."""
        cfg = _make_vllm_config(max_model_len=512)
        with pytest.raises(TypeError):
            _clean_config({"embed_seq_len": 512}, cfg)

    def test_string_list_missing_max_model_len_raises(self):
        """If max_model_len is not in the list, an AssertionError must be raised."""
        cfg = _make_vllm_config(max_model_len=1024)
        with pytest.raises(AssertionError):
            _clean_config({"embed_seq_len": "128,256,512"}, cfg)


# ===========================================================================
# 15. Passthrough of unknown compiler args
# ===========================================================================

class TestPassthroughArgs:
    def test_unknown_string_arg_passes_through(self):
        result = _clean_config({"aic_pmu_recipe": "some_recipe"})
        assert result["aic_pmu_recipe"] == "some_recipe"

    def test_qpc_path_preserved_verbatim(self):
        """qpc_path must not be lowercased."""
        path = "/opt/QPC/MyModel_v2"
        result = _clean_config({"qpc_path": path})
        assert result["qpc_path"] == path

    def test_mdp_load_partition_config_preserved(self):
        path = "/opt/MDP/partition.json"
        result = _clean_config({"mdp_load_partition_config": path})
        assert result["mdp_load_partition_config"] == path

    def test_numeric_string_passes_through(self):
        result = _clean_config({"prefill_seq_len": "128"})
        assert result["prefill_seq_len"] == "128"

    def test_multiple_keys_all_processed(self):
        result = _clean_config({
            "num_cores": "8",
            "mos": "2",
            "mxfp6": "true",
            "mxint8": "false",
            "dfs": "true",
        })
        assert result["num_cores"] == 8
        assert result["mos"] == 2
        assert result["mxfp6_matmul"] is True
        assert result["mxint8_kv_cache"] is False
        assert result["aic_enable_depth_first"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
