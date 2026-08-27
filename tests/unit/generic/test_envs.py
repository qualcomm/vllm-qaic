# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.envs — QAIC-specific environment variables.

All tests are pure-Python and require no QAIC hardware.  The conftest.py
``_clean_qaic_env`` fixture removes all VLLM_QAIC_* / QAIC_* variables
before each test so the caller's shell environment cannot interfere.

Coverage areas
--------------
1. Default values when env vars are not set
2. Correct values when env vars are set
3. Boolean conversion (VLLM_QAIC_DFS_EN)
4. Integer conversion (VLLM_QAIC_MAX_CPU_THREADS, VLLM_QAIC_MOS, VLLM_QAIC_NUM_CORES)
5. String pass-through (VLLM_QAIC_COMPILER_ARGS, VLLM_QAIC_QPC_PATH, paths)
6. __dir__ exposes all expected keys
7. __getattr__ raises AttributeError for unknown names

Note: VLLM_TORCH_QAIC_BASE_PATH does not exist in this env's vllm_qaic.envs
(it's a downstream-fork-only variable) — its coverage lives in
tests/_deprecated/test_envs_downstream.py, not here. See that file's docstring.
"""

import pytest

import vllm_qaic.envs as envs_module


# ===========================================================================
# 1. Default values when env vars are not set
# ===========================================================================

class TestDefaultValues:
    def test_compiler_args_default_is_none(self):
        assert envs_module.VLLM_QAIC_COMPILER_ARGS is None

    def test_dfs_en_default_is_none(self):
        # maybe_convert_bool(None) returns None when env var is unset
        assert envs_module.VLLM_QAIC_DFS_EN is None

    def test_max_cpu_threads_default_is_none(self):
        assert envs_module.VLLM_QAIC_MAX_CPU_THREADS is None

    def test_mos_default_is_none(self):
        assert envs_module.VLLM_QAIC_MOS is None

    def test_num_cores_default_is_none(self):
        assert envs_module.VLLM_QAIC_NUM_CORES is None

    def test_qpc_path_default_is_none(self):
        assert envs_module.VLLM_QAIC_QPC_PATH is None

    def test_torch_qaic_profiler_dir_default_is_none(self):
        assert envs_module.VLLM_TORCH_QAIC_PROFILER_DIR is None


# ===========================================================================
# 2. Correct values when env vars are set
# ===========================================================================

class TestSetValues:
    def test_compiler_args_set(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_COMPILER_ARGS", "--aic-hw --aic-num-cores=16")
        assert envs_module.VLLM_QAIC_COMPILER_ARGS == "--aic-hw --aic-num-cores=16"

    def test_qpc_path_set(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_QPC_PATH", "/opt/qpc/my_model")
        assert envs_module.VLLM_QAIC_QPC_PATH == "/opt/qpc/my_model"

    def test_torch_qaic_profiler_dir_set(self, monkeypatch):
        monkeypatch.setenv("VLLM_TORCH_QAIC_PROFILER_DIR", "/tmp/profiler")
        assert envs_module.VLLM_TORCH_QAIC_PROFILER_DIR == "/tmp/profiler"

    def test_compiler_args_empty_string(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_COMPILER_ARGS", "")
        assert envs_module.VLLM_QAIC_COMPILER_ARGS == ""


# ===========================================================================
# 3. Boolean conversion (VLLM_QAIC_DFS_EN)
# ===========================================================================

class TestBooleanConversion:
    @pytest.mark.parametrize("val,expected", [
        ("1", True),
        ("0", False),
        # Note: maybe_convert_bool() in vllm only accepts "0"/"1", not "true"/"false"
    ])
    def test_dfs_en_bool_conversion(self, monkeypatch, val, expected):
        monkeypatch.setenv("VLLM_QAIC_DFS_EN", val)
        result = envs_module.VLLM_QAIC_DFS_EN
        assert result == expected, (
            f"VLLM_QAIC_DFS_EN={val!r} → expected {expected}, got {result}"
        )


# ===========================================================================
# 4. Integer conversion
# ===========================================================================

class TestIntegerConversion:
    def test_max_cpu_threads_int(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_MAX_CPU_THREADS", "32")
        assert envs_module.VLLM_QAIC_MAX_CPU_THREADS == 32

    def test_mos_int(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_MOS", "4")
        assert envs_module.VLLM_QAIC_MOS == 4

    def test_num_cores_int(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_NUM_CORES", "16")
        assert envs_module.VLLM_QAIC_NUM_CORES == 16

    def test_max_cpu_threads_single_digit(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_MAX_CPU_THREADS", "1")
        assert envs_module.VLLM_QAIC_MAX_CPU_THREADS == 1

    def test_mos_zero(self, monkeypatch):
        monkeypatch.setenv("VLLM_QAIC_MOS", "0")
        assert envs_module.VLLM_QAIC_MOS == 0


# ===========================================================================
# 5. String pass-through
# ===========================================================================

class TestStringPassthrough:
    def test_compiler_args_with_spaces(self, monkeypatch):
        val = "--aic-hw --aic-num-cores=16 --aic-mos=2"
        monkeypatch.setenv("VLLM_QAIC_COMPILER_ARGS", val)
        assert envs_module.VLLM_QAIC_COMPILER_ARGS == val

    def test_qpc_path_with_subdirs(self, monkeypatch):
        path = "/opt/qpc/models/TinyLlama/seq128_ctx256_bs4"
        monkeypatch.setenv("VLLM_QAIC_QPC_PATH", path)
        assert envs_module.VLLM_QAIC_QPC_PATH == path


# ===========================================================================
# 6. __dir__ exposes all expected keys
# ===========================================================================

class TestDir:
    EXPECTED_QAIC_VARS = {
        "VLLM_QAIC_COMPILER_ARGS",
        "VLLM_QAIC_DFS_EN",
        "VLLM_QAIC_MAX_CPU_THREADS",
        "VLLM_QAIC_MOS",
        "VLLM_QAIC_NUM_CORES",
        "VLLM_QAIC_QPC_PATH",
        "VLLM_TORCH_QAIC_PROFILER_DIR",
    }

    def test_dir_contains_all_qaic_vars(self):
        exposed = set(dir(envs_module))
        missing = self.EXPECTED_QAIC_VARS - exposed
        assert not missing, f"Missing from __dir__: {missing}"

    def test_dir_returns_list(self):
        assert isinstance(dir(envs_module), list)


# ===========================================================================
# 7. __getattr__ raises AttributeError for unknown names
# ===========================================================================

class TestGetAttr:
    def test_unknown_var_raises_attribute_error(self):
        with pytest.raises(AttributeError, match="VLLM_QAIC_NONEXISTENT"):
            _ = envs_module.VLLM_QAIC_NONEXISTENT

    def test_known_var_does_not_raise(self):
        try:
            _ = envs_module.VLLM_QAIC_COMPILER_ARGS
        except AttributeError:
            pytest.fail("VLLM_QAIC_COMPILER_ARGS raised AttributeError unexpectedly")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
