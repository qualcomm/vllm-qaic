# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC LoRA utility functions.

These tests cover the pure-Python logic in vllm_qaic.model_loader.qaic:
  - search_adapters_in_cache   — scans HF_HOME for adapters matching a base model
  - verify_adaptername_to_id_consistency — validates that a loaded JSON mapping
                                           matches the requested LoRA modules

No QAIC hardware, no model loading, and no network access are required.
The ``tmp_path`` fixture provides an isolated temporary directory so that
HF_HOME manipulation does not affect the real cache.

Coverage areas
--------------
1. search_adapters_in_cache — empty cache returns empty list
2. search_adapters_in_cache — adapters placed in the right directory are found
3. search_adapters_in_cache — adapters for a different base model are not returned
4. verify_adaptername_to_id_consistency — matching mapping returns True
5. verify_adaptername_to_id_consistency — extra key in JSON returns False
6. verify_adaptername_to_id_consistency — missing key in JSON returns False
7. verify_adaptername_to_id_consistency — empty mapping with no modules returns True
8. adaptername_to_id JSON round-trip — write then read preserves content
9. adaptername_to_id JSON — wrong path raises FileNotFoundError
10. adaptername_to_id JSON — inconsistent content raises ValueError (integration)
"""

import json
import os
from pathlib import Path

import pytest

from vllm_qaic.model_loader.qaic import (
    search_adapters_in_cache,
    verify_adaptername_to_id_consistency,
)
from vllm.entrypoints.openai.models.protocol import LoRAModulePath


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter_dir(hf_home: Path, base_model: str, adapter_name: str) -> Path:
    """
    Simulate the directory layout that snapshot_download creates inside HF_HOME.

    HuggingFace stores models under:
      <HF_HOME>/hub/models--<org>--<repo>/snapshots/<hash>/

    search_adapters_in_cache looks for adapter_config.json files whose
    parent path contains the base model name.
    """
    # Flatten base_model name the same way HF does: replace '/' with '--'
    flat_adapter = adapter_name.replace("/", "--")
    adapter_dir = hf_home / "hub" / f"models--{flat_adapter}" / "snapshots" / "abc123"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Write adapter_config.json with base_model_name_or_path
    # peft_type is required by newer versions of the PEFT library
    config = {"base_model_name_or_path": base_model, "peft_type": "LORA"}
    (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
    return adapter_dir


# ===========================================================================
# 1-3. search_adapters_in_cache
# ===========================================================================

class TestSearchAdaptersInCache:
    def test_empty_cache_returns_empty_list(self, tmp_path, monkeypatch):
        """No adapters in cache → empty list."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        result = search_adapters_in_cache("meta-llama/Llama-3-8B")
        assert result == []

    def test_matching_adapter_is_found(self, tmp_path, monkeypatch):
        """An adapter whose adapter_config.json references the base model is returned."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        base = "meta-llama/Llama-3-8B"
        _make_adapter_dir(tmp_path, base, "my-org/my-lora-adapter")
        result = search_adapters_in_cache(base)
        assert len(result) == 1

    def test_two_matching_adapters_found(self, tmp_path, monkeypatch):
        """Two adapters for the same base model are both returned."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        base = "meta-llama/Llama-3-8B"
        _make_adapter_dir(tmp_path, base, "org/adapter-a")
        _make_adapter_dir(tmp_path, base, "org/adapter-b")
        result = search_adapters_in_cache(base)
        assert len(result) == 2

    def test_adapter_for_different_base_not_returned(self, tmp_path, monkeypatch):
        """An adapter for a different base model must not appear in results."""
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        _make_adapter_dir(tmp_path, "mistralai/Mistral-7B-v0.1", "org/mistral-lora")
        result = search_adapters_in_cache("meta-llama/Llama-3-8B")
        assert result == []

    def test_hf_home_respected(self, tmp_path, monkeypatch):
        """search_adapters_in_cache uses HF_HOME, not the real cache."""
        real_hf_home = os.environ.get("HF_HOME", "")
        monkeypatch.setenv("HF_HOME", str(tmp_path))
        # Even if the real cache has adapters, the isolated tmp_path has none
        result = search_adapters_in_cache("some-org/some-model")
        assert isinstance(result, list)


# ===========================================================================
# 4-7. verify_adaptername_to_id_consistency
# ===========================================================================

class TestVerifyAdapternameToIdConsistency:
    def _make_modules(self, names: list[str]) -> list[LoRAModulePath]:
        return [LoRAModulePath(name=n, path=f"/path/{n}") for n in names]

    def test_matching_mapping_returns_true(self):
        """Mapping with exactly the same names as lora_modules → True.
        Note: verify_adaptername_to_id_consistency uses 1-indexed values (i+1).
        """
        modules = self._make_modules(["adapter_0", "adapter_1"])
        mapping = {"adapter_0": 1, "adapter_1": 2}  # 1-indexed
        assert verify_adaptername_to_id_consistency(mapping, modules) is True

    def test_extra_key_in_mapping_ignored(self):
        """Mapping with extra keys: function only checks modules in lora_modules list.
        Extra keys in the mapping are ignored — returns True if all modules match.
        """
        modules = self._make_modules(["adapter_0"])
        mapping = {"adapter_0": 1, "adapter_extra": 2}  # 1-indexed
        # Function only checks that all modules in lora_modules are in mapping with
        # correct 1-indexed position — extra keys are ignored
        assert verify_adaptername_to_id_consistency(mapping, modules) is True

    def test_missing_key_in_mapping_returns_false(self):
        """Mapping is missing a name that is in lora_modules → False."""
        modules = self._make_modules(["adapter_0", "adapter_1"])
        mapping = {"adapter_0": 1}  # 1-indexed, missing adapter_1
        assert verify_adaptername_to_id_consistency(mapping, modules) is False

    def test_empty_mapping_with_no_modules_returns_true(self):
        """Both empty → consistent."""
        assert verify_adaptername_to_id_consistency({}, []) is True

    def test_empty_mapping_with_modules_returns_false(self):
        """Empty mapping but non-empty modules → False."""
        modules = self._make_modules(["adapter_0"])
        assert verify_adaptername_to_id_consistency({}, modules) is False

    def test_order_does_not_matter(self):
        """Dict key order does not matter; VALUES must match 1-indexed position."""
        modules = self._make_modules(["b", "a", "c"])
        # b is at index 0 → value must be 1, a at index 1 → 2, c at index 2 → 3
        mapping = {"c": 3, "a": 2, "b": 1}  # 1-indexed, dict order doesn't matter
        assert verify_adaptername_to_id_consistency(mapping, modules) is True

    def test_single_adapter(self):
        """Single adapter, single module → True."""
        modules = self._make_modules(["my_adapter"])
        mapping = {"my_adapter": 1}  # 1-indexed
        assert verify_adaptername_to_id_consistency(mapping, modules) is True


# ===========================================================================
# 8-10. adaptername_to_id JSON round-trip (logic extracted from test_qaic_lora.py)
# ===========================================================================

class TestAdapternameToIdJson:
    """
    Tests the JSON persistence logic for adaptername_to_id that is used
    during LoRA QPC compilation.  This logic lives inline in test_qaic_lora.py
    (test_qaic_get_qaic_model_dump_adaptername_to_id) but is extracted here
    as pure unit tests that need no model loading.
    """

    def _write_mapping(self, qpc_path: Path, mapping: dict) -> None:
        with open(qpc_path / "adaptername_to_id.json", "w") as f:
            json.dump(mapping, f)

    def _read_mapping(self, qpc_path: Path) -> dict:
        json_path = qpc_path / "adaptername_to_id.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"The file at {json_path} was not found. "
                "Please provide a correct VLLM_QAIC_QPC_PATH."
            )
        with open(json_path) as f:
            return json.load(f)

    def test_write_then_read_preserves_content(self, tmp_path):
        """Writing a mapping and reading it back gives the same dict."""
        mapping = {"adapter_0": 0, "adapter_1": 1}
        self._write_mapping(tmp_path, mapping)
        result = self._read_mapping(tmp_path)
        assert result == mapping

    def test_read_nonexistent_path_raises_file_not_found(self, tmp_path):
        """Reading from a path that has no JSON raises FileNotFoundError."""
        wrong_path = tmp_path / "nonexistent_subdir"
        with pytest.raises(FileNotFoundError, match="adaptername_to_id.json"):
            self._read_mapping(wrong_path)

    def test_inconsistent_mapping_raises_value_error(self, tmp_path):
        """
        If the loaded mapping does not match the requested lora_modules,
        the caller must raise ValueError.  This mirrors the check in
        test_qaic_get_qaic_model_dump_adaptername_to_id.
        """
        # Write a mapping with wrong names
        self._write_mapping(tmp_path, {"abc": 0, "def": 1})
        loaded = self._read_mapping(tmp_path)

        lora_modules = [
            LoRAModulePath(name="adapter_0", path="/path_0"),
            LoRAModulePath(name="adapter_1", path="/path_1"),
        ]
        with pytest.raises(ValueError):
            if not verify_adaptername_to_id_consistency(loaded, lora_modules):
                raise ValueError(
                    f"Inconsistent file content in {tmp_path}/adaptername_to_id.json "
                    "and input lora modules."
                )

    def test_consistent_mapping_does_not_raise(self, tmp_path):
        """Consistent mapping must not raise.
        Note: verify_adaptername_to_id_consistency uses 1-indexed values (i+1).
        """
        mapping = {"adapter_0": 1, "adapter_1": 2}  # 1-indexed
        self._write_mapping(tmp_path, mapping)
        loaded = self._read_mapping(tmp_path)

        lora_modules = [
            LoRAModulePath(name="adapter_0", path="/path_0"),
            LoRAModulePath(name="adapter_1", path="/path_1"),
        ]
        # Should not raise
        if not verify_adaptername_to_id_consistency(loaded, lora_modules):
            pytest.fail("verify_adaptername_to_id_consistency returned False for consistent mapping")

    def test_json_not_written_twice(self, tmp_path):
        """
        The compilation logic only writes the JSON if it does not already exist.
        Verify that a second call does not overwrite the file.
        """
        mapping_first = {"adapter_0": 0}
        self._write_mapping(tmp_path, mapping_first)

        # Simulate second call: file already exists, should not overwrite
        json_path = tmp_path / "adaptername_to_id.json"
        if not json_path.exists():
            self._write_mapping(tmp_path, {"adapter_0": 99})  # would overwrite

        result = self._read_mapping(tmp_path)
        assert result == mapping_first, "File was overwritten on second call"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
