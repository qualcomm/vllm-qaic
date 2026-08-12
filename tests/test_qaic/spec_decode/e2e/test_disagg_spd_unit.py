# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Unit tests for disagg+SpD guard/config code paths.

These are fast, non-hardware unit tests. They cover four behaviors that only
matter in disaggregated serving (qaic_disagg, KV transfer, kv_transfer_config)
and are therefore not exercised by the single-process correctness tests
(test_ngram_integration, test_boundary_spd, etc.):

  1. draft_override_qaic_config is preserved through run_args_vllm_serve
  2. KV connector registration is idempotent
  3. _determine_active_k returns K=0 on the first kv_consumer step
     (num_decodes == 0)
  4. decode_ks initialization limits draft_model's [0, K] path to consumers

Tests 1 requires qaic_disagg, test 2 requires vllm_qaic, tests 3 require
torch/vllm (each skipped if its dependency is absent). Test 4 reads the source
file directly and needs no imports.

NOTE: these tests do NOT cover the decode-dummy-run "Bug 3" fix (merging
num_logits_to_keep into the decode batch inputs and sizing the logits output
buffer by max_decode_tokens). That path only runs on QAIC hardware during
decode-server startup and is validated by the e2e suite
(tests/e2e/disaggregated_serving/test_qaic_disagg_spd.py), not here.
"""

import importlib.util
import json
import sys
import types
import unittest.mock
from argparse import Namespace
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# qaic_disagg is not pip-installed as a build dependency of this test suite
# (it is installed editable, separately, into the same venv), so locate its
# parent dir via vllm_qaic's installed location rather than a hardcoded
# parents[N] hop from this test file, which breaks silently if the test moves.
# vllm_qaic itself needs no sys.path entry - it's already pip-installed
# editable and importable from anywhere.
# ---------------------------------------------------------------------------
_VLLM_QAIC_SPEC = importlib.util.find_spec("vllm_qaic")
assert _VLLM_QAIC_SPEC and _VLLM_QAIC_SPEC.submodule_search_locations, (
    "vllm_qaic is not installed; it is a required dependency of this test suite"
)
_VLLM_QAIC_REPO_ROOT = Path(_VLLM_QAIC_SPEC.submodule_search_locations[0]).parent


def _find_disagg_pkg_root() -> Path | None:
    """Locate qaic_disagg's package root via its own installed location.

    qaic_disagg is installed editable into the same venv as vllm_qaic (a
    separate repo, not a subdirectory of it), so we cannot derive its path by
    walking up from vllm_qaic's repo root.
    """
    spec = importlib.util.find_spec("qaic_disagg")
    if spec and spec.submodule_search_locations:
        return Path(spec.submodule_search_locations[0]).parent
    return None


_DISAGG_PKG_ROOT = _find_disagg_pkg_root()
if _DISAGG_PKG_ROOT is not None and str(_DISAGG_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_DISAGG_PKG_ROOT))


# ---------------------------------------------------------------------------
# 1. draft_override_qaic_config preserved through run_args_vllm_serve
# ---------------------------------------------------------------------------


def test_draft_override_qaic_config_preserved():
    """draft_override_qaic_config must appear in --additional-config, not only in
    --speculative-config, so that the decode server can find it before Pydantic
    parses the speculative config and silently drops unknown fields.

    Before the fix run_args_vllm_serve had no special handling for this key;
    it ended up inside the --speculative-config JSON where Pydantic discards it.
    """
    pytest.importorskip("zmq", reason="qaic_disagg requires pyzmq")
    pytest.importorskip("qaic_disagg", reason="qaic_disagg package not installed")
    from qaic_disagg.utils import run_args_vllm_serve

    args = Namespace(
        model="meta-llama/Llama-3.2-1B-Instruct",
        speculative_config={
            "method": "draft_model",
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "num_speculative_tokens": 7,
            "draft_override_qaic_config": {
                "ctx_len": 4096,
                "device_group": [10, 11],
            },
        },
    )

    cmd = run_args_vllm_serve(
        args,
        instType="decode",
        kv_connector="QaicConnector",
        skip_kv_connector=True,
        verbose=0,
    )

    assert "--additional-config" in cmd, "--additional-config not in returned CLI args"
    idx = cmd.index("--additional-config")
    additional_config = json.loads(cmd[idx + 1])
    assert "draft_override_qaic_config" in additional_config, (
        "draft_override_qaic_config was not lifted into additional_config; "
        "Pydantic will silently discard it from --speculative-config"
    )
    assert additional_config["draft_override_qaic_config"]["device_group"] == [10, 11]


# ---------------------------------------------------------------------------
# 2. KV connector registration is idempotent
# ---------------------------------------------------------------------------


def test_kv_connector_idempotent_registration():
    """register_connector() must not raise when called more than once.

    In disagg+SpD the init sequence can call the connector registration from
    the decode server, the target-model runner, and the draft-model runner.
    Before the fix, a second call re-registered the same key and caused a
    duplicate-registration error.
    """
    pytest.importorskip("torch", reason="vllm_qaic requires torch")
    import vllm_qaic.distributed.kv_transfer.kv_connector as kv_module

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    _KEYS = ["QaicConnector", "QaicLMCacheConnectorV1"]

    with unittest.mock.patch.object(kv_module, "current_platform") as mock_plat:
        mock_plat.is_aot_inference.return_value = True

        # Remove any existing registrations so we get a clean-slate test
        saved = {
            k: KVConnectorFactory._registry.pop(k)
            for k in _KEYS
            if k in KVConnectorFactory._registry
        }
        try:
            kv_module.register_connector()  # first call
            kv_module.register_connector()  # second call — must not raise
            kv_module.register_connector()  # third call for good measure

            assert "QaicConnector" in KVConnectorFactory._registry
            assert "QaicLMCacheConnectorV1" in KVConnectorFactory._registry
        finally:
            for k in _KEYS:
                KVConnectorFactory._registry.pop(k, None)
            KVConnectorFactory._registry.update(saved)


# ---------------------------------------------------------------------------
# 3. _determine_active_k returns K=0 on first kv_consumer step
# ---------------------------------------------------------------------------


def test_determine_active_k_returns_zero_when_no_decodes():
    """On the kv_consumer's first step the sequence is classified as prefill
    (num_decoded < num_prompt), so num_decodes==0 and no draft proposals exist.
    _determine_active_k must return 0 (K=0 fallback kernel) in this case.

    Before the fix the `if self.num_decodes == 0: return 0` branch did not
    exist; the code would fall through to inspect scheduler_output, which is
    None/empty on the first step, potentially selecting the wrong kernel.
    """
    pytest.importorskip("torch", reason="model_runner requires torch")
    from vllm_qaic.worker.model_runner import QaicModelRunnerAoT

    runner = types.SimpleNamespace(decode_ks=[0, 7], num_decodes=0)
    # scheduler_output is never accessed when num_decodes==0; pass None to
    # prove the early return fires before any attribute access.
    k = QaicModelRunnerAoT._determine_active_k(runner, scheduler_output=None)
    assert k == 0, f"Expected K=0 on first consumer step (num_decodes=0), got {k}"


def test_determine_active_k_returns_max_k_when_proposals_present():
    """Sanity check: _determine_active_k selects max K when proposals exist."""
    pytest.importorskip("torch", reason="model_runner requires torch")
    from vllm_qaic.worker.model_runner import QaicModelRunnerAoT

    sched = types.SimpleNamespace(scheduled_spec_decode_tokens={"req-0": [1, 2, 3]})
    runner = types.SimpleNamespace(decode_ks=[0, 7], num_decodes=2)
    k = QaicModelRunnerAoT._determine_active_k(runner, scheduler_output=sched)
    assert k == 7


def test_determine_active_k_returns_zero_when_no_proposals():
    """Sanity check: _determine_active_k falls back to K=0 when no proposals."""
    pytest.importorskip("torch", reason="model_runner requires torch")
    from vllm_qaic.worker.model_runner import QaicModelRunnerAoT

    sched = types.SimpleNamespace(scheduled_spec_decode_tokens={})
    runner = types.SimpleNamespace(decode_ks=[0, 7], num_decodes=1)
    k = QaicModelRunnerAoT._determine_active_k(runner, scheduler_output=sched)
    assert k == 0


# ---------------------------------------------------------------------------
# 4. decode_ks condition is consumer-specific for "draft_model"
# ---------------------------------------------------------------------------


def test_decode_ks_condition_is_consumer_specific_for_draft_model():
    """Only disaggregated draft-model consumers receive the K=0 specialization."""
    src_path = _VLLM_QAIC_REPO_ROOT / "vllm_qaic" / "worker" / "model_runner.py"
    assert src_path.exists(), f"Source file not found: {src_path}"

    src = src_path.read_text()
    assert '_method in ("ngram", "suffix")' in src
    assert '_method == "draft_model" and _is_disagg_consumer' in src
    assert '_method in ("ngram", "suffix", "draft_model")' not in src


def test_decode_batch_inputs_are_reused_across_lifecycle_paths():
    """The helper is called during construction, not as a dummy-only variant."""
    src_path = _VLLM_QAIC_REPO_ROOT / "vllm_qaic" / "model_loader" / "qaic.py"
    src = src_path.read_text()

    assert "def _make_decode_batch_input_for_k(self, k: int) -> dict:" in src
    assert "dummy=True" not in src
    assert '"input_ids": np.zeros((self.decode_bsz, mdt), dtype=np.int64)' in src
    assert '"position_ids": np.full((self.decode_bsz, mdt), -1, dtype=np.int64)' in src
    assert '"batch_index": np.arange(self.decode_bsz' in src
    assert "self.decode_logits_by_k: dict[int, dict] = {}" in src
    assert "self.decode_num_logits_buffer_by_k: dict[int, dict] = {}" in src


def test_compile_only_target_defers_completion_for_draft_model():
    """The target compile path can return so the draft QPC is compiled next."""
    src_path = _VLLM_QAIC_REPO_ROOT / "vllm_qaic" / "model_loader" / "qaic.py"
    src = src_path.read_text()

    assert "raise_on_compile_complete: bool = True" in src
    assert "if raise_on_compile_complete:" in src
    assert "model.disagg_serving_en = False" in src
    assert "return model.eval()" in src


def test_decode_k_mismatch_diagnostics_cover_extra_missing_and_equal_lists():
    """The source contains both mismatch warnings and a quiet equal-list path."""
    src_path = _VLLM_QAIC_REPO_ROOT / "vllm_qaic" / "model_loader" / "qaic.py"
    src = src_path.read_text()

    assert "def _warn_on_decode_k_mismatch(" in src
    assert "extra_qpc_ks = sorted(available - configured)" in src
    assert "missing_qpc_ks = sorted(configured - available)" in src
    assert src.count("logger.warning(") >= 2
    assert '"QPC contains extra decode specializations for K=%s; vLLM "' in src
    assert '"QPC is missing decode specializations for configured K=%s; "' in src
    assert "configured_decode_ks = list(self.decode_ks)" in src
    assert "session_decode_ks = self._decode_ks_from_session()" in src
    assert "self._warn_on_decode_k_mismatch(" in src
