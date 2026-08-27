# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC disaggregated serving config-validation constraints.

Pure Python, no hardware: verifies that check_and_update_config() enforces
the correct constraints for disaggregated prefill/decode serving.

NOTE: despite the historical filename, this file contains no on-device
integration tests — TestDisaggregatedConstraints is pure Python, and
TestDisaggregatedSchedulingPolicies skips via `importorskip` on the real
top-level `qaic_disagg` package (not `vllm_qaic.qaic_disagg`, which does
not exist anywhere in this repo — vllm_qaic has no qaic_disagg submodule).
See disaggregated/test_policy.py for further (also pure-Python,
no-hardware) scheduling-policy coverage of the same two classes.

Usage:
  pytest disaggregated/test_disaggregated_integration.py -v
"""

import types

import pytest
import torch

import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
from vllm_qaic.platform_base import QaicPlatform


class TestDisaggregatedConstraints:
    """Verify disaggregated serving constraints are enforced."""

    def test_lora_with_disagg_raises(self, make_vllm_config):
        lora_config = types.SimpleNamespace()
        kv_cfg = types.SimpleNamespace(kv_role="kv_both")
        cfg = make_vllm_config(lora_config=lora_config, kv_transfer_config=kv_cfg)
        with pytest.raises(AssertionError, match="LORA with Disaggregated"):
            QaicPlatform.check_and_update_config(cfg)

    def test_unsupported_spd_type_raises(self, make_vllm_config):
        spec_config = types.SimpleNamespace(method="turbo", num_speculative_tokens=3)
        kv_cfg = types.SimpleNamespace(kv_role="kv_both")
        cfg = make_vllm_config(speculative_config=spec_config, kv_transfer_config=kv_cfg)
        with pytest.raises(AssertionError):
            QaicPlatform.check_and_update_config(cfg)

    def test_ngram_spd_allowed(self, make_vllm_config):
        spec_config = types.SimpleNamespace(method="ngram", num_speculative_tokens=3)
        kv_cfg = types.SimpleNamespace(kv_role="kv_consumer")
        cfg = make_vllm_config(speculative_config=spec_config, kv_transfer_config=kv_cfg)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "SPD Types" in str(e):
                pytest.fail(f"ngram SpD should be allowed with disagg: {e}")

    def test_draft_model_spd_allowed(self, make_vllm_config):
        spec_config = types.SimpleNamespace(method="draft_model", num_speculative_tokens=3)
        kv_cfg = types.SimpleNamespace(kv_role="kv_consumer")
        cfg = make_vllm_config(speculative_config=spec_config, kv_transfer_config=kv_cfg)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "SPD Types" in str(e):
                pytest.fail(f"draft_model SpD should be allowed with disagg: {e}")

    def test_prefix_caching_kv_consumer_disabled(self, make_vllm_config):
        """Prefix caching + kv_consumer: check_and_update_config() never raises
        here — it silently disables prefix caching first (verified empirically;
        see generic/test_platform.py::TestDisaggregatedServingConstraints for
        the same behaviour with an explicit no-exception assertion).
        """
        kv_cfg = types.SimpleNamespace(kv_role="kv_consumer")
        cfg = make_vllm_config(enable_prefix_caching=True, kv_transfer_config=kv_cfg)
        QaicPlatform.check_and_update_config(cfg)
        assert not cfg.cache_config.enable_prefix_caching

    def test_prefix_caching_kv_both_disabled(self, make_vllm_config):
        """Prefix caching + kv_both: same auto-disable behaviour as kv_consumer."""
        kv_cfg = types.SimpleNamespace(kv_role="kv_both")
        cfg = make_vllm_config(enable_prefix_caching=True, kv_transfer_config=kv_cfg)
        QaicPlatform.check_and_update_config(cfg)
        assert not cfg.cache_config.enable_prefix_caching

    def test_prefix_caching_kv_producer_allowed(self, make_vllm_config, patch_qaic_executor_import_bug):
        """Prefix caching + kv_producer must NOT raise on the prefix caching check.

        kv_role="kv_producer" also runs the "stages" pipeline-parallel check
        (requires 'stages' in override_qaic_config to avoid a TypeError in
        platform_base.py's `int(override_qaic_config.get("stages"))`) and
        imports vllm_qaic.executor.qaic_uniproc_executor, a subpackage that
        does not exist (real module: vllm_qaic.qaic_uniproc_executor). Both
        are source bugs we are not fixing here; "stages" is supplied
        explicitly and the patch_qaic_executor_import_bug fixture stubs the
        broken import path so this test can reach the prefix-caching check.
        """
        kv_cfg = types.SimpleNamespace(kv_role="kv_producer")
        cfg = make_vllm_config(enable_prefix_caching=True, kv_transfer_config=kv_cfg)
        cfg.additional_config["override_qaic_config"]["stages"] = "1"
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "Prefix caching" in str(e):
                pytest.fail(f"kv_producer should allow prefix caching: {e}")

    def test_no_prefix_caching_kv_consumer_allowed(self, make_vllm_config):
        kv_cfg = types.SimpleNamespace(kv_role="kv_consumer")
        cfg = make_vllm_config(enable_prefix_caching=False, kv_transfer_config=kv_cfg)
        try:
            QaicPlatform.check_and_update_config(cfg)
        except AssertionError as e:
            if "Prefix caching" in str(e):
                pytest.fail(f"No prefix caching with kv_consumer should be allowed: {e}")


class TestDisaggregatedSchedulingPolicies:
    """Verify scheduling policies work correctly.

    Note: requires the real qaic_disagg package to be installed (imported
    as top-level `qaic_disagg`, not `vllm_qaic.qaic_disagg` — see the file
    docstring). Tests skip gracefully if the package is not available.

    Constructors take no arguments and `schedule()` takes an
    itertools.cycle plus the instance list (see
    qaic_disagg.proxy.server.SchedulingPolicy) — there is no `get_server()`
    method and no instances-in-constructor form.
    """

    def test_round_robin_cycles(self):
        pytest.importorskip(
            "qaic_disagg",
            reason="qaic_disagg not installed",
        )
        import itertools

        from qaic_disagg.proxy.server import RoundRobinSchedulingPolicy
        policy = RoundRobinSchedulingPolicy()
        instances = ["server1", "server2", "server3"]
        cycler = itertools.cycle(instances)
        results = [policy.schedule(cycler) for _ in range(6)]
        assert results == ["server1", "server2", "server3", "server1", "server2", "server3"]

    def test_least_outstanding_initial_distribution(self):
        pytest.importorskip(
            "qaic_disagg",
            reason="qaic_disagg not installed",
        )
        from qaic_disagg.proxy.server import LeastOutstandingSchedulingPolicy
        policy = LeastOutstandingSchedulingPolicy()
        instances = ["s1", "s2", "s3"]
        first = policy.schedule(None, instances)
        assert first in instances


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
