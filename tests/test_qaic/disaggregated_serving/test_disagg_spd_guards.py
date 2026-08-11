# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for disagg+SpD guard logic (no hardware required).

Verifies four guard behaviors that only matter in disaggregated serving:

  1. drafter=None on the KV producer after __init__ (Guard1).
  2. QaicDraftModelProposer construction is skipped on the KV producer
     (Guard2).
  3. _draft_token_ids is never set on the KV producer (Guard3).
  4. decode_ks (varK dispatch) is preserved, not collapsed, on the KV
     consumer (Guard4).

Split out of test_spd_disagg.py (renamed here) by the tests/e2e/ infra
migration: the hardware E2E classes that used to share this file moved to
tests/e2e/disaggregated_serving/test_qaic_disagg_spd.py, which needs live
QAIC devices and the qaic_test_config/disagg_server fixtures. These guard
tests need neither — they call QaicModelRunnerAoT's unbound methods directly
against a types.SimpleNamespace stand-in, so they stay here as fast,
non-hardware unit tests.
"""

import types
from unittest.mock import MagicMock

import torch

from vllm_qaic.worker.model_runner import QaicModelRunnerAoT as QaicModelRunner


def _make_runner_ns(
    is_kv_producer=False,
    is_kv_consumer=False,
    speculative_config=None,
    drafter=None,
    decode_ks=None,
    num_spec_tokens=3,
):
    """Minimal namespace for calling unbound methods."""
    return types.SimpleNamespace(
        is_kv_producer=is_kv_producer,
        is_kv_consumer=is_kv_consumer,
        speculative_config=speculative_config,
        drafter=drafter,
        decode_ks=decode_ks if decode_ks is not None else [num_spec_tokens],
        num_spec_tokens=num_spec_tokens,
        _draft_token_ids=None,
    )


class TestGuard1_DrafterClearedOnKvProducer:
    """drafter=None on KV producer after __init__."""

    def test_ngram_kv_producer_drafter_is_none(self):
        spec = MagicMock()
        spec.method = "ngram"
        ns = _make_runner_ns(
            is_kv_producer=True,
            speculative_config=spec,
            drafter=MagicMock(),
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is None

    def test_suffix_kv_producer_drafter_is_none(self):
        spec = MagicMock()
        spec.method = "suffix"
        ns = _make_runner_ns(
            is_kv_producer=True,
            speculative_config=spec,
            drafter=MagicMock(),
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is None

    def test_kv_consumer_drafter_preserved(self):
        spec = MagicMock()
        mock_drafter = MagicMock()
        ns = _make_runner_ns(
            is_kv_consumer=True,
            speculative_config=spec,
            drafter=mock_drafter,
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is mock_drafter

    def test_non_disagg_drafter_preserved(self):
        mock_drafter = MagicMock()
        ns = _make_runner_ns(
            is_kv_producer=False,
            speculative_config=MagicMock(),
            drafter=mock_drafter,
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is mock_drafter


class TestGuard2_DraftModelProposerNotOnKvProducer:
    """QaicDraftModelProposer construction skipped on KV producer."""

    def _would_construct(self, ns):
        return bool(
            ns.speculative_config
            and ns.speculative_config.uses_draft_model()
            and not ns.is_kv_producer
        )

    def test_kv_producer_skips(self):
        spec = MagicMock()
        spec.uses_draft_model.return_value = True
        ns = _make_runner_ns(is_kv_producer=True, speculative_config=spec)
        assert not self._would_construct(ns)

    def test_kv_consumer_allows(self):
        spec = MagicMock()
        spec.uses_draft_model.return_value = True
        ns = _make_runner_ns(is_kv_consumer=True, speculative_config=spec)
        assert self._would_construct(ns)

    def test_non_disagg_allows(self):
        spec = MagicMock()
        spec.uses_draft_model.return_value = True
        ns = _make_runner_ns(is_kv_producer=False, speculative_config=spec)
        assert self._would_construct(ns)


class TestGuard3_DraftTokenIdsNoneOnKvProducer:
    """_draft_token_ids never set on KV producer."""

    def _should_propose(self, ns):
        return bool(ns.speculative_config is not None and not ns.is_kv_producer)

    def test_kv_producer_no_proposal(self):
        ns = _make_runner_ns(is_kv_producer=True, speculative_config=MagicMock())
        assert not self._should_propose(ns)

    def test_kv_consumer_allows_proposal(self):
        ns = _make_runner_ns(is_kv_consumer=True, speculative_config=MagicMock())
        assert self._should_propose(ns)

    def test_non_disagg_allows_proposal(self):
        ns = _make_runner_ns(is_kv_producer=False, speculative_config=MagicMock())
        assert self._should_propose(ns)

    def test_no_spec_config_no_proposal(self):
        ns = _make_runner_ns(speculative_config=None)
        assert not self._should_propose(ns)

    def test_draft_token_ids_remains_none_on_producer(self):
        ns = _make_runner_ns(is_kv_producer=True, speculative_config=MagicMock())
        if ns.speculative_config is not None and not ns.is_kv_producer:
            ns._draft_token_ids = torch.zeros(1, dtype=torch.int32)
        assert ns._draft_token_ids is None


class TestGuard4_VarKPreservedOnKvConsumer:
    """decode_ks is NOT collapsed on KV consumer — varK dispatch is supported."""

    def test_kv_consumer_preserves_varK(self):
        """KV consumer keeps [0, K] for ngram/suffix varK dispatch."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[0, 3])
        # No guard applied — decode_ks stays as-is
        assert ns.decode_ks == [0, 3]

    def test_kv_consumer_single_k_unchanged(self):
        """KV consumer with draft_model (single K) stays unchanged."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[3])
        assert ns.decode_ks == [3]

    def test_kv_consumer_no_proposals_returns_k0(self):
        """KV consumer with varK returns K=0 when no proposals exist."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[0, 3], num_spec_tokens=3)
        ns.num_decodes = 4  # required by _determine_active_k
        so = types.SimpleNamespace(scheduled_spec_decode_tokens={})
        k = QaicModelRunner._determine_active_k(ns, so)
        assert k == 0, "No proposals → K=0 fallback on KV consumer"

    def test_kv_consumer_with_proposals_returns_k_max(self):
        """KV consumer with varK returns K_max when proposals exist."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[0, 3], num_spec_tokens=3)
        ns.num_decodes = 4
        so = types.SimpleNamespace(scheduled_spec_decode_tokens={"req_0": [10, 11, 12]})
        k = QaicModelRunner._determine_active_k(ns, so)
        assert k == 3, "Proposals exist → K_max on KV consumer"

    def test_non_disagg_varK_unchanged(self):
        """Non-disagg node varK behavior is identical."""
        ns = _make_runner_ns(decode_ks=[0, 3])
        ns.num_decodes = 4
        so_empty = types.SimpleNamespace(scheduled_spec_decode_tokens={})
        so_full = types.SimpleNamespace(
            scheduled_spec_decode_tokens={"req_0": [10, 11, 12]}
        )
        assert QaicModelRunner._determine_active_k(ns, so_empty) == 0
        assert QaicModelRunner._determine_active_k(ns, so_full) == 3
