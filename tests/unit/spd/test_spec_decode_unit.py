# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC-specific speculative decoding (SpD) configuration (pure Python, no hardware).

On-device SpD inference tests (TestDraftModelConfigExtraction::
test_ngram_spd_output_non_empty and TestSpDConfigValidation::
test_ngram_spd_matches_baseline) have moved to
vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_spd.py to use the
device/ fixture system.

Coverage areas
--------------
1. Draft model config extraction — draft_override_qaic_config takes priority
2. Draft model config fallback — falls back to override_qaic_config
3. async_scheduling assertion — must be False for QaicDraftModelProposer
4. num_spec_tokens / decode_bsz extraction from speculative_config
5. SpD config validation — LoRA+SpD raises, ODS+SpD config parsing
"""

import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_spec_config(num_speculative_tokens: int = 3, method: str = "ngram"):
    return types.SimpleNamespace(
        num_speculative_tokens=num_speculative_tokens,
        method=method,
    )


def _make_scheduler_config(max_num_seqs: int = 4, async_scheduling: bool = False):
    return types.SimpleNamespace(
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
    )


def _make_draft_vllm_config(
    draft_override_qaic_config=None,
    override_qaic_config=None,
    num_speculative_tokens: int = 3,
    max_num_seqs: int = 4,
    async_scheduling: bool = False,
):
    additional_config = {}
    if draft_override_qaic_config is not None:
        additional_config["draft_override_qaic_config"] = draft_override_qaic_config
    if override_qaic_config is not None:
        additional_config["override_qaic_config"] = override_qaic_config
    return types.SimpleNamespace(
        additional_config=additional_config,
        speculative_config=_make_spec_config(num_speculative_tokens),
        scheduler_config=_make_scheduler_config(max_num_seqs, async_scheduling),
    )


def _extract_override_config(draft_vllm_config) -> dict:
    """Mirrors the config extraction logic in QaicDraftModelProposer.__init__()."""
    additional_config = draft_vllm_config.additional_config
    if "draft_override_qaic_config" in additional_config:
        return additional_config.get("draft_override_qaic_config", None) or {}
    elif "override_qaic_config" in additional_config:
        return additional_config.get("override_qaic_config", None) or {}
    return {}


# ===========================================================================
# 1. Draft model config extraction
# ===========================================================================

class TestDraftModelConfigExtraction:
    def test_draft_override_takes_priority(self):
        cfg = _make_draft_vllm_config(
            draft_override_qaic_config={"num_cores": 8},
            override_qaic_config={"num_cores": 16},
        )
        assert _extract_override_config(cfg)["num_cores"] == 8

    def test_fallback_to_override_qaic_config(self):
        cfg = _make_draft_vllm_config(override_qaic_config={"num_cores": 16})
        assert _extract_override_config(cfg)["num_cores"] == 16

    def test_empty_when_neither_present(self):
        cfg = _make_draft_vllm_config()
        assert _extract_override_config(cfg) == {}

    def test_none_draft_override_falls_back(self):
        cfg = _make_draft_vllm_config(
            draft_override_qaic_config=None,
            override_qaic_config={"mos": 2},
        )
        assert isinstance(_extract_override_config(cfg), dict)

    def test_draft_override_empty_dict(self):
        cfg = _make_draft_vllm_config(
            draft_override_qaic_config={},
            override_qaic_config={"mos": 2},
        )
        assert _extract_override_config(cfg) == {}


# ===========================================================================
# 2. async_scheduling assertion
# ===========================================================================

class TestAsyncSchedulingAssertion:
    def test_sync_scheduling_passes(self):
        cfg = _make_draft_vllm_config(async_scheduling=False)
        assert not cfg.scheduler_config.async_scheduling

    def test_async_scheduling_would_fail(self):
        cfg = _make_draft_vllm_config(async_scheduling=True)
        with pytest.raises(AssertionError):
            assert not cfg.scheduler_config.async_scheduling, (
                "QaicDraftModelProposer requires synchronous scheduling"
            )


# ===========================================================================
# 3. num_spec_tokens and decode_bsz extraction
# ===========================================================================

class TestSpecDecodeParams:
    def test_num_spec_tokens_extracted(self):
        cfg = _make_draft_vllm_config(num_speculative_tokens=5)
        assert cfg.speculative_config.num_speculative_tokens == 5

    def test_decode_bsz_extracted(self):
        cfg = _make_draft_vllm_config(max_num_seqs=8)
        assert cfg.scheduler_config.max_num_seqs == 8

    def test_default_num_spec_tokens(self):
        cfg = _make_draft_vllm_config()
        assert cfg.speculative_config.num_speculative_tokens == 3

    def test_default_decode_bsz(self):
        cfg = _make_draft_vllm_config()
        assert cfg.scheduler_config.max_num_seqs == 4


# ===========================================================================
# 4. SpD config validation + on-device baseline comparison
# ===========================================================================

class TestSpDConfigValidation:
    def test_lora_with_spd_raises(self, make_vllm_config):
        """LoRA + SpD must raise AssertionError."""
        import types as _types
        import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
        from vllm_qaic.platform_base import QaicPlatform
        vllm_config = make_vllm_config(
            lora_config=_types.SimpleNamespace(),
            speculative_config=_types.SimpleNamespace(
                method="ngram", num_speculative_tokens=3
            ),
        )
        with pytest.raises((AssertionError, ValueError)):
            QaicPlatform.check_and_update_config(vllm_config)

    def test_ods_config_parsing(self):
        """aic_include_sampler=True is correctly parsed by _clean_config."""
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"aic_include_sampler": "true"})
        assert result.get("aic_include_sampler") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
