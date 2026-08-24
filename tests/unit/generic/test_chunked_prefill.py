# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC-specific chunked prefill (pure Python, no hardware).

On-device tests (long prompt produces output, mixed short/long batch,
chunked output matches non-chunked, async_scheduling matches sync) have
moved to vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_chunked_prefill.py
to use the device/ fixture system.

All prefill_seq_len / max_num_batched_tokens assertions below drive the real
QaicPlatform.check_and_update_config() (vllm/platforms/qaic_base.py, AOT
branch) via the shared make_vllm_config fixture, rather than a hand-copied
reimplementation of its formula — so a source-side change to that formula
would be caught here.

Coverage areas
--------------
1. prefill_seq_len = min(long_prefill_token_threshold, max_model_len)
2. prefill_seq_len = min(128, max_model_len) when threshold is 0 (disabled)
3. prefill_seq_len from override_qaic_config takes precedence
4. max_num_batched_tokens = min(max_num_seqs * prefill_seq_len, max_num_batched_tokens)
5. Chunked prefill disabled for pooling models (runner_type="pooling")
6. _clean_config prefill_seq_len pass-through
7. Eager mode: the AOT-only prefill_seq_len/max_num_batched_tokens formula
   is skipped entirely (driven via monkeypatched QaicPlatform.is_aot)
"""

import pytest


def _check_and_update(make_vllm_config, overrides=None, **kwargs):
    """Build a vllm_config via the shared fixture, apply any post-construction
    scheduler_config overrides, run it through the real
    QaicPlatform.check_and_update_config(), and return it for assertions."""
    import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
    from vllm_qaic.platform_base import QaicPlatform

    cfg = make_vllm_config(**kwargs)
    for key, value in (overrides or {}).items():
        setattr(cfg.scheduler_config, key, value)
    QaicPlatform.check_and_update_config(cfg)
    return cfg


# ===========================================================================
# 1. prefill_seq_len calculation
# ===========================================================================

class TestPrefillSeqLen:
    @pytest.mark.parametrize("threshold,max_len,expected", [
        (64, 2048, 64),
        (128, 2048, 128),
        (256, 2048, 256),
        (512, 2048, 512),
        (1024, 2048, 1024),
        (2048, 2048, 2048),
        (4096, 2048, 2048),
    ])
    def test_various_thresholds(self, make_vllm_config, threshold, max_len, expected):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=max_len,
            overrides={"long_prefill_token_threshold": threshold},
        )
        assert (
            cfg.additional_config["override_qaic_config"]["prefill_seq_len"]
            == expected
        )

    def test_threshold_zero_uses_128_default(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            overrides={"long_prefill_token_threshold": 0},
        )
        assert cfg.additional_config["override_qaic_config"]["prefill_seq_len"] == 128

    def test_threshold_zero_small_max_model_len(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=64,
            overrides={"long_prefill_token_threshold": 0},
        )
        assert cfg.additional_config["override_qaic_config"]["prefill_seq_len"] == 64

    def test_override_takes_precedence(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            additional_config={"override_qaic_config": {"prefill_seq_len": 256}},
            overrides={"long_prefill_token_threshold": 128},
        )
        assert cfg.additional_config["override_qaic_config"]["prefill_seq_len"] == 256

    def test_override_zero_not_used(self, make_vllm_config):
        """override_qaic_config["prefill_seq_len"]=0 is falsy, so the formula
        recomputes from long_prefill_token_threshold instead of using 0."""
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            additional_config={"override_qaic_config": {"prefill_seq_len": 0}},
            overrides={"long_prefill_token_threshold": 128},
        )
        assert cfg.additional_config["override_qaic_config"]["prefill_seq_len"] == 128


# ===========================================================================
# 2. max_num_batched_tokens calculation
# ===========================================================================

class TestMaxNumBatchedTokens:
    def test_budget_limited_by_seqs_times_prefill(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            max_num_seqs=4,
            overrides={
                "long_prefill_token_threshold": 128,
                "max_num_batched_tokens": 4096,
            },
        )
        assert cfg.scheduler_config.max_num_batched_tokens == 512

    def test_budget_limited_by_current_when_smaller(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            max_num_seqs=4,
            overrides={
                "long_prefill_token_threshold": 512,
                "max_num_batched_tokens": 256,
            },
        )
        assert cfg.scheduler_config.max_num_batched_tokens == 256

    def test_budget_single_seq(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            max_num_seqs=1,
            overrides={
                "long_prefill_token_threshold": 128,
                "max_num_batched_tokens": 4096,
            },
        )
        assert cfg.scheduler_config.max_num_batched_tokens == 128

    def test_budget_exact_match(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=2048,
            max_num_seqs=4,
            overrides={
                "long_prefill_token_threshold": 128,
                "max_num_batched_tokens": 512,
            },
        )
        assert cfg.scheduler_config.max_num_batched_tokens == 512


# ===========================================================================
# 3. Chunked prefill disabled for pooling models
# ===========================================================================

class TestChunkedPrefillPoolingModels:
    """QAIC pooling QPCs are compiled with seq_len == max_model_len: for
    runner_type="pooling", check_and_update_config() forces
    long_prefill_token_threshold = max_model_len (and disables chunked
    prefill) before the prefill_seq_len formula runs, so prefill_seq_len
    ends up equal to max_model_len regardless of the caller-supplied
    threshold."""

    def test_pooling_model_uses_max_model_len_as_threshold(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=512,
            runner_type="pooling",
            overrides={"max_num_batched_tokens": 4096},
        )
        assert not cfg.scheduler_config.enable_chunked_prefill
        assert cfg.scheduler_config.long_prefill_token_threshold == 512
        assert cfg.additional_config["override_qaic_config"]["prefill_seq_len"] == 512

    def test_pooling_model_budget_equals_max_model_len(self, make_vllm_config):
        cfg = _check_and_update(
            make_vllm_config,
            max_model_len=512,
            max_num_seqs=1,
            runner_type="pooling",
            overrides={"max_num_batched_tokens": 512},
        )
        assert cfg.scheduler_config.max_num_batched_tokens == 512


# ===========================================================================
# 4. prefill_seq_len from _clean_config
# ===========================================================================

class TestPrefillSeqLenFromCleanConfig:
    def test_prefill_seq_len_passes_through(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"prefill_seq_len": "128"})
        assert "prefill_seq_len" in result

    def test_prefill_seq_len_integer_passes_through(self):
        from vllm_qaic.utils.qaic_utils import _clean_config
        result = _clean_config({"prefill_seq_len": 256})
        assert "prefill_seq_len" in result


# ===========================================================================
# 5. Eager mode: the entire AOT-only prefill_seq_len/max_num_batched_tokens
# formula (platform_base.py's `if cls.is_aot:` block) must not run at all.
# ===========================================================================

class TestChunkedPrefillEagerModeSkipsFormula:
    """QaicPlatform.is_aot is a plain class attribute — monkeypatching it to
    False drives check_and_update_config()'s eager branch without requiring
    torch_qaic or real hardware. torch.device("qaic") also requires the
    PrivateUse1 backend to have been named "qaic" (normally a side effect of
    importing real torch_qaic); register it directly here so the device_config
    assignment in the eager branch (which runs before the formula block)
    doesn't raise."""

    @pytest.fixture(autouse=True)
    def _register_qaic_device(self):
        import torch
        try:
            torch.device("qaic")
        except RuntimeError:
            torch.utils.rename_privateuse1_backend("qaic")

    def test_max_num_batched_tokens_untouched_in_eager(self, make_vllm_config, monkeypatch):
        """In AOT mode this same config (max_num_seqs=4, prefill threshold=128)
        would shrink max_num_batched_tokens to 512 (test_budget_limited_by_seqs_
        times_prefill above). In eager mode the whole `if cls.is_aot:` formula
        block is skipped, so the caller-supplied value must be left as-is."""
        import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
        from vllm_qaic.platform_base import QaicPlatform

        monkeypatch.setattr(QaicPlatform, "is_aot", False)
        cfg = make_vllm_config(max_model_len=2048, max_num_seqs=4)
        cfg.scheduler_config.long_prefill_token_threshold = 128
        cfg.scheduler_config.max_num_batched_tokens = 4096
        QaicPlatform.check_and_update_config(cfg)
        assert cfg.scheduler_config.max_num_batched_tokens == 4096

    def test_prefill_seq_len_not_set_in_eager(self, make_vllm_config, monkeypatch):
        """The formula block is also what writes override_qaic_config
        ["prefill_seq_len"] in AOT mode; in eager mode it must stay absent."""
        import vllm  # ensure vllm is fully initialized before vllm_qaic.platform_base
        from vllm_qaic.platform_base import QaicPlatform

        monkeypatch.setattr(QaicPlatform, "is_aot", False)
        cfg = make_vllm_config(max_model_len=2048)
        QaicPlatform.check_and_update_config(cfg)
        assert "prefill_seq_len" not in cfg.additional_config["override_qaic_config"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
