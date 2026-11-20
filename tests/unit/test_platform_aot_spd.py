# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""CPU tests for the AOT speculative-decoding platform config checks."""

from types import SimpleNamespace

import pytest

from vllm_qaic import platform as qaic_platform
from vllm_qaic.platform import QaicPlatform

METHODS = ["ngram", "suffix", "draft_model", "dflash"]


def _vllm_config(method, async_scheduling, dynamic=False):
    speculative_config = SimpleNamespace(
        method=method, uses_dynamic_speculative_decoding=lambda: dynamic
    )
    return SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )


@pytest.fixture
def warnings(monkeypatch):
    calls = []
    monkeypatch.setattr(
        qaic_platform.logger, "warning_once", lambda *a, **k: calls.append(a)
    )
    return calls


@pytest.mark.parametrize("method", METHODS)
def test_async_scheduling_falls_back_to_sync(method, warnings):
    # vLLM >= 0.30 resolves async_scheduling=None to True for draft_model and
    # dflash before the platform hook; the QAIC proposers require sync.
    config = _vllm_config(method, async_scheduling=True)
    QaicPlatform._check_aot_speculative_config(config)
    assert config.scheduler_config.async_scheduling is False
    assert len(warnings) == 1 and method in warnings[0]


@pytest.mark.parametrize("method", METHODS)
def test_sync_scheduling_is_unchanged(method, warnings):
    config = _vllm_config(method, async_scheduling=False)
    QaicPlatform._check_aot_speculative_config(config)
    assert config.scheduler_config.async_scheduling is False
    assert warnings == []


def test_dynamic_speculative_decoding_is_rejected():
    # The decode QPCs are compiled for a fixed num_speculative_tokens.
    config = _vllm_config("ngram", async_scheduling=False, dynamic=True)
    with pytest.raises(ValueError, match="Dynamic speculative decoding"):
        QaicPlatform._check_aot_speculative_config(config)
