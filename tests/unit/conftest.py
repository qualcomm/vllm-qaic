# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
conftest.py for vllm-qaic unit tests.

All on-device tests run in AOT (Ahead-of-Time) mode only.
torch_qaic / eager mode is not used.
"""

import os
import types

import pytest
import torch


# ---------------------------------------------------------------------------
# Shared mock vllm_config factory for QaicPlatform.check_and_update_config()
# tests. Consolidates what used to be three near-identical, drifting
# `_make_vllm_config()` helpers (generic/test_platform.py,
# disaggregated/test_disaggregated_constraints.py,
# prefixcaching/test_prefix_caching.py) plus a fourth ad-hoc version
# (generic/test_error_handling.py) into a single source of truth.
# ---------------------------------------------------------------------------

@pytest.fixture
def make_vllm_config():
    """Return a factory building a SimpleNamespace vllm_config for
    QaicPlatform.check_and_update_config() tests.

    Common fields are explicit kwargs; anything else can be set via
    **overrides (applied with setattr after the namespace is built).
    """

    def _make(
        max_model_len=2048,
        model_type="llama",
        is_multimodal=False,
        runner_type="generate",
        mm_processor_kwargs=None,
        additional_config=None,
        enable_prefix_caching=False,
        block_size=16,
        mamba_block_size=2048,
        mamba_cache_mode="none",
        max_num_seqs=4,
        lora_config=None,
        speculative_config=None,
        kv_transfer_config=None,
        **overrides,
    ):
        model_config = types.SimpleNamespace(
            max_model_len=max_model_len,
            hf_config=types.SimpleNamespace(model_type=model_type),
            is_multimodal_model=is_multimodal,
            runner_type=runner_type,
            mm_processor_kwargs=mm_processor_kwargs,
            enforce_eager=False,
        )
        device_config = types.SimpleNamespace(
            device=torch.device("cpu"), device_type="cpu"
        )
        scheduler_config = types.SimpleNamespace(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=128,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=512,
            async_scheduling=False,
        )
        cache_config = types.SimpleNamespace(
            enable_prefix_caching=enable_prefix_caching,
            block_size=block_size,
            mamba_block_size=mamba_block_size,
            mamba_cache_mode=mamba_cache_mode,
        )
        parallel_config = types.SimpleNamespace(
            worker_cls="auto", world_size=1, distributed_executor_backend=None,
        )
        cfg = types.SimpleNamespace(
            additional_config=(
                additional_config
                if additional_config is not None
                else {"override_qaic_config": {}}
            ),
            model_config=model_config,
            device_config=device_config,
            scheduler_config=scheduler_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            lora_config=lora_config,
            speculative_config=speculative_config,
            kv_transfer_config=kv_transfer_config,
            compilation_config=None,
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    return _make


@pytest.fixture
def make_kv_transfer_config():
    """Return a factory building a SimpleNamespace kv_transfer_config."""

    def _make(kv_role="kv_both"):
        return types.SimpleNamespace(kv_role=kv_role)

    return _make


# ---------------------------------------------------------------------------
# Workaround for a known platform_base.py bug (not fixed here — see
# generic/test_platform.py::TestDisaggregatedServingConstraints for details):
# with kv_role="kv_producer", check_and_update_config() does
#   from vllm_qaic.executor.qaic_uniproc_executor import QaicUniProcExecutor
# but no "vllm_qaic.executor" subpackage exists (the real module lives at
# vllm_qaic.qaic_uniproc_executor), so the import always raises
# ModuleNotFoundError before the kv_producer logic under test can run.
# This fixture stubs the broken import path in sys.modules so tests can
# exercise the kv_producer code path without touching the source.
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_qaic_executor_import_bug(monkeypatch):
    import sys
    import vllm_qaic.qaic_uniproc_executor as real_module

    fake_package = type(sys)("vllm_qaic.executor")
    monkeypatch.setitem(sys.modules, "vllm_qaic.executor", fake_package)
    monkeypatch.setitem(
        sys.modules, "vllm_qaic.executor.qaic_uniproc_executor", real_module
    )
    yield


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_global_cleanup: skip the global torch/distributed cleanup fixture",
    )


# ---------------------------------------------------------------------------
# Skip global cleanup for all unit tests (no torch.distributed used here)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _skip_global_cleanup(request):
    """Override the parent conftest cleanup: unit tests need no teardown."""
    yield  # nothing to do before or after


# ---------------------------------------------------------------------------
# Environment-variable isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_qaic_env(monkeypatch):
    """
    Remove all VLLM_QAIC_* and QAIC_* environment variables before each test
    so that tests are not affected by the caller's shell environment.
    """
    qaic_vars = [k for k in os.environ if k.startswith(("VLLM_QAIC_", "QAIC_", "VLLM_TORCH_QAIC_"))]
    for var in qaic_vars:
        monkeypatch.delenv(var, raising=False)
    yield
