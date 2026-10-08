# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""CPU tests for AOT speculative-sampler startup gating."""

import builtins
import sys
from types import SimpleNamespace

import pytest
import torch

import vllm_qaic.v1.sample as sample_package
import vllm_qaic.worker.worker as worker_module
from vllm_qaic.worker.worker import QaicWorkerAoT


def _worker(speculative_config):
    worker = object.__new__(QaicWorkerAoT)
    worker.speculative_config = speculative_config
    return worker


def _init_worker(speculative_config):
    worker = _worker(speculative_config)
    worker.device_config = SimpleNamespace(device=torch.device("cpu"))
    worker.vllm_config = SimpleNamespace(
        additional_config={"device_group": [19]},
    )
    worker.rank = 0
    worker.local_rank = 0
    worker.distributed_init_method = ""
    worker.model_config = SimpleNamespace(seed=0)
    return worker


def _stub_init_device(monkeypatch, worker, runner):
    monkeypatch.setattr(
        worker,
        "_init_qaic_worker_distributed_environment",
        lambda *args: None,
    )
    monkeypatch.setattr(worker, "_configure_thread_parallelism", lambda: None)
    monkeypatch.setattr(worker_module, "set_random_seed", lambda seed: None)
    monkeypatch.setattr(
        worker_module, "QaicModelRunnerAoT", lambda config, device: runner
    )
    monkeypatch.setitem(
        sys.modules,
        "qaicrt",
        SimpleNamespace(
            Util=lambda: SimpleNamespace(getDeviceIds=lambda: (None, [19]))
        ),
    )


def test_no_spec_does_not_import_or_require_numba(monkeypatch):
    real_import = builtins.__import__

    def reject_numba_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_qaic.v1.sample" and "rejection_sampler_numba" in fromlist:
            raise AssertionError("Numba rejection sampler imported for no-spec AOT")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_numba_import)

    _worker(None)._install_aot_rejection_sampler()


def test_no_spec_init_device_does_not_import_numba(monkeypatch):
    real_import = builtins.__import__

    def reject_numba_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vllm_qaic.v1.sample" and "rejection_sampler_numba" in fromlist:
            raise AssertionError("Numba rejection sampler imported for no-spec AOT")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_numba_import)
    runner = SimpleNamespace()
    worker = _init_worker(None)
    _stub_init_device(monkeypatch, worker, runner)

    QaicWorkerAoT.init_device(worker)

    assert worker.model_runner is runner


def test_speculative_init_device_installs_numba(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(install=lambda: calls.append("install"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )
    worker = _init_worker(SimpleNamespace(method="ngram"))
    _stub_init_device(monkeypatch, worker, SimpleNamespace())

    QaicWorkerAoT.init_device(worker)

    assert calls == ["install"]


def test_speculative_aot_installs_numba(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(install=lambda: calls.append("install"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )

    _worker(SimpleNamespace(method="ngram"))._install_aot_rejection_sampler()

    assert calls == ["install"]


def test_no_spec_skips_numba_prewarm_and_drafter_warmup(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(prewarm=lambda: calls.append("prewarm"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )
    worker = _worker(None)
    worker.model_runner = SimpleNamespace(
        _qaic_warm_up_drafter=lambda: calls.append("drafter"),
        _qaic_dummy_run=lambda: calls.append("dummy"),
    )

    worker.compile_or_warm_up_model()

    assert calls == ["dummy"]


def test_speculative_aot_prewarms_before_drafter_and_dummy_run(monkeypatch):
    calls = []
    fake_numba = SimpleNamespace(prewarm=lambda: calls.append("prewarm"))
    monkeypatch.setattr(
        sample_package, "rejection_sampler_numba", fake_numba, raising=False
    )
    worker = _worker(SimpleNamespace(method="ngram"))
    worker.model_runner = SimpleNamespace(
        _qaic_warm_up_drafter=lambda: calls.append("drafter"),
        _qaic_dummy_run=lambda: calls.append("dummy"),
    )

    worker.compile_or_warm_up_model()

    assert calls == ["prewarm", "drafter", "dummy"]


@pytest.mark.parametrize("method", ["suffix", "draft_model", "dflash"])
def test_drafter_warmup_gating_for_non_ngram_and_missing_drafter(method):
    # The model-runner helper itself remains the safe boundary: only an active
    # ngram config with a drafter reaches the Numba proposer warm-up.
    from vllm_qaic.spec_decode.ngram_warmup import maybe_warm_up_ngram_proposer

    spec = SimpleNamespace(method=method)
    assert maybe_warm_up_ngram_proposer(spec, object(), object()) is None
    assert (
        maybe_warm_up_ngram_proposer(SimpleNamespace(method="ngram"), None, object())
        is None
    )
