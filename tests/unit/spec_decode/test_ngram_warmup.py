# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU-only tests for the AOT n-gram proposer startup warm-up."""

import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

numba = pytest.importorskip("numba")
from vllm.v1.spec_decode.ngram_proposer import NgramProposer  # noqa: E402

from vllm_qaic.spec_decode import ngram_warmup  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
MAX_MODEL_LEN, MAX_NUM_SEQS, K = 256, 4, 4


def _make_config(method="ngram"):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method=method,
            prompt_lookup_min=1,
            prompt_lookup_max=7,
            num_speculative_tokens=K,
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_SEQS),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )


def _production_input_batch():
    """Arrays laid out as QaicModelRunnerAoT._postprocess_tensors builds them."""
    tok = torch.zeros((MAX_NUM_SEQS, MAX_MODEL_LEN), dtype=torch.int32)
    tok = tok.to(torch.int64) - 1
    pad = torch.zeros((MAX_NUM_SEQS, K), dtype=torch.int64) - 1
    token_ids_cpu = torch.cat([tok, pad], dim=1).numpy()
    num_tokens_no_spec = torch.zeros(MAX_NUM_SEQS, dtype=torch.int32).numpy()
    token_ids_cpu[:2, :12] = np.tile([5, 6, 7, 8], 3)
    num_tokens_no_spec[:2] = 12
    return SimpleNamespace(
        token_ids_cpu=token_ids_cpu, num_tokens_no_spec=num_tokens_no_spec
    )


# Fresh interpreter so the numba dispatcher has no prior specializations.
_COMPILE_PROBE = textwrap.dedent(
    """
    import json, sys
    from numba.core import event
    sys.path.insert(0, {test_dir!r})
    import test_ngram_warmup as t
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer
    from vllm_qaic.spec_decode.ngram_warmup import maybe_warm_up_ngram_proposer

    cfg = t._make_config()
    drafter = NgramProposer(cfg)
    batch = t._production_input_batch()
    if {warm}:
        maybe_warm_up_ngram_proposer(cfg.speculative_config, drafter, batch)
    with event.install_recorder("numba:compile") as rec:
        drafts = drafter.propose(
            [[1], [1], [], []], batch.num_tokens_no_spec, batch.token_ids_cpu,
            slot_mappings=None,
        )
    n = sum(1 for _, ev in rec.buffer if ev.is_start)
    print("RESULT", json.dumps({{"compiles": n, "drafts": drafts}}))
    """
)


def _run_probe(warm: bool) -> dict:
    code = _COMPILE_PROBE.format(test_dir=str(Path(__file__).parent), warm=warm)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


@pytest.mark.parametrize("warm", [False, True])
def test_production_call_compiles_only_without_warmup(warm):
    result = _run_probe(warm)
    assert result["drafts"] == [[5, 6, 7, 8], [5, 6, 7, 8], [], []]
    if warm:
        assert result["compiles"] == 0
    else:
        # Control: shows the probe can detect the first-request compile.
        assert result["compiles"] > 0


def test_warmup_leaves_live_state_untouched():
    cfg = _make_config()
    drafter = NgramProposer(cfg)
    batch = _production_input_batch()
    drafter.valid_ngram_draft[:] = 7
    drafter.valid_ngram_num_drafts[:] = 3
    before = {
        "tok": batch.token_ids_cpu.copy(),
        "ntok": batch.num_tokens_no_spec.copy(),
        "draft": drafter.valid_ngram_draft.copy(),
        "num_drafts": drafter.valid_ngram_num_drafts.copy(),
    }
    threads = numba.get_num_threads()
    torch_rng = torch.random.get_rng_state()
    np_rng = np.random.get_state(legacy=False)

    dt = ngram_warmup.maybe_warm_up_ngram_proposer(
        cfg.speculative_config, drafter, batch
    )

    assert dt is not None and dt >= 0.0
    np.testing.assert_array_equal(batch.token_ids_cpu, before["tok"])
    np.testing.assert_array_equal(batch.num_tokens_no_spec, before["ntok"])
    np.testing.assert_array_equal(drafter.valid_ngram_draft, before["draft"])
    np.testing.assert_array_equal(drafter.valid_ngram_num_drafts, before["num_drafts"])
    assert batch.token_ids_cpu.dtype == np.int64
    assert numba.get_num_threads() == threads
    assert torch.equal(torch.random.get_rng_state(), torch_rng)
    np_rng_after = np.random.get_state(legacy=False)
    np.testing.assert_array_equal(np_rng_after["state"]["key"], np_rng["state"]["key"])
    assert np_rng_after["state"]["pos"] == np_rng["state"]["pos"]
    assert np_rng_after["has_gauss"] == np_rng["has_gauss"]


class _RecordingDrafter:
    def __init__(self, exc: Exception | None = None):
        self.calls = 0
        self.exc = exc
        self.max_model_len = MAX_MODEL_LEN
        self.valid_ngram_draft: np.ndarray = np.zeros((MAX_NUM_SEQS, K), dtype=np.int32)
        self.valid_ngram_num_drafts: np.ndarray = np.zeros(MAX_NUM_SEQS, dtype=np.int32)

    def propose(self, *args, **kwargs):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return [[]]


@pytest.mark.parametrize("method", [None, "suffix", "draft_model", "dflash"])
def test_gating_skips_non_ngram(method):
    spec = None if method is None else _make_config(method).speculative_config
    drafter = _RecordingDrafter()
    batch = _production_input_batch()
    assert ngram_warmup.maybe_warm_up_ngram_proposer(spec, drafter, batch) is None
    assert drafter.calls == 0


def test_gating_skips_missing_drafter():
    # Disagg KV producer: the runner clears the drafter.
    spec = _make_config().speculative_config
    batch = _production_input_batch()
    assert ngram_warmup.maybe_warm_up_ngram_proposer(spec, None, batch) is None


def test_runner_warmup_handles_a_nospec_runner_without_drafter(monkeypatch):
    """Regression for AOT nospec startup: GPUModelRunner omits ``drafter``."""
    from vllm_qaic.worker.model_runner import QaicModelRunnerAoT

    runner = SimpleNamespace(
        speculative_config=None,
        input_batch=object(),
    )
    calls = []
    monkeypatch.setattr(
        ngram_warmup,
        "maybe_warm_up_ngram_proposer",
        lambda spec, drafter, batch: calls.append((spec, drafter, batch)),
    )

    QaicModelRunnerAoT._qaic_warm_up_drafter(runner)

    assert calls == [(None, None, runner.input_batch)]


def test_failure_warns_and_continues(caplog):
    spec = _make_config().speculative_config
    drafter = _RecordingDrafter(exc=RuntimeError("boom"))
    batch = _production_input_batch()
    logger = logging.getLogger(ngram_warmup.__name__)
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=ngram_warmup.__name__):
            result = ngram_warmup.maybe_warm_up_ngram_proposer(spec, drafter, batch)
    finally:
        logger.removeHandler(caplog.handler)
    assert result is None
    assert drafter.calls == 1
    assert any("warm-up failed" in r.getMessage() for r in caplog.records)
