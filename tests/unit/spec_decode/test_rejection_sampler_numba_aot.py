# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""CPU-only tests for the QAIC AOT Numba rejection sampler.

Bit-equivalence of the Numba kernels against triton-cpu is covered by
tests/unit/spec_decode/rejection_parity/; these tests cover
install/uninstall, validation guards and counter instrumentation.
"""

import json

import pytest
import torch

pytest.importorskip("numba")
import vllm.v1.sample.rejection_sampler as rs  # noqa: E402

from vllm_qaic.v1.sample import rejection_sampler_numba as rsn  # noqa: E402

UPSTREAM = {n: getattr(rs, n) for n in rsn.KERNEL_NAMES + rsn.DISPATCHER_NAMES}


@pytest.fixture(autouse=True)
def _restore(monkeypatch, tmp_path):
    for var in (
        rsn._REMOVED_IMPL_ENV,
        "VLLM_QAIC_RS_COUNTERS",
        "VLLM_QAIC_RS_COUNTERS_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba_cache"))
    rsn.uninstall()
    # Captured per test: other modules (vllm_qaic.patch) may legitimately
    # replace RejectionSampler.forward at import time.
    forward = rs.RejectionSampler.forward
    yield forward
    rsn.uninstall()
    for n, k in UPSTREAM.items():
        assert getattr(rs, n) is k
    assert rs.RejectionSampler.forward is forward


def test_install_swaps_and_uninstall_restores(_restore):
    assert rsn.install() == "numba"
    for n in rsn.KERNEL_NAMES:
        assert getattr(rs, n) is not UPSTREAM[n]
        assert isinstance(getattr(rs, n), rsn._NumbaKernel)
    for n in rsn.DISPATCHER_NAMES:
        assert isinstance(getattr(rs, n), rsn._NumbaDispatcher)
        assert getattr(rs, n).kernel is getattr(rs, rsn.DISPATCHER_KERNELS[n])
    assert rs.RejectionSampler.forward is _restore
    rsn.uninstall()
    for n, k in UPSTREAM.items():
        assert getattr(rs, n) is k


def test_rejection_sample_routes_through_numba(monkeypatch):
    """v0.30 ``rejection_sample()`` must reach the Numba kernels.

    Upstream dispatchers capture their Triton kernel at import, so replacing
    only the ``*_kernel`` globals is silently bypassed (vLLM v0.30 regression).
    """
    from vllm.v1.sample.metadata import SamplingMetadata

    calls = []
    for n in rsn.KERNEL_NAMES:
        fn = rsn._NUMBA_FNS[n]

        def _spy(*args, _n=n, _fn=fn, **kwargs):
            calls.append(_n)
            return _fn(*args, **kwargs)

        # Spy before install(); the _restore fixture uninstalls afterwards.
        monkeypatch.setitem(rsn._NUMBA_FNS, n, _spy)
    rsn.install()

    B, K, V = 2, 2, 16
    draft = torch.tensor([3, 4, 5, 6], dtype=torch.int64)
    cu = torch.tensor([K, 2 * K], dtype=torch.int32)
    logits = torch.randn(B * K, V, generator=torch.Generator().manual_seed(0))
    logits[0, 3] = 100.0  # request 0 accepts its first draft under greedy
    bonus = torch.zeros((B, 1), dtype=torch.int32)
    meta = SamplingMetadata(
        temperature=torch.tensor([0.0, 1.0]),
        all_greedy=False,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(B),
        presence_penalties=torch.zeros(B),
        repetition_penalties=torch.ones(B),
        output_token_ids=[[], []],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=None,
    )
    out = rs.rejection_sample(draft, [K, K], K, cu, None, logits, bonus, meta)
    temp = rs.expand_batch_to_tokens(meta.temperature, cu, B * K)
    assert torch.equal(temp, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert out.shape == (B, K + 1)
    assert out[0, 0] == 3
    assert set(calls) >= {
        "expand_kernel",
        "rejection_greedy_sample_kernel",
        "sample_recovered_tokens_kernel",
        "rejection_random_sample_kernel",
    }


def test_register_warmup_is_noop_after_install(monkeypatch):
    # RejectionSampler.__init__ registers every dispatcher for Triton warm-up;
    # after install() none of them may reach the registry.
    from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

    calls = []
    monkeypatch.setattr(JitWarmupRegistry, "register", lambda *a, **k: calls.append(a))
    rsn.install()
    for n in rsn.DISPATCHER_NAMES:
        getattr(rs, n).register_warmup(synthetic_mode=True)
    assert calls == []


def test_install_idempotent():
    rsn.install()
    first = rs.expand_kernel
    rsn.install()
    assert rs.expand_kernel is first


@pytest.mark.parametrize("value", ["triton", "numba"])
def test_removed_selector_warns_and_is_ignored(monkeypatch, value):
    monkeypatch.setenv(rsn._REMOVED_IMPL_ENV, value)
    warnings = []
    monkeypatch.setattr(rsn.logger, "warning", lambda *a, **k: warnings.append(a))
    assert rsn.install() == "numba"
    assert isinstance(rs.expand_kernel, rsn._NumbaKernel)
    assert len(warnings) == 1 and rsn._REMOVED_IMPL_ENV in warnings[0]


def test_install_fails_loudly_without_numba(monkeypatch):
    def _broken(threads):
        raise ImportError("No module named 'numba'")

    monkeypatch.setattr(rsn, "_configure_numba", _broken)
    with pytest.raises(RuntimeError, match="requires Numba"):
        rsn.install()
    # Nothing was swapped, so the fixture's restore check still holds.
    assert not rsn._installed


def test_expand_matches_reference_through_upstream_helper():
    rsn.install()
    temp = torch.tensor([0.0, 0.7, 1.3], dtype=torch.float32)
    cu = torch.tensor([2, 2, 5], dtype=torch.int32)
    out = rs.expand_batch_to_tokens(temp, cu, 5, replace_from=0, replace_to=1)
    assert out.dtype == torch.float32
    assert torch.equal(out, torch.tensor([1.0, 1.0, 1.3, 1.3, 1.3]))
    top_k = torch.tensor([5, 50, 7], dtype=torch.int32)
    out = rs.expand_batch_to_tokens(top_k, cu, 5)
    assert out.dtype == torch.int32
    assert out.tolist() == [5, 5, 7, 7, 7]


def _greedy_args(draft_dtype=torch.int64):
    cu = torch.tensor([2, 4], dtype=torch.int32)
    draft = torch.tensor([3, 4, 5, 6], dtype=draft_dtype)
    argmax = torch.tensor([3, 9, 5, 6], dtype=torch.int64)
    bonus = torch.tensor([[11], [12]], dtype=torch.int32)
    out = torch.full((2, 3), -1, dtype=torch.int32)
    return [out, cu, draft, argmax, bonus, None, 2, None, None]


def test_greedy_int64_production_dtypes():
    rsn.install()
    args = _greedy_args()
    rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)
    assert args[0].tolist() == [[3, 9, -1], [5, 6, 12]]


def test_greedy_synthetic_rejects_padded_draft():
    # v0.30: a -1 draft id marks a padded slot and is rejected even when the
    # synthetic acceptance rate would accept it.
    rsn.install()
    out = torch.full((1, 3), -1, dtype=torch.int32)
    rs._rejection_greedy_sample(
        out,
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([3, -1], dtype=torch.int64),
        torch.tensor([7, 8], dtype=torch.int64),
        torch.tensor([[11]], dtype=torch.int32),
        None,
        2,
        torch.zeros(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float32),
    )
    assert out.tolist() == [[3, 8, -1]]


@pytest.mark.parametrize("with_draft_probs", [False, True])
@pytest.mark.parametrize("synthetic", [False, True])
def test_random_rejects_padded_draft(with_draft_probs, synthetic):
    # v0.30: a -1 draft id is rejected before any probability is indexed and
    # replaced by the recovered token.
    rsn.install()
    V = 16
    probs = torch.full((2, V), 1.0 / V, dtype=torch.float32)
    out = torch.full((1, 3), -1, dtype=torch.int32)
    rs._rejection_random_sample(
        out,
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([3, -1], dtype=torch.int64),
        probs.clone() if with_draft_probs else None,
        probs,
        torch.tensor([[11]], dtype=torch.int32),
        torch.tensor([5, 9], dtype=torch.int64),
        torch.zeros(2, dtype=torch.float64),  # uniform 0: accept any valid draft
        torch.zeros(1, dtype=torch.bool),
        2,
        torch.ones(2, dtype=torch.float32) if synthetic else None,
    )
    assert out.tolist() == [[3, 9, -1]]


@pytest.mark.parametrize(
    "mutate,exc,match",
    [
        # Dtype allowlist (_check): only prewarmed, parity-covered dtypes.
        (
            lambda a: a.__setitem__(2, a[2].to(torch.float32)),
            TypeError,
            "draft_token_ids has unsupported dtype",
        ),
        (
            lambda a: a.__setitem__(3, a[3].to(torch.int16)),
            TypeError,
            "target_argmax has unsupported dtype",
        ),
        (
            lambda a: a.__setitem__(7, torch.rand(4, dtype=torch.float32)),
            TypeError,
            "uniform_probs has unsupported dtype",
        ),
        # Contiguity / device are enforced by numba_rejection_kernels._np.
        (
            lambda a: a.__setitem__(0, torch.full((3, 2), -1, dtype=torch.int32).t()),
            ValueError,
            "contiguous",
        ),
        (
            lambda a: a.__setitem__(
                2, torch.empty(4, dtype=torch.int64, device="meta")
            ),
            TypeError,
            "meta",
        ),
    ],
)
def test_guards_raise(mutate, exc, match):
    rsn.install()
    args = _greedy_args()
    mutate(args)
    with pytest.raises(exc, match=match):
        rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)


def test_recovered_rejects_wrong_inv_q_dtype():
    rsn.install()
    cu = torch.tensor([2], dtype=torch.int32)
    draft = torch.zeros(2, dtype=torch.int64)
    probs = torch.softmax(torch.randn(2, 16), -1)
    inv_q = torch.ones(1, 16, dtype=torch.float32)
    with pytest.raises(TypeError, match="inv_q"):
        rs.sample_recovered_tokens_kernel[(1, 2)](
            torch.empty_like(draft),
            cu,
            draft,
            None,
            probs,
            inv_q,
            16,
            8192,
            NO_DRAFT_PROBS=True,
            USE_FP64_GUMBEL=True,
        )


def test_prewarm_noop_before_install():
    assert rsn.prewarm() == 0.0


def test_prewarm_numba():
    rsn.install()
    assert rsn.prewarm() > 0.0


def test_counters_json(_restore, monkeypatch, tmp_path):
    impl = "numba"
    monkeypatch.setenv("VLLM_QAIC_RS_COUNTERS", "1")
    monkeypatch.setenv("VLLM_QAIC_RS_COUNTERS_DIR", str(tmp_path / "c"))
    rsn.install()
    assert rs.RejectionSampler.forward is not _restore
    for _ in range(3):
        args = _greedy_args()
        rs.rejection_greedy_sample_kernel[(2,)](*args, SYNTHETIC_MODE=False)
    rsn._instrumentation.flush()
    (counter_file,) = (tmp_path / "c").glob("rs_counters_*.json")
    payload = json.loads(counter_file.read_text())
    assert payload["impl"] == impl
    (key,) = payload["kernels"]
    assert key.startswith("rejection_greedy_sample|SYNTHETIC_MODE=0|mask=none")
    assert "int64" in key and payload["kernels"][key]["calls"] == 3


def test_prewarm_does_not_consume_global_rng():
    # The worker seeds the global RNG before warm-up; prewarm must not shift
    # the stream unseeded requests later sample from.
    rsn.install()
    torch.manual_seed(1234)
    expected = torch.rand(8)
    torch.manual_seed(1234)
    rsn.prewarm()
    assert torch.equal(torch.rand(8), expected)


def test_install_preserves_torch_thread_cap(tmp_path):
    # Numba's OpenMP pool shares torch's libgomp and, on start-up, resets the
    # OpenMP thread count to NUMBA_NUM_THREADS (CPU count).  The pool starts
    # once per process, so check in a fresh interpreter, mirroring the worker
    # (torch capped first, then install()).
    import os
    import subprocess
    import sys

    code = (
        "import torch\n"
        "torch.set_num_threads(2)\n"
        # Import vllm first, as the worker does: importing vllm_qaic.logger
        # first re-enters it via vllm's platform plugin resolution.
        "import vllm  # noqa: F401\n"
        "from vllm_qaic.v1.sample import rejection_sampler_numba as rsn\n"
        "rsn.install()\n"
        "import numba\n"
        "print(torch.get_num_threads(), numba.config.NUMBA_NUM_THREADS)\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS")
    }
    env["NUMBA_CACHE_DIR"] = str(tmp_path / "numba_cache")
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    torch_threads, numba_max = int(out[-2]), int(out[-1])
    if numba_max <= 2:
        pytest.skip("host too small for the pool to change the torch thread count")
    assert torch_threads == 2
