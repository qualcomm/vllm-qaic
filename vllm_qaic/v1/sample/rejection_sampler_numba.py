# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC AOT rejection sampler: Numba replacement for the triton kernels.

On AOT the upstream rejection sampler runs on the host.  Its four
``@triton.jit`` kernels are module globals of
``vllm.v1.sample.rejection_sampler`` looked up at call time, so ``install()``
only replaces those globals with the bit-exact Numba ports from
``vllm_qaic.v1.sample.numba_rejection_kernels``; the upstream call sites
(``kernel[grid](...)``) are untouched.  No triton (triton-cpu) install is
needed: without triton the upstream globals are uncallable placeholders, so
``install()`` must run before the first speculative-decoding step.

Debug-only instrumentation (adds per-launch overhead):

* ``VLLM_QAIC_RS_COUNTERS=1`` -- count launches per (kernel, constexpr combo,
  dtype signature, batch, token bucket) plus kernel/forward wall time, written
  as JSON to ``$VLLM_QAIC_RS_COUNTERS_DIR/rs_counters_<pid>.json``.
"""

from __future__ import annotations

import atexit
import json
import os
import socket
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

NUMBA_IMPL = "numba"
# Removed backend selector; only checked to warn users who still set it.
_REMOVED_IMPL_ENV = "VLLM_QAIC_AOT_REJECTION_SAMPLER_IMPL"

KERNEL_NAMES = (
    "expand_kernel",
    "rejection_greedy_sample_kernel",
    "rejection_random_sample_kernel",
    "sample_recovered_tokens_kernel",
)

_installed = False
_originals: dict[str, Any] = {}
_original_forward: Any = None
_numba_threads: int | None = None


# --------------------------------------------------------------------------
# Numba launchers (upstream ``kernel[grid](*args, **kwargs)`` layout)
# --------------------------------------------------------------------------
_INT = (torch.int32, torch.int64)
_FLOAT = (torch.float32, torch.float64)


def _check(name: str, t: torch.Tensor | None, dtypes: tuple, optional=False):
    # Dtype allowlist only: these are the dtypes prewarm() compiles and the
    # Triton parity suites cover, so anything else would JIT-compile mid-serving
    # with unverified numerics.  Device / contiguity are enforced by
    # numba_rejection_kernels._np (via .numpy()).
    if t is None:
        if optional:
            return
        raise ValueError(f"numba rejection sampler: {name} must not be None")
    if t.dtype not in dtypes:
        raise TypeError(
            f"numba rejection sampler: {name} has unsupported dtype {t.dtype}; "
            f"expected one of {dtypes}"
        )


def _ensure_numba_threads() -> None:
    # numba.set_num_threads is thread-local; a launch from a thread other than
    # the one that ran install() would otherwise use NUMBA_NUM_THREADS.
    if _numba_threads is None:
        return
    import numba

    if numba.get_num_threads() != _numba_threads:
        numba.set_num_threads(_numba_threads)


def _expand(out, x, cu, replace_from, replace_to, MAX_NUM_TOKENS=None):
    from vllm_qaic.v1.sample import numba_rejection_kernels as nrk

    _check("output", out, _INT + _FLOAT)
    _check("input", x, (out.dtype,))
    _check("cu_num_tokens", cu, _INT)
    nrk.expand_numba(
        out,
        x,
        cu,
        getattr(replace_from, "value", replace_from),
        getattr(replace_to, "value", replace_to),
    )


def _greedy(
    out,
    cu,
    draft,
    argmax,
    bonus,
    is_greedy,
    max_spec_len,
    uniform,
    rates,
    SYNTHETIC_MODE=False,
):
    from vllm_qaic.v1.sample import numba_rejection_kernels as nrk

    _check("output_token_ids", out, _INT)
    _check("cu_num_draft_tokens", cu, _INT)
    _check("draft_token_ids", draft, _INT)
    _check("target_argmax", argmax, _INT)
    _check("bonus_token_ids", bonus, _INT)
    _check("is_greedy", is_greedy, (torch.bool,), optional=True)
    _check("uniform_probs", uniform, (torch.float64,), optional=True)
    _check("synthetic_conditional_rates", rates, (torch.float32,), optional=True)
    if SYNTHETIC_MODE and (uniform is None or rates is None):
        raise ValueError("SYNTHETIC_MODE requires uniform_probs and rates")
    nrk.greedy_numba(
        out,
        cu,
        draft,
        argmax,
        bonus,
        is_greedy,
        int(max_spec_len),
        uniform,
        rates,
        bool(SYNTHETIC_MODE),
    )


def _random(
    out,
    cu,
    draft,
    draft_probs,
    target_probs,
    bonus,
    recovered,
    uniform,
    is_greedy,
    max_spec_len,
    vocab_size,
    rates,
    NO_DRAFT_PROBS=False,
    SYNTHETIC_MODE=False,
):
    from vllm_qaic.v1.sample import numba_rejection_kernels as nrk

    _check("output_token_ids", out, _INT)
    _check("cu_num_draft_tokens", cu, _INT)
    _check("draft_token_ids", draft, _INT)
    _check("draft_probs", draft_probs, (torch.float32,), optional=True)
    _check("target_probs", target_probs, (torch.float32,))
    _check("bonus_token_ids", bonus, _INT)
    _check("recovered_token_ids", recovered, _INT)
    _check("uniform_probs", uniform, (torch.float64,))
    _check("is_greedy", is_greedy, (torch.bool,))
    _check("synthetic_conditional_rates", rates, (torch.float32,), optional=True)
    if SYNTHETIC_MODE and rates is None:
        raise ValueError("SYNTHETIC_MODE requires synthetic_conditional_rates")
    if not NO_DRAFT_PROBS and draft_probs is None:
        raise ValueError("NO_DRAFT_PROBS=False requires draft_probs")
    nrk.random_numba(
        out,
        cu,
        draft,
        draft_probs,
        target_probs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        int(max_spec_len),
        vocab_size,
        rates,
        bool(NO_DRAFT_PROBS),
        bool(SYNTHETIC_MODE),
    )


def _recovered(
    out,
    cu,
    draft,
    draft_probs,
    target_probs,
    inv_q,
    vocab_size,
    BLOCK_SIZE=None,
    NO_DRAFT_PROBS=False,
    USE_FP64_GUMBEL=False,
):
    from vllm_qaic.v1.sample import numba_rejection_kernels as nrk

    _check("output_token_ids", out, _INT)
    _check("cu_num_draft_tokens", cu, _INT)
    _check("draft_token_ids", draft, _INT)
    _check("draft_probs", draft_probs, (torch.float32,), optional=True)
    _check("target_probs", target_probs, (torch.float32,))
    _check("inv_q", inv_q, (torch.float64,) if USE_FP64_GUMBEL else (torch.float32,))
    if not NO_DRAFT_PROBS and draft_probs is None:
        raise ValueError("NO_DRAFT_PROBS=False requires draft_probs")
    _ensure_numba_threads()
    nrk.recovered_numba(
        out,
        cu,
        draft,
        draft_probs,
        target_probs,
        inv_q,
        vocab_size,
        bool(NO_DRAFT_PROBS),
    )


class _NumbaKernel:
    """Grid-launch adapter: ``k[grid](*args, **kwargs)`` ignores the grid."""

    def __init__(self, name: str, fn):
        self.__name__ = name
        self._fn = fn

    def __getitem__(self, _grid):
        return self._fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<numba rejection kernel {self.__name__}>"


_NUMBA_FNS = {
    "expand_kernel": _expand,
    "rejection_greedy_sample_kernel": _greedy,
    "rejection_random_sample_kernel": _random,
    "sample_recovered_tokens_kernel": _recovered,
}


# --------------------------------------------------------------------------
# Debug instrumentation (counters)
# --------------------------------------------------------------------------
class _Instrumentation:
    def __init__(self, impl: str):
        self.impl = impl
        self.counters_dir = (
            os.environ.get("VLLM_QAIC_RS_COUNTERS_DIR")
            if os.environ.get("VLLM_QAIC_RS_COUNTERS", "0") not in ("", "0")
            else None
        )
        if self.counters_dir is not None and not self.counters_dir:
            self.counters_dir = os.getcwd()
        self.counts: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "ns": 0}
        )
        self.forward = {"calls": 0, "ns": 0}
        self.calls = 0
        self.lock = threading.Lock()
        if self.counters_dir:
            Path(self.counters_dir).mkdir(parents=True, exist_ok=True)
        if self.counters_dir:
            atexit.register(self.flush)

    @property
    def enabled(self) -> bool:
        return bool(self.counters_dir)

    @staticmethod
    def combo(name: str, grid, args, kwargs) -> str:
        parts = [name.replace("_kernel", "")]
        parts += [
            f"{k}={int(v) if isinstance(v, bool) else v}"
            for k, v in sorted(kwargs.items())
            if k not in ("BLOCK_SIZE", "MAX_NUM_TOKENS")
        ]
        if name == "rejection_greedy_sample_kernel":
            parts.append("mask=none" if args[5] is None else "mask=tensor")
        dts = ",".join(
            "None" if a is None else str(a.dtype).replace("torch.", "")
            for a in args
            if a is None or isinstance(a, torch.Tensor)
        )
        parts.append(f"dtypes=[{dts}]")
        return "|".join(parts)

    def wrap(self, name: str, inner):
        instr = self

        class _Wrapped:
            __name__ = name

            def __getitem__(self, grid):
                launch = inner[grid]

                def _launch(*args, **kwargs):
                    return instr.record(name, grid, launch, args, kwargs)

                return _launch

        return _Wrapped()

    def record(self, name, grid, launch, args, kwargs):
        combo = self.combo(name, grid, args, kwargs)
        t0 = time.perf_counter_ns()
        result = launch(*args, **kwargs)
        dt = time.perf_counter_ns() - t0
        with self.lock:
            batch = int(grid[0]) if isinstance(grid, tuple) and grid else -1
            num_tokens = (
                int(args[2].numel())
                if name != "expand_kernel"
                else int(args[0].numel())
            )
            bucket = 1 << max(num_tokens - 1, 0).bit_length()
            key = f"{combo}|B={batch}|T<={bucket}"
            self.counts[key]["calls"] += 1
            self.counts[key]["ns"] += dt
            entry = self.counts[key]
            entry["max_T"] = max(entry.get("max_T", 0), num_tokens)
            self.calls += 1
            if self.counters_dir and self.calls % 200 == 0:
                self.flush()
        return result

    def record_forward(self, dt_ns: int) -> None:
        with self.lock:
            self.forward["calls"] += 1
            self.forward["ns"] += dt_ns

    def flush(self) -> None:
        if not self.counters_dir:
            return
        path = Path(self.counters_dir) / f"rs_counters_{os.getpid()}.json"
        payload = {
            "impl": self.impl,
            "pid": os.getpid(),
            "numba_threads": _numba_threads,
            "torch_threads": torch.get_num_threads(),
            "forward": dict(self.forward),
            "kernels": {k: dict(v) for k, v in sorted(self.counts.items())},
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(path)


_instrumentation: _Instrumentation | None = None


def _wrap_forward(rs_mod, instr: _Instrumentation) -> None:
    global _original_forward
    cls = rs_mod.RejectionSampler
    _original_forward = cls.forward

    def _timed_forward(self, *args, **kwargs):
        t0 = time.perf_counter_ns()
        try:
            return _original_forward(self, *args, **kwargs)
        finally:
            instr.record_forward(time.perf_counter_ns() - t0)

    cls.forward = _timed_forward


# --------------------------------------------------------------------------
# Install / prewarm
# --------------------------------------------------------------------------
def _configure_numba(threads: int) -> dict[str, Any]:
    """Pin the Numba threading layer, cache dir and thread count.

    The cache dir must be set before the kernel module is imported (njit
    resolves its cache locator at decoration time); the layer and pool size
    must be set before the first parallel launch.
    """
    global _numba_threads
    cache_dir = os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path.home() / ".cache" / "vllm_qaic" / "numba" / socket.gethostname()),
    )
    os.environ.setdefault("NUMBA_THREADING_LAYER", "omp")
    import numba
    from numba.np.ufunc import parallel as numba_parallel

    numba.config.CACHE_DIR = cache_dir
    if not numba_parallel._is_initialized:
        numba.config.THREADING_LAYER = os.environ["NUMBA_THREADING_LAYER"]
    threads = max(1, min(threads, numba.config.NUMBA_NUM_THREADS))
    # The first numba.set_num_threads() starts Numba's OpenMP pool, which
    # binds to torch's libgomp and calls omp_set_num_threads(NUMBA_NUM_THREADS)
    # (the CPU count by default).  That silently undoes the worker's torch
    # thread cap, and torch ops in the sampler then oversubscribe the host and
    # stall for tens of ms.  Restore the caller's torch thread count.
    torch_threads = torch.get_num_threads()
    numba.set_num_threads(threads)
    if torch.get_num_threads() != torch_threads:
        torch.set_num_threads(torch_threads)
    _numba_threads = threads
    return {
        "threads": threads,
        "layer": numba.config.THREADING_LAYER,
        "cache": cache_dir,
    }


def install() -> str:
    """Install the Numba rejection-sampler kernels.  Idempotent per process."""
    global _installed, _instrumentation
    if _installed:
        return NUMBA_IMPL
    if os.environ.get(_REMOVED_IMPL_ENV):
        logger.warning(
            "%s is no longer supported and is ignored; the QAIC AOT rejection "
            "sampler always uses Numba.",
            _REMOVED_IMPL_ENV,
        )

    import vllm.v1.sample.rejection_sampler as rs

    try:
        info = _configure_numba(torch.get_num_threads())
        from vllm_qaic.v1.sample import numba_rejection_kernels  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "QAIC AOT speculative decoding requires Numba for the host rejection "
            "sampler (see requirements/vllm_dependency_aot.txt for the pinned "
            f"numba version), but it could not be loaded: {e}"
        ) from e

    _originals.update({n: getattr(rs, n) for n in KERNEL_NAMES})
    kernels: dict[str, Any] = {n: _NumbaKernel(n, _NUMBA_FNS[n]) for n in KERNEL_NAMES}
    detail = ", ".join(f"{k}={v}" for k, v in info.items())

    instr = _Instrumentation(NUMBA_IMPL)
    if instr.enabled:
        kernels = {n: instr.wrap(n, k) for n, k in kernels.items()}
        _wrap_forward(rs, instr)
        _instrumentation = instr
        detail += f", counters={instr.counters_dir}"
    for n, k in kernels.items():
        setattr(rs, n, k)
    _installed = True
    logger.info("AOT rejection sampler backend: %s (%s)", NUMBA_IMPL, detail)
    return NUMBA_IMPL


def uninstall() -> None:
    """Restore the upstream kernels (tests only)."""
    global _installed, _instrumentation, _original_forward
    if not _installed:
        return
    import vllm.v1.sample.rejection_sampler as rs

    for n, k in _originals.items():
        setattr(rs, n, k)
    if _original_forward is not None:
        rs.RejectionSampler.forward = _original_forward
        _original_forward = None
    if _instrumentation is not None:
        _instrumentation.flush()
    _originals.clear()
    _installed = False
    _instrumentation = None


def prewarm() -> float:
    """Compile/load every production Numba signature; returns seconds.

    No-op unless ``install()`` has run.  Inputs mirror
    the AOT production dtypes: int64 draft ids / argmax / recovered, int32
    cu / bonus / output, fp64 uniform, fp32 probs, fp32 or fp64 inv_q, and
    fp32 / int32 expand inputs (temperature, top_p / top_k).
    """
    if not _installed:
        return 0.0
    t0 = time.perf_counter()
    # Inputs come from a private generator: drawing from the global torch RNG
    # here would shift every later unseeded sample (the engine seeds the global
    # RNG before warm-up), so outputs would depend on whether prewarm ran.
    gen = torch.Generator().manual_seed(0)
    B, K, V = 2, 2, 64
    T = B * K
    cu = torch.tensor([K, 2 * K], dtype=torch.int32)
    draft = torch.zeros(T, dtype=torch.int64)
    out = torch.full((B, K + 1), -1, dtype=torch.int32)
    bonus = torch.zeros((B, 1), dtype=torch.int32)
    is_greedy = torch.tensor([True, False])
    uniform = torch.rand(T, dtype=torch.float64, generator=gen)
    rates = torch.ones(K, dtype=torch.float32)
    probs = torch.softmax(torch.randn(T, V, generator=gen), dim=-1)
    for x in (torch.ones(B, dtype=torch.float32), torch.ones(B, dtype=torch.int32)):
        _expand(x.new_empty(T), x, cu, 0, 1)
    for mask, syn in ((None, False), (is_greedy, False), (is_greedy, True)):
        _greedy(out, cu, draft, draft.clone(), bonus, mask, K, uniform, rates, syn)
    for q_dtype in (torch.float32, torch.float64):
        inv_q = torch.rand(B, V, dtype=q_dtype, generator=gen) + 1
        rec = torch.empty_like(draft)
        _recovered(
            rec, cu, draft, None, probs, inv_q, V, 8192, True, q_dtype == torch.float64
        )
    for syn in (False, True):
        _random(
            out,
            cu,
            draft,
            None,
            probs,
            bonus,
            rec,
            uniform,
            is_greedy,
            K,
            V,
            rates,
            True,
            syn,
        )
    dt = time.perf_counter() - t0
    logger.info("AOT rejection sampler numba prewarm: %.3f s", dt)
    return dt
