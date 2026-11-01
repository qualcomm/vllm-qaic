# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared helpers for the rejection-sampler Numba-vs-Triton parity suites.

Importable WITHOUT triton: the module only captures whatever objects
``vllm.v1.sample.rejection_sampler`` exposes for the four kernels (real
``JITFunction`` objects with triton-cpu, plain functions under vLLM's
``TritonPlaceholder`` otherwise).  Call ``assert_triton_originals()`` before
launching Triton.

Import this module BEFORE anything installs the Numba backend, or while it is
installed (the originals are then read back from
``rejection_sampler_numba._originals``).

Two Numba entry points take the upstream call-site layout
``kernel[grid](*args, **kwargs)``:

* ``production_wrappers()`` -- the objects that
  ``vllm_qaic.v1.sample.rejection_sampler_numba.install()`` puts into the
  upstream module globals (installed, read back, then uninstalled).
* ``RAW_LAUNCHERS`` -- adapters onto the raw kernels in
  ``vllm_qaic.v1.sample.numba_rejection_kernels`` (``expand_numba`` ...).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

os.environ.setdefault("TRITON_CPU_BACKEND", "1")

import torch  # noqa: E402
from vllm.v1.sample import rejection_sampler as rs  # noqa: E402

KERNEL_NAMES = (
    "expand_kernel",
    "rejection_greedy_sample_kernel",
    "rejection_random_sample_kernel",
    "sample_recovered_tokens_kernel",
)

PROD_MODULE = "vllm_qaic.v1.sample.rejection_sampler_numba"

# Upstream greedy kernel + SYNTHETIC_MODE + int64 draft ids does not compile on
# triton-cpu (dynamic ternary mixes int64 draft id with the int32 argmax).
KNOWN_TRITON_SYNTHETIC_INT64 = "inconsistent types int64 and int32"


def _capture_originals() -> dict[str, Any]:
    kernels = {name: getattr(rs, name) for name in KERNEL_NAMES}
    try:
        from vllm_qaic.v1.sample import rejection_sampler_numba as rsn

        if rsn._installed:
            kernels.update(
                {n: rsn._originals[n] for n in KERNEL_NAMES if n in rsn._originals}
            )
    except ImportError:
        pass
    return kernels


ORIGINAL_KERNELS: dict[str, Any] = _capture_originals()


def assert_triton_originals(check_globals: bool = True) -> None:
    """Fail unless the captured (and, optionally, current) kernels are Triton."""
    from triton.runtime.jit import JITFunction

    bad = {
        n: type(k).__name__
        for n, k in ORIGINAL_KERNELS.items()
        if not isinstance(k, JITFunction)
    }
    if bad:
        raise RuntimeError(
            "upstream rejection-sampler kernels are not triton JITFunctions "
            f"(captured after a Numba install, or vLLM TritonPlaceholder): {bad}"
        )
    if check_globals:
        stale = [n for n in KERNEL_NAMES if getattr(rs, n) is not ORIGINAL_KERNELS[n]]
        if stale:
            raise RuntimeError(
                "vllm.v1.sample.rejection_sampler globals were left replaced by "
                f"an earlier test (Numba installed?): {stale}"
            )


_WRAPPER_CACHE: dict[str, Any] = {}


def production_wrappers() -> dict[str, Any] | None:
    """Return the production Numba wrapper objects, leaving ``rs`` untouched.

    Installs via the no-arg ``install()``, reads the module globals, and
    ``uninstall()``s again unless the backend was already installed by the
    caller.  Returns ``None`` when the production module does not exist.
    """
    import importlib

    if _WRAPPER_CACHE:
        return dict(_WRAPPER_CACHE)
    try:
        mod = importlib.import_module(PROD_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name and PROD_MODULE.startswith(exc.name):
            return None
        raise
    was_installed = mod._installed
    try:
        impl = mod.install()
        assert impl == "numba", impl
        wrappers = {name: getattr(rs, name) for name in KERNEL_NAMES}
    finally:
        if not was_installed:
            mod.uninstall()
    leaked = [n for n in KERNEL_NAMES if wrappers[n] is ORIGINAL_KERNELS[n]]
    if leaked:
        raise RuntimeError(f"{PROD_MODULE}.install() left upstream kernels: {leaked}")
    _WRAPPER_CACHE.update(wrappers)
    return wrappers


def clone_args(args: list[Any]) -> list[Any]:
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


# --------------------------------------------------------------------------
# Raw-kernel adapters (upstream layout -> numba_rejection_kernels)
# --------------------------------------------------------------------------
_EMPTY_F64 = torch.zeros(0, dtype=torch.float64)
_EMPTY_F32 = torch.zeros(0, dtype=torch.float32)


def _or(t, empty):
    return empty if t is None else t


def _nrk():
    from vllm_qaic.v1.sample import numba_rejection_kernels as nrk

    return nrk


def _raw_expand(grid, args, kwargs):
    out, x, cu, replace_from, replace_to = args
    _nrk().expand_numba(out, x, cu, replace_from, replace_to)


def _raw_greedy(grid, args, kwargs):
    out, cu, draft, argmax, bonus, is_greedy, max_spec_len, uniform, rates = args
    _nrk().greedy_numba(
        out,
        cu,
        draft,
        argmax,
        bonus,
        is_greedy,
        max_spec_len,
        _or(uniform, _EMPTY_F64),
        _or(rates, _EMPTY_F32),
        kwargs["SYNTHETIC_MODE"],
    )


def _raw_random(grid, args, kwargs):
    (
        out,
        cu,
        draft,
        dprobs,
        tprobs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        max_spec_len,
        vocab,
        rates,
    ) = args
    _nrk().random_numba(
        out,
        cu,
        draft,
        dprobs,
        tprobs,
        bonus,
        recovered,
        uniform,
        is_greedy,
        max_spec_len,
        vocab,
        _or(rates, _EMPTY_F32),
        kwargs["NO_DRAFT_PROBS"],
        kwargs["SYNTHETIC_MODE"],
    )


def _raw_recovered(grid, args, kwargs):
    out, cu, draft, dprobs, tprobs, inv_q, vocab = args[:7]
    _nrk().recovered_numba(
        out, cu, draft, dprobs, tprobs, inv_q, vocab, kwargs["NO_DRAFT_PROBS"]
    )


RAW_LAUNCHERS: dict[str, Callable[[tuple, list, dict], None]] = {
    "expand_kernel": _raw_expand,
    "rejection_greedy_sample_kernel": _raw_greedy,
    "rejection_random_sample_kernel": _raw_random,
    "sample_recovered_tokens_kernel": _raw_recovered,
}


def launch_triton(name: str, grid: tuple, args: list, kwargs: dict) -> None:
    ORIGINAL_KERNELS[name][tuple(grid)](*args, **kwargs)


def launch_raw(name: str, grid: tuple, args: list, kwargs: dict) -> None:
    RAW_LAUNCHERS[name](tuple(grid), args, kwargs)


def launch_wrapper(
    wrappers: dict[str, Any], name: str, grid: tuple, args: list, kwargs: dict
) -> None:
    wrappers[name][tuple(grid)](*args, **kwargs)
