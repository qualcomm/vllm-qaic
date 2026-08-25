# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/rejection_sampler.py

"""QAIC PYT rejection-sampler helpers for speculative decoding.

QAIC PYT always installs the three local Qualcomm-Triton kernels for expand,
greedy rejection, and random rejection.  The legacy
``VLLM_QAIC_REJECTION_SAMPLER_IMPL=hybrid`` selector remains a compatibility
alias for that default.  The former PyTorch fallback has been removed.

vLLM's global Triton detection intentionally remains unchanged: it does not
recognize the Qualcomm backend and therefore exposes placeholder objects in
the upstream rejection-sampler module.  The QAIC installer patches only its
module globals with real, locally defined Qualcomm-Triton kernels.

``sample_recovered_tokens_kernel`` and ``generate_uniform_probs`` remain
PyTorch helpers.  QAIC does not support the upstream fp64 uniform path, and
the Qualcomm-Triton recovered-token implementation is not production-safe on
the current compiler backend.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)


class _GridLaunchable:
    """Make the retained PyTorch helper compatible with Triton launch syntax."""

    def __init__(self, fn):
        self._fn = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))

    def __getitem__(self, _grid: Any):
        fn = self._fn

        def _launch(*args, **kwargs):
            return fn(*args, **kwargs)

        return _launch

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _sample_recovered_tokens_kernel_pyt(
    output_token_ids_ptr: torch.Tensor,
    cu_num_draft_tokens_ptr: torch.Tensor,
    draft_token_ids_ptr: torch.Tensor,
    draft_probs_ptr,
    target_probs_ptr: torch.Tensor,
    inv_q_ptr: torch.Tensor,
    vocab_size: int,
    BLOCK_SIZE: int = 8192,
    NO_DRAFT_PROBS: bool = True,
    USE_FP64_GUMBEL: bool = False,
) -> None:
    """Generate QAIC-compatible recovered tokens for random rejection.

    The QAIC status-26 workaround intentionally performs the scalar scatter
    on the host.  The recovered IDs are copied back as contiguous QAIC int32
    tensors before the local Qualcomm-Triton random-rejection kernel consumes
    them.
    """
    del vocab_size, BLOCK_SIZE, USE_FP64_GUMBEL
    batch_size = int(cu_num_draft_tokens_ptr.shape[0])
    device = cu_num_draft_tokens_ptr.device

    previous = torch.zeros(
        batch_size,
        dtype=cu_num_draft_tokens_ptr.dtype,
        device=device,
    )
    previous[1:] = cu_num_draft_tokens_ptr[:-1]
    start_indices = previous.to(torch.int64)
    end_indices = cu_num_draft_tokens_ptr.to(torch.int64)

    for request_index in range(batch_size):
        start_index = int(start_indices[request_index].item())
        end_index = int(end_indices[request_index].item())
        if end_index == start_index:
            continue

        token_slice = slice(start_index, end_index)
        if NO_DRAFT_PROBS:
            probabilities = target_probs_ptr[token_slice].clone()
            draft_tokens = draft_token_ids_ptr[token_slice].to(torch.int64)
            draft_tokens = draft_tokens.view(-1, 1)
            if device.type == "qaic":
                probabilities = probabilities.cpu()
                probabilities.scatter_(1, draft_tokens.cpu(), 0.0)
                probabilities = probabilities.to(device=device)
            else:
                probabilities.scatter_(1, draft_tokens, 0.0)
        else:
            probabilities = torch.clamp(
                target_probs_ptr[token_slice] - draft_probs_ptr[token_slice],
                min=0.0,
            )

        scores = probabilities * inv_q_ptr[request_index].unsqueeze(0)
        recovered = scores.argmax(dim=-1).to(torch.int32)
        output_token_ids_ptr[start_index:end_index].copy_(recovered)


sample_recovered_tokens_kernel = _GridLaunchable(_sample_recovered_tokens_kernel_pyt)


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list,
    generators: dict,
    device: torch.device,
) -> torch.Tensor:
    """Generate fp32 uniform samples because QAIC does not support fp64."""
    uniform_probs = torch.rand(
        (num_tokens,),
        dtype=torch.float32,
        device=device,
    )
    start_index = 0
    for request_index, num_drafts in enumerate(num_draft_tokens):
        if num_drafts == 0:
            continue
        end_index = start_index + num_drafts
        generator = generators.get(request_index)
        if generator is not None:
            uniform_probs[start_index:end_index].uniform_(generator=generator)
        start_index = end_index
    return uniform_probs


_REJECTION_SAMPLER_IMPL_ENV = "VLLM_QAIC_REJECTION_SAMPLER_IMPL"
_HYBRID_IMPL = "hybrid"
_PYTORCH_IMPL = "pytorch"
_TRITON_IMPL = "triton"

_shim_installed = False
_shim_mode: str | None = None


def _selected_implementation() -> str:
    """Resolve the only supported QAIC PYT kernel implementation."""
    implementation = os.environ.get(_REJECTION_SAMPLER_IMPL_ENV, "").strip().lower()
    if implementation in {"", _HYBRID_IMPL}:
        return _TRITON_IMPL
    if implementation == _PYTORCH_IMPL:
        raise RuntimeError(
            f"{_REJECTION_SAMPLER_IMPL_ENV}=pytorch is no longer supported: "
            "the QAIC PYT PyTorch fallback for expand, greedy rejection, and "
            "random rejection was removed. Unset the variable or set it to "
            "'hybrid' to use the Qualcomm-Triton default."
        )
    raise RuntimeError(
        f"{_REJECTION_SAMPLER_IMPL_ENV} must be unset or '{_HYBRID_IMPL}'; "
        f"got {implementation!r}."
    )


def install() -> None:
    """Install QAIC PYT rejection-sampler helpers before LLM construction."""
    global _shim_installed, _shim_mode
    implementation = _selected_implementation()
    if _shim_installed:
        if implementation != _shim_mode:
            raise RuntimeError(
                "The QAIC rejection-sampler implementation is already installed "
                f"as {_shim_mode!r}; changing {_REJECTION_SAMPLER_IMPL_ENV} "
                "within one Python process is unsupported. Use an isolated "
                "subprocess for each implementation."
            )
        return

    try:
        from vllm_qaic.v1.sample.rejection_sampler_triton import (
            get_qaic_triton_kernels,
        )

        triton_kernels = get_qaic_triton_kernels()
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        raise RuntimeError(
            "QAIC PYT rejection sampling requires Qualcomm Triton with "
            "qcom_hexagon_backend. Install and verify the QAIC Triton wheel "
            "using docs/qaic/build_hexagon_triton_backend.md."
        ) from exc

    import vllm.v1.sample.rejection_sampler as rejection_sampler

    rejection_sampler.expand_kernel = triton_kernels.expand_kernel
    rejection_sampler.rejection_greedy_sample_kernel = (
        triton_kernels.rejection_greedy_sample_kernel
    )
    rejection_sampler.rejection_random_sample_kernel = (
        triton_kernels.rejection_random_sample_kernel
    )
    rejection_sampler.sample_recovered_tokens_kernel = sample_recovered_tokens_kernel
    rejection_sampler.generate_uniform_probs = generate_uniform_probs

    _shim_installed = True
    _shim_mode = implementation
    logger.info(
        "vllm_qaic: Qualcomm-Triton rejection-sampler kernels installed for "
        "QAIC PYT (ngram/suffix SpD enabled)."
    )
