# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/rejection_sampler.py

"""QAIC PYT rejection-sampler shim for speculative decoding.

Upstream's ``@triton.jit`` rejection-sampler kernels (``expand_kernel``,
``rejection_greedy_sample_kernel``, ``rejection_random_sample_kernel`` and
``sample_recovered_tokens_kernel``) launch directly on QAIC tensors through the
Qualcomm Triton backend, so none of them is replaced.  The single remaining
difference is ``generate_uniform_probs``: upstream draws the uniforms in
float64, which the QAIC backend does not support
(``RuntimeError: Unsupported datatype for QAIC: Double``).  This shim rebinds it
to a float32 version.

``vllm_qaic.patch.patch_qaic_triton_import`` sets ``HAS_TRITON`` when the
Qualcomm backend is present; the Qualcomm Triton wheel with
``qcom_hexagon_backend`` must be installed.
"""

from __future__ import annotations

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)


def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list,
    generators: dict,
    device: torch.device,
) -> torch.Tensor:
    """Generate fp32 uniform samples because QAIC does not support fp64.

    Upstream uses fp64 because fp32 ``torch.rand`` can return exactly 0.0,
    which would accept a draft token with zero target probability. Clamp to
    the smallest positive fp32 value instead.
    """
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
    return uniform_probs.clamp_(min=torch.finfo(torch.float32).tiny)


_shim_installed = False


def _require_qualcomm_triton() -> None:
    try:
        from triton.backends import backends
    except ImportError as exc:
        raise RuntimeError(
            "QAIC PYT rejection sampling requires Qualcomm Triton with "
            "qcom_hexagon_backend, but triton is not importable. Install the "
            "Qualcomm Triton wheel."
        ) from exc
    if "qcom_hexagon_backend" not in backends:
        raise RuntimeError(
            "QAIC PYT rejection sampling requires Qualcomm Triton with "
            "qcom_hexagon_backend. Install the Qualcomm Triton wheel."
        )


def install() -> None:
    """Install the QAIC PYT rejection-sampler shim before LLM construction."""
    global _shim_installed
    if _shim_installed:
        return

    _require_qualcomm_triton()

    import vllm.v1.sample.rejection_sampler as rejection_sampler

    rejection_sampler.generate_uniform_probs = generate_uniform_probs

    _shim_installed = True
    logger.info(
        "vllm_qaic: rejection sampler uses upstream Triton kernels with fp32 "
        "uniform probabilities on QAIC PYT (ngram/suffix SpD enabled)."
    )
