# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/ops/topk_topp_sampler.py

"""Gate the sampler's Triton top-k/top-p kernel behind VLLM_QAIC_TRITON_TOPK_TOPP.

Upstream ``apply_top_k_top_p`` picks its implementation implicitly, from
``HAS_TRITON`` plus a CUDA-tuned batch-size heuristic: the Triton kernel at batch
>= 8, the PyTorch sort below it. This replaces the function with a version where the
flag is the only thing that decides: ``VLLM_QAIC_TRITON_TOPK_TOPP=1`` routes to
``apply_top_k_top_p_triton`` at any batch size, and anything else routes to the
PyTorch implementation. The flag is resolved per call through
``triton_op_enabled()``, the same infrastructure as the per-op CustomOp flags.
"""

import torch

from vllm.v1.sample.ops import topk_topp_sampler
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_qaic.ops._triton_flags import triton_op_enabled


def _qaic_apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k/top-p masks, on Triton only when the QAIC flag is set.

    The logits tensor may be updated in-place.
    """
    if p is None and k is None:
        return logits

    if triton_op_enabled("TOPK_TOPP"):
        from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

        return apply_top_k_top_p_triton(logits, k, p)

    return apply_top_k_top_p_pytorch(logits, k, p)


topk_topp_sampler.apply_top_k_top_p = _qaic_apply_top_k_top_p
