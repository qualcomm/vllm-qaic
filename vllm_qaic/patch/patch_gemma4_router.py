# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/models/gemma4.py

"""Gate Gemma 4's Triton MoE routing kernel behind VLLM_QAIC_TRITON_GEMMA4_ROUTER.

Upstream ``Gemma4MoE.routing_function`` calls the Triton kernel only on
CUDA-alike/XPU platforms and otherwise calls ``gemma4_routing_function_torch``.
On QAIC it therefore always takes the torch path. This replaces that torch
function with a wrapper where the flag is the only thing that decides:
``VLLM_QAIC_TRITON_GEMMA4_ROUTER=1`` routes to upstream's
``gemma4_fused_routing_kernel_triton``, anything else to the original torch
function. The flag is resolved per call through ``triton_op_enabled()``.
"""

import torch

from vllm.model_executor.models import gemma4

from vllm_qaic.ops._triton_flags import triton_op_enabled

# Saved before rebinding below: the wrapper replaces this module attribute.
_gemma4_routing_function_torch = gemma4.gemma4_routing_function_torch


def _qaic_gemma4_routing_function(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma 4 routing, on Triton only when the QAIC flag is set."""
    if triton_op_enabled("GEMMA4_ROUTER"):
        return gemma4.gemma4_fused_routing_kernel_triton(
            gating_output, topk, per_expert_scale
        )

    return _gemma4_routing_function_torch(gating_output, topk, per_expert_scale)


gemma4.gemma4_routing_function_torch = _qaic_gemma4_routing_function
