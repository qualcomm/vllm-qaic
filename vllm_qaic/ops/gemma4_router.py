# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Register QAIC Gemma4 routing override.

Upstream ``Gemma4MoE.routing_function`` calls the Triton kernel
``gemma4_fused_routing_kernel_triton`` only on CUDA-alike/XPU platforms and
otherwise calls ``gemma4_routing_function_torch``, so on QAIC it always takes
the torch path. The closure resolves that name as a module global at call time,
so rebinding it reroutes every Gemma4 MoE layer:

  VLLM_QAIC_TRITON_GEMMA4_ROUTER=1  -> upstream's Triton routing kernel
  otherwise, supported inputs       -> QAIC NSP ``gemma4_topk`` kernel
  otherwise                         -> the original torch function
"""

from __future__ import annotations

import torch

from vllm_qaic.ops._triton_flags import triton_op_enabled

triton_en = triton_op_enabled("GEMMA4_ROUTER")

_PATCH_APPLIED_ATTR = "_qaic_gemma4_router_patch_applied"
_ORIGINAL_ROUTING_ATTR = "_qaic_original_gemma4_routing_function_torch"


def _get_qaic_ops():
    try:
        from vllm_qaic import _custom_ops as qaic_ops

        return qaic_ops
    except Exception:
        return None


def _supports_qaic_gemma4_topk(
    gating_output: torch.Tensor,
    topk: int,
    per_expert_scale: torch.Tensor,
) -> bool:
    return (
        gating_output.device.type == "qaic"
        and gating_output.dtype == torch.float32
        and gating_output.dim() == 2
        and topk > 0
        and topk <= 32
        and topk <= gating_output.shape[-1]
        and gating_output.shape[-1] <= 1024
        and per_expert_scale.numel() == gating_output.shape[-1]
    )


def register_qaic_gemma4_router() -> None:
    """Patch Gemma4's custom routing function to dispatch to QAIC when supported."""
    try:
        from vllm.model_executor.models import gemma4
    except Exception:
        return
    if not hasattr(gemma4, "gemma4_routing_function_torch"):
        return

    if getattr(gemma4, _PATCH_APPLIED_ATTR, False):
        return

    setattr(
        gemma4,
        _ORIGINAL_ROUTING_ATTR,
        gemma4.gemma4_routing_function_torch,
    )

    def _patched_gemma4_routing_function_torch(
        gating_output: torch.Tensor,
        topk: int,
        per_expert_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if triton_en:
            return gemma4.gemma4_fused_routing_kernel_triton(
                gating_output, topk, per_expert_scale
            )

        qaic_ops = _get_qaic_ops()
        if qaic_ops is not None and _supports_qaic_gemma4_topk(
            gating_output, topk, per_expert_scale
        ):
            return qaic_ops.gemma4_topk(gating_output, topk, per_expert_scale)

        original = getattr(gemma4, _ORIGINAL_ROUTING_ATTR)
        return original(gating_output, topk, per_expert_scale)

    gemma4.gemma4_routing_function_torch = _patched_gemma4_routing_function_torch
    setattr(gemma4, _PATCH_APPLIED_ATTR, True)
