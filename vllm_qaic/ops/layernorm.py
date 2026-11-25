# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/layers/layernorm.py

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm,
    RMSNorm,
    RMSNormGated,
)

from vllm_qaic._custom_ops import rms_norm_hexagon as _call_hexagon_rms_norm
from vllm_qaic.ops._triton_flags import triton_op_enabled
from vllm.model_executor.determinism.batch_invariant import (
    rms_norm_batch_invariant,
)

from vllm.models.minimax_m3.amd.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
    rmsnorm_fn,
)

triton_en = triton_op_enabled("RMS_NORM")
triton_gated_en = triton_op_enabled("RMS_NORM_GATED")


def _fused_add_rms_norm_supported(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor
) -> bool:
    """qaic::add_rms_norm needs fp32/fp16/bf16 tensors sharing one dtype."""
    return (
        x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.dtype == residual.dtype == weight.dtype
        and x.shape == residual.shape
        and x.shape[-1] == weight.shape[0]
    )


class QAicRMSNorm(RMSNorm):
    """RMSNorm with a fast-path to the fused NSP add+norm kernel.

    When `residual` is provided ``torch.ops.qaic.add_rms_norm`` fuses the
    residual add and the RMS normalization into a single NSP dispatch,
    avoiding a round-trip to DDR between the two ops. The ``rms_norm_hexagon``
    kernel remains as the fallback for configurations the
    fused op does not cover.

    When ``VLLM_QAIC_TRITON_RMS_NORM`` is set the no-residual path instead
    dispatches to vLLM's Triton ``rms_norm_batch_invariant`` kernel. The
    residual path stays on the Hexagon kernel regardless: vLLM has no
    pure-Triton fused add+RMS-norm (its residual launcher falls back to the
    CUDA ``_C`` op).
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            if (
                self.has_weight
                and self.variance_size_override is None
                and _fused_add_rms_norm_supported(x, residual, self.weight)
            ):
                normed, new_residual = torch.ops.qaic.add_rms_norm(
                    x,
                    [x.shape[-1]],
                    self.weight,
                    self.variance_epsilon,
                    residual,
                    False,
                )
                return normed, new_residual
            normed, new_residual = _call_hexagon_rms_norm(
                residual, x, self.weight, self.variance_epsilon
            )
            return normed, new_residual

        # No residual: plain normalization only. Like upstream, a norm built
        # with has_weight=False gets weight=None (no per-channel multiply).
        weight = self.weight if self.pass_weight else None
        if triton_en:
            return rms_norm_batch_invariant(x, weight, self.variance_epsilon)

        return F.rms_norm(x, [self.hidden_size], weight, self.variance_epsilon)


class QAicGemmaRMSNorm(GemmaRMSNorm):
    """RMS normalization for Gemma with an optional Triton path.

    Gemma uses a shifted weight convention: the learned parameter represents
    (gamma - 1), so the effective scale is (1 + weight).
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if triton_en:
            if residual is None:
                return gemma_rmsnorm(x, self.weight, self.variance_epsilon)
            return gemma_fused_add_rmsnorm(
                x, residual, self.weight, self.variance_epsilon
            )

        if residual is not None:
            if _fused_add_rms_norm_supported(x, residual, self.weight):
                normed, new_residual = torch.ops.qaic.add_rms_norm(
                    x, [x.shape[-1]], self.weight, self.variance_epsilon, residual, True
                )
                return normed, new_residual
            return self.forward_native(x, residual)

        return F.rms_norm(
            x, self.weight.shape, 1.0 + self.weight, self.variance_epsilon
        )


class QAicRMSNormGated(RMSNormGated):
    """RMS Normalization with optional gating.
    Also supports optional gating via Triton.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Group norm not handled here; delegate to the parent's native path
        if self.group_size is not None:
            return super().forward_native(x, z)

        if triton_gated_en:
            return rmsnorm_fn(
                x,
                self.weight,
                self.bias,
                z=z,
                eps=self.eps,
                group_size=self.group_size,
                norm_before_gate=self.norm_before_gate,
                activation=self.activation,
            )

        # norm_before_gate=False: gate first, then normalize (Mamba2 / Hawk style)
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)

        out = F.rms_norm(x, [self.hidden_size], self.weight, self.eps)

        # norm_before_gate=True: normalize first, then gate (some SSM variants)
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)

        return out
