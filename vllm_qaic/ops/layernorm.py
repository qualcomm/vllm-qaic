# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/layers/layernorm.py

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm, RMSNormGated

from vllm_qaic._custom_ops import rms_norm_hexagon as _call_hexagon_rms_norm


class QAicRMSNorm(RMSNorm):
    """RMSNorm with a fast-path to the Hexagon NSP fused add+norm kernel.

    When `residual` is provided the kernel fuses the residual add and the
    RMS normalization into a single NSP dispatch, avoiding a round-trip to
    DDR between the two ops.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            normed, new_residual = _call_hexagon_rms_norm(
                residual, x, self.weight, self.variance_epsilon
            )
            return normed, new_residual

        # No residual: plain normalization only
        return F.rms_norm(x, [self.hidden_size], self.weight, self.variance_epsilon)


class QAicGemmaRMSNorm(GemmaRMSNorm):
    """RMS normalization for Gemma.

    Gemma uses a shifted weight convention: the learned parameter represents
    (gamma - 1), so the effective scale is (1 + weight).  No NSP kernel path
    here because the weight offset makes it non-trivially different from the
    standard RMS norm kernel interface.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            normed, new_residual = _call_hexagon_rms_norm(
                residual, x, self.weight, self.variance_epsilon
            )
            return normed, new_residual

        return F.rms_norm(
            x, [self.hidden_size], 1.0 + self.weight, self.variance_epsilon
        )


class QAicRMSNormGated(RMSNormGated):
    """RMS Normalization with optional gating.

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

        # norm_before_gate=False: gate first, then normalize (Mamba2 / Hawk style)
        if z is not None and not self.norm_before_gate:
            x = x * F.silu(z)

        out = F.rms_norm(x, [self.hidden_size], self.weight, self.eps)

        # norm_before_gate=True: normalize first, then gate (some SSM variants)
        if z is not None and self.norm_before_gate:
            out = out * F.silu(z)

        return out
