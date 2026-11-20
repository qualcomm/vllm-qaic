# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/layers/quantization/input_quant_fp8.py
"""QAIC override of vLLM's ``QuantFP8`` activation quantizer.

Upstream's native path computes the scaled activations in fp32 and casts them
straight to fp8.  QAIC's ``IntraDeviceCopy`` cannot convert fp32 -> fp8
(``run failed with status 26``); the runtime then falls back to the host, which
yields correct values but logs a ``QAIC_ERROR`` for every cast and costs a
device round trip.  fp16 -> fp8 is supported on device and matches the CPU
cast exactly, so this override inserts an fp16 hop before the fp8 cast.

The scaled values are clamped to the fp8 range (+-448 for e4m3) first, which is
representable in fp16, so the hop cannot overflow.  The hop does double-round:
a small fraction of elements (~0.4% on random data) can land one fp8 step away
from a direct fp32 -> fp8 cast.
"""

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    _FP8_DTYPE,
    _FP8_MAX,
    _FP8_MIN,
    _FP8_MIN_SCALING_FACTOR,
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    group_broadcast,
    prep_scale_for_group_broadcast,
)


def _to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Clamp to the fp8 range and cast to fp8 via fp16 (fp32 -> fp8 is
    unsupported by QAIC's IntraDeviceCopy)."""
    return x.clamp(_FP8_MIN, _FP8_MAX).to(torch.float16).to(_FP8_DTYPE)


class QAicQuantFP8(QuantFP8):
    """``QuantFP8`` whose fp8 cast goes through fp16.

    ``QuantFP8`` only defines ``forward_native`` and, for OOT platforms,
    ``CustomOp.forward_oot`` delegates to it, so overriding ``forward_native``
    covers both the enabled (``forward_oot``) and disabled
    (``maybe_compile(forward_native)``) dispatch paths.
    """

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
    ):
        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"
            return self._quantize_group_native(x)

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
        else:
            scale = prep_scale_for_group_broadcast(scale, x, self.group_shape)

        out = (
            x.to(torch.float32)
            * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
        )
        out = _to_fp8(out)

        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _quantize_group_native(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales_raw = absmax / _FP8_MAX
        if self.use_ue8m0:
            scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
        scales = (scales_raw).clamp(min=_FP8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = _to_fp8(x_scaled)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales
