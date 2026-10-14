# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# Patch: register QAIC (OOT platform) in vLLM's FP8 kernel selector dicts.
# QaicFP8BlockKernel: dequantizes FP8 block-quant weights to fp16 at load time
# and executes as a plain fp16 matmul (W8A16 via dequant-at-load).

import torch
from vllm.platforms import PlatformEnum
import vllm.model_executor.kernels.linear as _linear_mod


def _patch_fp8_kernel_dicts():
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
            PerTensorTorchFP8ScaledMMLinearKernel,
            ChannelWiseTorchFP8ScaledMMLinearKernel,
        )
        from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
            CPUFp8BlockScaledMMKernel,
        )
        from vllm.model_executor.layers.quantization.utils.layer_utils import (
            replace_parameter,
        )
    except ImportError:
        return

    class QaicPerTensorFP8Kernel(PerTensorTorchFP8ScaledMMLinearKernel):
        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

    class QaicChannelWiseFP8Kernel(ChannelWiseTorchFP8ScaledMMLinearKernel):
        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

    class QaicFP8BlockKernel(CPUFp8BlockScaledMMKernel):
        @classmethod
        def is_supported(cls, compute_capability=None):
            return True, None

        @classmethod
        def can_implement(cls, c):
            return True, None

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            params = self._get_layer_params(layer)
            w = params.weight  # FP8 tensor
            w_scale = (
                params.weight_scale_inv
                if params.weight_scale_inv is not None
                else params.weight_scale
            )

            # Dequantize block-FP8 → fp16 eagerly so apply_weights is a plain matmul.
            # w: [M, K] in fp8_e4m3, w_scale: [M/bs, K/bs] in bfloat16
            bs_m, bs_k = self.weight_group_shape  # e.g. (128, 128)
            w_f16 = w.to(torch.float16)
            # Expand scale to full weight shape via repeat_interleave
            scale_f16 = w_scale.to(torch.float16)
            scale_exp = scale_f16.repeat_interleave(bs_m, dim=0).repeat_interleave(
                bs_k, dim=1
            )
            # Trim to exact weight shape in case of padding
            scale_exp = scale_exp[: w_f16.shape[0], : w_f16.shape[1]]
            w_dequant = w_f16 * scale_exp

            replace_parameter(
                layer,
                params.WEIGHT,
                torch.nn.Parameter(w_dequant, requires_grad=False),
            )
            # Zero out the scale so apply_weights doesn't try to use it
            scale_attr = (
                params.WEIGHT_SCALE_INV
                if params.weight_scale_inv is not None
                else params.WEIGHT_SCALE
            )
            replace_parameter(
                layer,
                scale_attr,
                torch.nn.Parameter(
                    torch.ones(1, dtype=torch.float16), requires_grad=False
                ),
            )

        def apply_weights(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
            **kwargs,
        ) -> torch.Tensor:
            # Weight is already dequantized to fp16; just do a plain matmul.
            params = self._get_layer_params(layer)
            w = params.weight  # fp16 dequantized
            x_2d = x.reshape(-1, x.shape[-1]) if x.dim() > 2 else x
            out = torch.nn.functional.linear(x_2d.to(w.dtype), w, bias)
            return out.reshape(x.shape[:-1] + (out.size(-1),)) if x.dim() > 2 else out

    oot = PlatformEnum.OOT

    for attr, fallback in (
        ("_POSSIBLE_FP8_KERNELS", [QaicPerTensorFP8Kernel, QaicChannelWiseFP8Kernel]),
        ("_POSSIBLE_FP8_BLOCK_KERNELS", [QaicFP8BlockKernel]),
        (
            "_POSSIBLE_WFP8A16_KERNELS",
            [QaicPerTensorFP8Kernel, QaicChannelWiseFP8Kernel],
        ),
    ):
        d = getattr(_linear_mod, attr, None)
        if d is not None and oot not in d:
            d[oot] = fallback


_patch_fp8_kernel_dicts()
