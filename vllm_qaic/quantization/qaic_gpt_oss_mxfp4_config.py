# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC (eager-mode) MXFP4 support for GPT-OSS checkpoints.

Upstream's ``GptOssMxfp4MoEMethod`` (vllm.model_executor.layers.quantization.
mxfp4) dispatches to a CUDA/AMD/Triton MoE kernel selected by
``select_mxfp4_moe_backend``. QAIC has no branch there, so the backend always
resolves to ``Mxfp4MoeBackend.NONE`` and upstream's method silently no-ops in
``process_weights_after_loading`` and then crashes in ``apply``/
``apply_monolithic`` (``assert self.moe_kernel is not None``).

This module registers a QAIC-specific ``QuantizationConfig`` under the name
``"gpt_oss_mxfp4"`` whose MoE method keeps the weights packed and runs the
per-expert matmuls through QAIC's native ``qaic::mxfp4_mm`` op, which decodes
MXFP4 into VTCM tile-by-tile on device. That preserves the ~4x weight
footprint/bandwidth reduction which is the whole point of MXFP4 — a
dequantize-to-dense-at-load-time approach throws it away.

Routing, activation, bias and the weighted reduce are inherited from
``QAicUnquantizedFusedMoEMethod._forward_per_expert``; only its
``_gate_up_matmul``/``_down_matmul`` hooks are overridden here.

The in-tree fused HVX MoE kernel is deliberately bypassed: it assumes dense
fp16 weight rows (no scale operand, no packed-format awareness) and its Python
wrapper casts the weight tensors to ``x.dtype``, which would silently
reinterpret packed MXFP4 bytes as fp16 values. ``forward_oot`` is therefore
overridden to always take the per-expert path.

Only enabled in eager mode (``is_aot == False``); see
``QaicPlatform.pre_register_and_update``.
"""

import torch

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import biased_moe_quant_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from vllm_qaic.ops.unquantized_fused_moe_method import QAicUnquantizedFusedMoEMethod
from vllm_qaic.utils import QAIC_GPT_OSS_MXFP4_METHOD

_MXFP4_BLOCK_SIZE = 32


@register_quantization_config(QAIC_GPT_OSS_MXFP4_METHOD)
class QaicGptOssMxfp4Config(QuantizationConfig):
    """QAIC eager-mode MXFP4 config for GPT-OSS checkpoints."""

    def get_name(self) -> str:
        # Must be exactly "gpt_oss_mxfp4": FusedMoE.weight_loader() has a
        # hardcoded string check on quant_config.get_name() to special-case
        # gpt-oss's combined-experts checkpoint layout.
        return "gpt_oss_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # Unused on QAIC: current_platform.get_device_capability() returns
        # None, so VllmConfig._get_quantization_config() skips this check.
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "QaicGptOssMxfp4Config":
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None, **kwargs
    ) -> str | None:
        # Only take over from upstream's GptOssMxfp4Config in QAIC eager
        # mode. In AOT mode, fall through so gpt_oss_mxfp4 stays unsupported.
        if current_platform.is_aot_inference():
            return None
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        if getattr(hf_config, "model_type", None) != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            return QaicGptOssMxfp4MoEMethod(layer.moe_config)
        if isinstance(layer, Attention):
            return None
        return None


class QaicGptOssMxfp4MoEMethod(QAicUnquantizedFusedMoEMethod):
    """QAIC MoE method for GPT-OSS MXFP4 checkpoints.

    Weights stay packed (uint8, MXFP4) exactly as loaded, matching both the
    checkpoint's on-disk layout and the operand layout ``qaic::mxfp4_mm``
    expects: ``[N, K/2]`` weights plus ``[N, K/32]`` E8M0 block scales, per
    expert. Compute reuses ``QAicUnquantizedFusedMoEMethod``'s per-expert loop
    with both matmuls redirected to that native op.
    """

    def __init__(self, moe: FusedMoEConfig):
        # QAicUnquantizedFusedMoEMethod has no __init__ of its own, so this
        # resolves to UnquantizedFusedMoEMethod.__init__, which calls
        # select_unquantized_moe_backend() and (since QAIC is an
        # out-of-tree platform) sets self.unquantized_backend = OOT. That is
        # what makes the inherited CustomOp dispatch route apply()->forward()
        # to forward_oot(). We additionally set weight_dtype, which
        # gpt_oss.py reads to select its MXFP4 weight-loading path.
        super().__init__(moe)
        self.weight_dtype = "gpt_oss_mxfp4"

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> tuple[int, int]:
        # gpt_oss.py's _load_weights_mxfp4() shards the checkpoint's
        # intermediate dimension by rounding the per-rank block count up
        # (cdiv) before multiplying back by the MXFP4 block size, so the
        # loaded weight slice hitting weight_loader() is padded to a
        # multiple of the block size. create_weights() must allocate that
        # same padded size or FusedMoE.weight_loader()'s direct copy_()
        # into param.data[:, :dim1, :dim2] raises a shape mismatch.
        hidden_size, intermediate_size_per_partition = super().maybe_roundup_sizes(
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            act_dtype=act_dtype,
            moe_parallel_config=moe_parallel_config,
        )
        intermediate_size_per_partition = round_up(
            intermediate_size_per_partition, _MXFP4_BLOCK_SIZE
        )
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.params_dtype = params_dtype
        layer.num_experts = num_experts

        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        w13_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // _MXFP4_BLOCK_SIZE,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _MXFP4_BLOCK_SIZE,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if self.moe.has_bias:
            # Unlike upstream (bf16-only), QAIC has no bf16 tensor support at
            # all, so biases are allocated directly in params_dtype (the
            # checkpoint's bf16 values are cast on copy by weight_loader()).
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

    # process_weights_after_loading() is deliberately not overridden. The
    # inherited UnquantizedFusedMoEMethod implementation is a no-op for QAIC:
    # its only weight mutation (_maybe_pad_weight) is gated on
    # current_platform.is_rocm(), and it then returns early because QAIC's
    # unquantized_backend is OOT. So the packed weights and their scales are
    # left untouched, which is exactly what mxfp4_mm needs.

    def forward_oot(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Never take the fused HVX fast path: that kernel expects dense fp16
        # weight rows and its wrapper casts the weight tensors to x.dtype,
        # which would reinterpret packed MXFP4 bytes (0-255) as fp16 values
        # without raising. The per-expert loop's matmul hooks below dispatch to
        # qaic::mxfp4_mm on the packed weights instead.
        return self._forward_per_expert(layer, x, topk_weights, topk_ids)

    def _gate_up_matmul(
        self, layer: RoutedExperts, w13_input: torch.Tensor, expert_id: int
    ) -> torch.Tensor:
        # mxfp4_mm computes lhs @ dequant(rhs).T, so no explicit .t() here.
        # The .to(float16) is a no-op in the expected fp16 case and only
        # guards a params_dtype=float32 run against the op's hard fp16 check.
        return torch.ops.qaic.mxfp4_mm(
            w13_input.to(torch.float16).contiguous(),
            layer.w13_weight[expert_id],  # uint8 [2I, H/2]
            layer.w13_weight_scale[expert_id],  # uint8 [2I, H/32]
        )

    def _down_matmul(
        self, layer: RoutedExperts, hidden: torch.Tensor, expert_id: int
    ) -> torch.Tensor:
        # .contiguous() matters here: the swigluoai branch builds `hidden` from
        # strided gate_up[..., ::2] / [..., 1::2] slices.
        return torch.ops.qaic.mxfp4_mm(
            hidden.to(torch.float16).contiguous(),
            layer.w2_weight[expert_id],  # uint8 [H, I/2]
            layer.w2_weight_scale[expert_id],  # uint8 [H, I/32]
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        if self.moe.has_bias:
            return biased_moe_quant_config(layer.w13_bias, layer.w2_bias)
        return None
