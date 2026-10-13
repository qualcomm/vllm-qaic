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

Routing, bias and the weighted reduce are inherited from
``QAicUnquantizedFusedMoEMethod._forward_per_expert``; its
``_gate_up_matmul``/``_down_matmul`` hooks are overridden to call
``qaic::mxfp4_mm`` and its ``_activation`` hook to call ``qaic::swigluoai``.

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
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.mxfp4 import GptOssMxfp4MoEMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

from vllm_qaic.logger import init_logger
from vllm_qaic.ops._triton_flags import triton_op_enabled
from vllm_qaic.ops.unquantized_fused_moe_method import QAicUnquantizedFusedMoEMethod

_MXFP4_BLOCK_SIZE = 32
# GPT-OSS's clamped SwiGLU constants (vllm's swigluoai activation).
_SWIGLUOAI_ALPHA = 1.702
_SWIGLUOAI_LIMIT = 7.0
# Activation dtypes qaic::mxfp4_mm accepts; it computes natively in lhs dtype.
_MXFP4_MM_ACT_DTYPES = (torch.float16, torch.bfloat16)


def check_qaic_mxfp4_ops_available() -> None:
    """Raise if the installed torch_qaic lacks the ops this method calls.

    Without this, an older torch_qaic wheel only fails on the first MoE
    forward, after the full checkpoint has been loaded.
    """
    # The qaic ops are registered on import; don't rely on the caller having
    # imported torch_qaic already.
    import torch_qaic  # noqa: F401

    missing = [
        name for name in ("mxfp4_mm", "swigluoai") if not hasattr(torch.ops.qaic, name)
    ]
    if missing:
        raise RuntimeError(
            "gpt_oss_mxfp4 on QAIC requires torch_qaic ops "
            f"{', '.join('qaic::' + n for n in missing)}, which the installed "
            "torch_qaic does not provide. Install a torch_qaic build that "
            "includes qaic::mxfp4_mm and qaic::swigluoai."
        )


logger = init_logger(__name__)


class QaicTritonGptOssMxfp4MoEMethod(GptOssMxfp4MoEMethod):
    """QAIC adapter for the upstream modular Triton MXFP4 MoE path."""

    def __init__(self, moe: FusedMoEConfig):
        # Keep triton_kernels optional until a GPT-OSS routed-expert layer
        # actually selects the Triton backend. Importing this module during
        # platform pre-registration affects every model and can fail when the
        # installed triton_kernels revision does not match vLLM's revision.
        from vllm_qaic.patch.patch_qaic_triton_import import (
            enable_qaic_triton_kernels,
        )

        enable_qaic_triton_kernels()
        from vllm_qaic.quantization.qaic_oai_triton_experts import (
            QaicOAITritonExperts,
        )

        super().__init__(moe)

        # Upstream auto-selection returns (NONE, None) for an OOT platform
        # when none of its built-in platform branches match. QAIC has already
        # validated this Triton backend separately, so explicitly select the
        # modular OAI implementation here.
        self.mxfp4_backend = Mxfp4MoeBackend.TRITON
        self.experts_cls = QaicOAITritonExperts
        logger.info_once(
            "QAIC GPT-OSS MXFP4: forcing backend=%s, experts=%s",
            self.mxfp4_backend.value,
            self.experts_cls.__name__,
        )

    @property
    def is_monolithic(self) -> bool:
        # QAIC uses OAITritonExperts, which is a modular experts
        # implementation. During FusedMoE construction the backend oracle can
        # leave experts_cls unset until weights are processed, so the base
        # property cannot safely dereference it here.
        return False

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create packed MXFP4 weights without allocating BF16 tensors.

        Upstream hardcodes the GPT-OSS expert biases to BF16. QAIC does not
        support BF16 tensors, so allocate them in the model's resolved dtype
        (FP16 or FP32) instead. Keep the rest of the upstream layout and scale
        metadata unchanged because ``OAITritonExperts`` relies on them.
        """
        self.num_experts = num_experts
        weight_dtype = torch.uint8
        scale_dtype = torch.uint8

        layer.params_dtype = params_dtype
        layer.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size

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
        w13_weight_scale.quant_method = "block"

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
        w2_weight_scale.quant_method = "block"

        if self.moe.has_bias:
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
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=params_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)


class QaicGptOssMxfp4Config(QuantizationConfig):
    """QAIC eager-mode MXFP4 config for GPT-OSS checkpoints."""

    def get_name(self) -> str:
        # Must be exactly "gpt_oss_mxfp4": FusedMoE.weight_loader() has a
        # hardcoded string check on quant_config.get_name() to special-case
        # gpt-oss's combined-experts checkpoint layout.
        return "gpt_oss_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # bf16 is only reachable on AI200; QaicPlatform.check_and_update_config
        # rejects it elsewhere.
        return [torch.float16, torch.bfloat16, torch.float32]

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
        check_qaic_mxfp4_ops_available()
        return "gpt_oss_mxfp4"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            if triton_op_enabled("GPT_OSS_MOE"):
                logger.info_once(
                    "QAIC GPT-OSS MXFP4: selecting modular upstream Triton MoE method"
                )
                return QaicTritonGptOssMxfp4MoEMethod(layer.moe_config)
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
            # Biases are allocated in params_dtype rather than upstream's
            # fixed bf16, since the model may run in fp16 (bf16 is AI200-only);
            # weight_loader() casts the checkpoint's bf16 values on copy.
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

    @staticmethod
    def _mxfp4_mm_lhs(x: torch.Tensor) -> torch.Tensor:
        # mxfp4_mm takes fp16 or bf16 activations and computes in that dtype.
        # Only an fp32 model needs a cast; fp16 is the safe choice for it,
        # since bf16 would need AI200.
        if x.dtype not in _MXFP4_MM_ACT_DTYPES:
            x = x.to(torch.float16)
        return x.contiguous()

    def _gate_up_matmul(
        self, layer: RoutedExperts, w13_input: torch.Tensor, expert_id: int
    ) -> torch.Tensor:
        # mxfp4_mm computes lhs @ dequant(rhs).T, so no explicit .t() here.
        return torch.ops.qaic.mxfp4_mm(
            self._mxfp4_mm_lhs(w13_input),
            layer.w13_weight[expert_id],  # uint8 [2I, H/2]
            layer.w13_weight_scale[expert_id],  # uint8 [2I, H/32]
        )

    def _down_matmul(
        self, layer: RoutedExperts, hidden: torch.Tensor, expert_id: int
    ) -> torch.Tensor:
        return torch.ops.qaic.mxfp4_mm(
            self._mxfp4_mm_lhs(hidden),
            layer.w2_weight[expert_id],  # uint8 [H, I/2]
            layer.w2_weight_scale[expert_id],  # uint8 [H, I/32]
        )

    def _activation(self, gate_up: torch.Tensor, activation: str) -> torch.Tensor:
        # GPT-OSS always uses swigluoai. Run it as one device kernel instead of
        # the inherited elementwise chain; qaic::swigluoai takes the same
        # interleaved gate/up layout and clamp semantics, and returns a
        # contiguous tensor for the down projection.
        if activation != "swigluoai":
            return super()._activation(gate_up, activation)
        return torch.ops.qaic.swigluoai(
            gate_up.contiguous(), _SWIGLUOAI_ALPHA, _SWIGLUOAI_LIMIT
        )

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        if self.moe.has_bias:
            return biased_moe_quant_config(layer.w13_bias, layer.w2_bias)
        return None
