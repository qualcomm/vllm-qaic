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

This module registers the QAIC config directly under ``"gpt_oss_mxfp4"``.
Both execution paths load packed MXFP4 weights and allocate biases in the model
dtype. With ``VLLM_QAIC_TRITON_FUSED_MOE=1``, the upstream modular Triton path
handles routing, expert matmuls, fused SwiGLU and output reduction. The QAIC
adapter supplies routing dtypes and weight metadata needed by that path.

The native path runs per-expert matmuls through ``qaic::mxfp4_mm`` and activation
through ``qaic::swigluoai``. It bypasses the dense HVX fused-MoE kernel, which
does not understand packed MXFP4 weights or their scales. Native-op validation
applies only when this path is selected.

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
from vllm.model_executor.layers.quantization import register_quantization_config
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
from vllm_qaic.utils import QAIC_GPT_OSS_MXFP4_METHOD

logger = init_logger(__name__)

_MXFP4_BLOCK_SIZE = 32
# GPT-OSS's clamped SwiGLU constants (vllm's swigluoai activation).
_SWIGLUOAI_ALPHA = 1.702
_SWIGLUOAI_LIMIT = 7.0
# Activation dtypes qaic::mxfp4_mm accepts; it computes natively in lhs dtype.
_MXFP4_MM_ACT_DTYPES = (torch.float16, torch.bfloat16)
# The fused kernel's DMA byte-stride limit: dim * sizeof(fp16) <= UINT16_MAX.
_FUSED_MAX_DIM = 0xFFFF // 2
# Largest batch for which the fused-experts combine uses index_add_. On QAIC
# (H=2880, K=4) index_add_ in the activation dtype is ~3x faster than
# argsort + gather + fp32 sum up to 128 tokens, and ~2x slower at 2048+.
_COMBINE_INDEX_ADD_MAX_TOKENS = 128

# Row order that qaic::mxfp_fused_swiglu_experts expects for w13, w13_scale and
# w13_bias. GPT-OSS checkpoints store gate/up rows interleaved as
# [g0, u0, g1, u1, ...]. torch_qaic builds from before pytorch 70b6e837c6
# ("Use concatenated SwiGLU-OAI layout for fused experts") read that layout
# directly. Later builds read concatenated [gate(0:I), up(0:I)] rows. The op
# schema is the same in both, so _fused_w13_layout() probes the kernel once
# per process.
_W13_INTERLEAVED = "interleaved"
_W13_CONCAT = "concat"
_fused_w13_layout_cache: str | None = None


def _interleaved_to_concat(t: torch.Tensor) -> torch.Tensor:
    """Reorder dim 1 of a W13 tensor from [g0,u0,g1,u1,...] to [g..., u...]."""
    # torch.cat of the strided halves runs on device. view/transpose/contiguous
    # does not: QAIC lacks a strided uint8 device copy and falls back to CPU.
    return torch.cat((t[:, 0::2], t[:, 1::2]), dim=1).contiguous()


def _probe_fused_w13_layout(device: torch.device) -> str:
    """Find the fused kernel's W13 row order by checking it against unfused ops.

    Runs one tiny expert through mxfp_fused_swiglu_experts and through
    mxfp4_mm -> swigluoai -> mxfp4_mm, reading the same gate_up output both
    interleaved and concatenated. Whichever reading matches the fused result
    is the kernel's layout.
    """
    h, inter, rows = 128, 64, 4
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn(rows, h, generator=gen) * 0.5).half().to(device)
    w13 = torch.randint(
        0, 256, (1, 2 * inter, h // 2), dtype=torch.uint8, generator=gen
    )
    w2 = torch.randint(0, 256, (1, h, inter // 2), dtype=torch.uint8, generator=gen)
    # E8M0 exponent 124 scales the e2m1 values by 2^-3, so activations stay
    # mostly inside the SwiGLU clamp.
    s13 = torch.full((1, 2 * inter, h // 32), 124, dtype=torch.uint8)
    s2 = torch.full((1, h, inter // 32), 124, dtype=torch.uint8)
    w13, w2, s13, s2 = (t.to(device) for t in (w13, w2, s13, s2))
    router_w = torch.ones(rows, 1, dtype=torch.float16, device=device)
    bnd = torch.tensor([0, rows], dtype=torch.int32).to(device)
    fused = (
        torch.ops.qaic.mxfp_fused_swiglu_experts(
            x,
            router_w,
            bnd,
            w13,
            s13,
            w2,
            s2,
            None,
            None,
            False,
            0,
            _SWIGLUOAI_ALPHA,
            _SWIGLUOAI_LIMIT,
        )
        .cpu()
        .float()
    )
    gate_up = torch.ops.qaic.mxfp4_mm(x, w13[0], s13[0])

    def down(interleaved: torch.Tensor) -> torch.Tensor:
        act = torch.ops.qaic.swigluoai(
            interleaved.contiguous(), _SWIGLUOAI_ALPHA, _SWIGLUOAI_LIMIT
        )
        return torch.ops.qaic.mxfp4_mm(act.contiguous(), w2[0], s2[0]).cpu().float()

    candidates = {
        _W13_INTERLEAVED: down(gate_up),
        _W13_CONCAT: down(
            gate_up.view(rows, 2, inter).transpose(1, 2).reshape(rows, 2 * inter)
        ),
    }
    err = {
        name: ((fused - ref).norm() / ref.norm().clamp_min(1e-6)).item()
        for name, ref in candidates.items()
    }
    best = min(err, key=err.__getitem__)
    other = max(err, key=err.__getitem__)
    # The two readings differ by O(1) relative error; ~1e-4 is fp16 noise.
    if err[best] < 5e-2 and err[other] > 0.5:
        logger.info("mxfp_fused_swiglu_experts reads W13 rows %s", best)
        return best
    raise RuntimeError(
        "qaic::mxfp_fused_swiglu_experts matches neither the interleaved nor the "
        f"concatenated W13 layout (relative error {err}); the installed torch_qaic "
        "fused-experts kernel is broken"
    )


def _fused_w13_layout(device: torch.device) -> str:
    global _fused_w13_layout_cache
    if _fused_w13_layout_cache is None:
        _fused_w13_layout_cache = _probe_fused_w13_layout(device)
    return _fused_w13_layout_cache


def _create_gpt_oss_mxfp4_weights(
    layer: RoutedExperts,
    moe: FusedMoEConfig,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
) -> None:
    """Allocate shared packed weights and biases for both QAIC MoE backends.

    Match upstream's packed shapes, but allocate biases in params_dtype instead
    of its fixed BF16 dtype so FP16 QAIC models can load the same checkpoint.
    """
    layer.params_dtype = params_dtype
    layer.num_experts = num_experts
    w13_size = moe.w13_num_shards * intermediate_size_per_partition

    parameters: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "w13_weight": ((num_experts, w13_size, hidden_size // 2), torch.uint8),
        "w13_weight_scale": (
            (num_experts, w13_size, hidden_size // _MXFP4_BLOCK_SIZE),
            torch.uint8,
        ),
        "w2_weight": (
            (num_experts, hidden_size, intermediate_size_per_partition // 2),
            torch.uint8,
        ),
        "w2_weight_scale": (
            (
                num_experts,
                hidden_size,
                intermediate_size_per_partition // _MXFP4_BLOCK_SIZE,
            ),
            torch.uint8,
        ),
    }
    if moe.has_bias:
        parameters["w13_bias"] = ((num_experts, w13_size), params_dtype)
        parameters["w2_bias"] = ((num_experts, hidden_size), params_dtype)

    for name, (shape, dtype) in parameters.items():
        parameter = torch.nn.Parameter(
            torch.zeros(shape, dtype=dtype), requires_grad=False
        )
        layer.register_parameter(name, parameter)
        set_weight_attrs(parameter, extra_weight_attrs)


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


class QaicTritonGptOssMxfp4MoEMethod(GptOssMxfp4MoEMethod):
    """QAIC adapter for the upstream modular Triton MXFP4 MoE path."""

    def __init__(self, moe: FusedMoEConfig):
        # Platform startup already imports and patches optional triton_kernels.
        # Require the package before patching the GPT-OSS Triton experts.
        from vllm_qaic.patch.patch_qaic_triton_import import (
            enable_qaic_triton_kernels,
        )

        enable_qaic_triton_kernels()
        import vllm_qaic.patch.patch_oai_triton_experts  # noqa: F401

        from vllm.model_executor.layers.fused_moe.experts import (
            gpt_oss_triton_kernels_moe as oai_triton_experts,
        )

        # GptOssMxfp4MoEMethod.__init__ invokes the upstream backend oracle
        # before a subclass can replace its result. The oracle rejects OOT
        # platforms, so scope the QAIC-approved selection to this constructor
        # call and immediately restore the global selector afterwards.
        import vllm.model_executor.layers.quantization.mxfp4 as upstream_mxfp4

        original_selector = upstream_mxfp4.select_mxfp4_moe_backend
        upstream_mxfp4.select_mxfp4_moe_backend = lambda _moe: (
            Mxfp4MoeBackend.TRITON,
            oai_triton_experts.OAITritonExperts,
        )
        try:
            super().__init__(moe)
        finally:
            upstream_mxfp4.select_mxfp4_moe_backend = original_selector

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
        """Allocate shared weights and retain the upstream Triton metadata."""
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size_per_partition
        self.hidden_size = hidden_size
        _create_gpt_oss_mxfp4_weights(
            layer,
            self.moe,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )
        layer.w13_weight_scale.quant_method = "block"
        layer.w2_weight_scale.quant_method = "block"


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
        if not triton_op_enabled("FUSED_MOE"):
            check_qaic_mxfp4_ops_available()
        return "gpt_oss_mxfp4"

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, RoutedExperts):
            if triton_op_enabled("FUSED_MOE"):
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
        check_qaic_mxfp4_ops_available()
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
        _create_gpt_oss_mxfp4_weights(
            layer,
            self.moe,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # The inherited UnquantizedFusedMoEMethod implementation is a no-op
        # for QAIC: its only weight mutation (_maybe_pad_weight) is gated on
        # current_platform.is_rocm(), and it then returns early because QAIC's
        # unquantized_backend is OOT. mxfp4_mm takes the packed weights and
        # scales as loaded. The only change made here is reordering W13 rows
        # for the fused kernel when it needs that. Doing it at load time keeps
        # the reorder's temporary allocation out of the first forward pass.
        super().process_weights_after_loading(layer)
        activation = getattr(layer.activation, "value", layer.activation)
        if layer.w13_weight.device.type == "qaic" and self._fused_weights_supported(
            layer, activation
        ):
            self._prepare_fused_w13(layer)

    def _fused_weights_supported(self, layer: RoutedExperts, activation: str) -> bool:
        if activation != "swigluoai":
            return False
        if layer.expert_map is not None:
            return False
        if not hasattr(torch.ops.qaic, "mxfp_fused_swiglu_experts"):
            return False
        w13, w2 = layer.w13_weight, layer.w2_weight
        hidden, inter = w2.shape[1], w2.shape[2] * 2
        if hidden % _MXFP4_BLOCK_SIZE or inter % _MXFP4_BLOCK_SIZE:
            return False
        if hidden > _FUSED_MAX_DIM or inter > _FUSED_MAX_DIM:
            return False
        if w13.shape[1] != 2 * inter or w13.shape[2] * 2 != hidden:
            return False
        return not self.moe.has_bias or (
            layer.w13_bias.dtype == torch.float16
            and layer.w2_bias.dtype == torch.float16
        )

    def _fused_experts_supported(
        self, layer: RoutedExperts, x: torch.Tensor, activation: str
    ) -> bool:
        if x.dtype != torch.float16 or x.ndim != 2 or x.device.type != "qaic":
            return False
        if x.shape[1] != layer.w2_weight.shape[1]:
            return False
        if not self._fused_weights_supported(layer, activation):
            return False
        self._prepare_fused_w13(layer)
        return True

    def _prepare_fused_w13(self, layer: RoutedExperts) -> None:
        """Put the layer's W13 in the fused kernel's row order, once.

        Raises if the layout probe can't match the kernel to either row
        order. For a concat kernel, w13_weight and w13_weight_scale are
        reordered in place, since copying them would
        double their memory. _gate_up_matmul then re-interleaves the
        per-expert matmul output. w13_bias keeps the checkpoint's interleaved
        order so the shared per-expert loop can add it. The fused kernel gets
        a small reordered copy, _qaic_fused_w13_bias.
        """
        if getattr(layer, "_qaic_fused_w13_ready", False):
            return
        layout = _fused_w13_layout(layer.w13_weight.device)
        bias = layer.w13_bias if self.moe.has_bias else None
        if layout == _W13_CONCAT:
            layer.w13_weight.data = _interleaved_to_concat(layer.w13_weight.data)
            layer.w13_weight_scale.data = _interleaved_to_concat(
                layer.w13_weight_scale.data
            )
            layer._qaic_w13_concat = True
            if bias is not None:
                bias = _interleaved_to_concat(bias.data)
        layer._qaic_fused_w13_bias = bias
        layer._qaic_fused_w13_ready = True

    def _forward_fused_experts(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Single-launch grouped MXFP4 experts (qaic::mxfp_fused_swiglu_experts)."""
        self._prepare_fused_w13(layer)
        num_tokens, top_k = topk_ids.shape
        num_experts = layer.w13_weight.shape[0]

        aligned = self._moe_align(x, topk_weights, topk_ids, num_experts)
        if aligned is None:
            bnd, sorted_token_idx, x_sorted, sorted_weights = self._sort_by_expert(
                x, topk_weights, topk_ids, num_experts
            )
        else:
            bnd, sorted_token_idx, x_sorted, sorted_weights = aligned

        has_bias = self.moe.has_bias
        y = torch.ops.qaic.mxfp_fused_swiglu_experts(
            x_sorted,
            sorted_weights,
            bnd,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            layer._qaic_fused_w13_bias,
            layer.w2_bias if has_bias else None,
            layer.apply_router_weight_on_input,
            0,
            _SWIGLUOAI_ALPHA,
            _SWIGLUOAI_LIMIT,
        )
        # Sum each token's top_k output rows. Don't use index_put_
        # (accumulate=True): duplicate indices drop contributions on QAIC.
        if num_tokens <= _COMBINE_INDEX_ADD_MAX_TOKENS:
            # Small batches: accumulate in the activation dtype in place. This
            # needs no sort and no dtype conversion, but rounds after each
            # add, so it isn't bit-identical to the fp32 sum below.
            out = torch.zeros(num_tokens, y.shape[1], dtype=y.dtype, device=y.device)
            return out.index_add_(0, sorted_token_idx, y).to(x.dtype)
        # Large batches: index_add_ is 2x-4x slower here. Argsort by token
        # groups each token's rows together; the sum doesn't depend on order
        # within a token, so the exact (token, k) slot permutation isn't needed.
        inv = torch.argsort(sorted_token_idx)
        y = y[inv].view(num_tokens, top_k, -1)
        return y.float().sum(dim=1).to(x.dtype)

    @staticmethod
    def _moe_align(
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Expert-grouped rows from qaic::moe_align_block_size; None if unusable.

        One NSP launch for a stable counting sort by expert plus the hidden-state
        and weight gathers. Output is bit-identical to ``_sort_by_expert`` and
        5x-17x faster on QAIC (1..8192 tokens, E=32, K=4, H=2880).
        """
        if not hasattr(torch.ops.qaic, "moe_align_block_size"):
            return None
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        bnd, token_idx, x_sorted, w_sorted = torch.ops.qaic.moe_align_block_size(
            x, topk_weights.contiguous(), ids.contiguous(), num_experts, None
        )
        return (
            bnd.to(torch.int32).contiguous(),
            token_idx,
            x_sorted.contiguous(),
            w_sorted.to(x.dtype).reshape(-1, 1).contiguous(),
        )

    @staticmethod
    def _sort_by_expert(
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager fallback for ``_moe_align``: (bnd, token_idx, x_sorted, w_sorted)."""
        num_tokens, top_k = topk_ids.shape
        token_idx = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        flat_expert_ids = topk_ids.reshape(-1)
        sorted_idx = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_idx]
        sorted_token_idx = token_idx[sorted_idx]
        sorted_weights = topk_weights.reshape(-1)[sorted_idx].to(x.dtype)
        x_sorted = x[sorted_token_idx].contiguous()
        # bnd[e] = first sorted row owned by expert e; bnd[E] = R.
        bnd = torch.searchsorted(
            sorted_expert_ids,
            torch.arange(
                num_experts + 1, device=x.device, dtype=sorted_expert_ids.dtype
            ),
        ).to(torch.int32)
        return (
            bnd.contiguous(),
            sorted_token_idx,
            x_sorted,
            sorted_weights.unsqueeze(-1).contiguous(),
        )

    def forward_oot(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Never take the dense-fp16 fused HVX fast path inherited from the
        # unquantized method: it would reinterpret packed MXFP4 bytes as fp16
        # values without raising. Use the MXFP4-aware single-launch kernel when
        # eligible, else the per-expert loop (matmul hooks below call
        # qaic::mxfp4_mm on the packed weights).
        activation = getattr(layer.activation, "value", layer.activation)
        if self._fused_experts_supported(layer, x, activation):
            return self._forward_fused_experts(layer, x, topk_weights, topk_ids)
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
        gate_up = torch.ops.qaic.mxfp4_mm(
            self._mxfp4_mm_lhs(w13_input),
            layer.w13_weight[expert_id],  # uint8 [2I, H/2]
            layer.w13_weight_scale[expert_id],  # uint8 [2I, H/32]
        )
        if getattr(layer, "_qaic_w13_concat", False):
            # _prepare_fused_w13 reordered the rows for a concat fused kernel.
            # The bias add and swigluoai that follow expect interleaved rows.
            rows, cols = gate_up.shape
            gate_up = (
                gate_up.view(rows, 2, cols // 2).transpose(1, 2).reshape(rows, cols)
            )
        return gate_up

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
