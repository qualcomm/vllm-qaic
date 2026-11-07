# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Register QAIC native forward for unquantized fused MoE."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )
from vllm_qaic.ops._triton_flags import triton_op_enabled

triton_en = triton_op_enabled("FUSED_MOE")


class QAicUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """QAIC OOT implementation for unquantized fused MoE."""

    @property
    def is_monolithic(self) -> bool:
        return False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        return None

    def forward_oot(
        self,
        layer: FusedMoE,  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute unquantized MoE on QAIC by batching tokens per expert."""
        # Triton fast path (opt-in): dispatch to vLLM's hand-written Triton
        # fused-MoE kernel (fused_moe_kernel, reached via fused_experts ->
        # torch.ops.vllm.fused_experts -> fused_experts_impl). This replaces the
        # QAIC NSP MoE path below with the fused grouped-GEMM + fused activation
        # kernel. fused_experts takes the full stacked expert
        # weights (layer.w13_weight / layer.w2_weight), the router topk tensors,
        # and applies the gated activation internally (apply_moe_activation), so
        # the routing/activation branching below is handled entirely in-kernel.
        # Biases (has_bias) are threaded through via biased_moe_quant_config; the
        # activation string is parsed to the MoEActivation enum. Flag off -> the
        # QAIC NSP path below, bit-for-bit.
        if triton_en:
            from vllm.model_executor.layers.fused_moe.activation import (
                MoEActivation,
            )
            from vllm.model_executor.layers.fused_moe.config import (
                biased_moe_quant_config,
            )
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                fused_experts,
            )

            quant_config = (
                biased_moe_quant_config(layer.w13_bias, layer.w2_bias)
                if self.moe.has_bias
                else None
            )
            return fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.from_str(layer.activation),
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=layer.global_num_experts,
                expert_map=layer.expert_map,
                quant_config=quant_config,
            )

        activation = getattr(layer.activation, "value", layer.activation)
        apply_router_weight_on_input = layer.apply_router_weight_on_input

        qaic_kernel_activations = {
            "silu",
            "gelu",
            "gelu_tanh",
            "swigluoai",
            "swiglustep",
            "silu_no_mul",
            "gelu_no_mul",
            "gelu_tanh_no_mul",
            "relu2_no_mul",
        }
        if x.dtype == torch.float16 and activation in qaic_kernel_activations:
            kernel = os.environ.get("QAIC_UNQUANTIZED_FUSED_MOE_KERNEL", "hmx").lower()
            if kernel == "hmx":
                from vllm_qaic._custom_ops import (
                    unquantized_fused_moe_hmx as unquantized_fused_moe_impl,
                )
            elif kernel == "hvx":
                from vllm_qaic._custom_ops import (
                    unquantized_fused_moe_hvx as unquantized_fused_moe_impl,
                )
            else:
                raise ValueError(
                    "QAIC_UNQUANTIZED_FUSED_MOE_KERNEL must be 'hvx' or 'hmx', "
                    f"got {kernel!r}"
                )

            return unquantized_fused_moe_impl(
                x=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_bias=layer.w13_bias if self.moe.has_bias else None,
                w2_bias=layer.w2_bias if self.moe.has_bias else None,
                activation=activation,
                has_bias=self.moe.has_bias,
                apply_router_weight_on_input=apply_router_weight_on_input,
                expert_map=layer.expert_map,
            )

        return self._forward_per_expert(layer, x, topk_weights, topk_ids)

    def _gate_up_matmul(
        self,
        layer: FusedMoE,  # type: ignore[name-defined] # noqa: F821
        w13_input: torch.Tensor,
        expert_id: int,
    ) -> torch.Tensor:
        """Gate/up projection for one expert. Overridable by quantized methods."""
        return w13_input @ layer.w13_weight[expert_id].t()

    def _down_matmul(
        self,
        layer: FusedMoE,  # type: ignore[name-defined] # noqa: F821
        hidden: torch.Tensor,
        expert_id: int,
    ) -> torch.Tensor:
        """Down projection for one expert. Overridable by quantized methods."""
        return hidden @ layer.w2_weight[expert_id].t()

    def _activation(self, gate_up: torch.Tensor, activation: str) -> torch.Tensor:
        """Gated activation for one expert. Overridable by quantized methods."""
        if activation.endswith("_no_mul"):
            if activation == "silu_no_mul":
                return F.silu(gate_up)
            if activation == "gelu_no_mul":
                return F.gelu(gate_up)
            if activation == "gelu_tanh_no_mul":
                return F.gelu(gate_up, approximate="tanh")
            return F.relu(gate_up).square()
        if activation == "swigluoai":
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=7.0)
            up = up.clamp(-7.0, 7.0)
            return (up + 1) * (gate * torch.sigmoid(gate * 1.702))
        if activation == "swiglustep":
            half = gate_up.shape[-1] // 2
            gate, up = gate_up[..., :half], gate_up[..., half:]
            gate = F.silu(gate).clamp(max=7.0)
            up = up.clamp(-7.0, 7.0)
            return gate * up
        half = gate_up.shape[-1] // 2
        gate, up = gate_up[..., :half], gate_up[..., half:]
        if activation == "silu":
            return F.silu(gate) * up
        if activation == "gelu":
            return F.gelu(gate) * up
        return F.gelu(gate, approximate="tanh") * up

    def _forward_per_expert(
        self,
        layer: FusedMoE,  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-expert PyTorch MoE loop, used when the fused kernel can't be.

        The two matmuls go through ``_gate_up_matmul``/``_down_matmul`` and the
        activation through ``_activation`` so that quantized subclasses can
        substitute device kernels without reimplementing routing or the
        weighted reduce. Subclasses whose weights are not dense rows in
        ``x.dtype`` (e.g. packed MXFP4) must also override ``forward_oot`` so
        the fused kernel path above never sees them.
        """
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        activation = getattr(layer.activation, "value", layer.activation)
        apply_router_weight_on_input = layer.apply_router_weight_on_input

        if layer.expert_map is not None:
            raise NotImplementedError(
                "QAIC unquantized fused MoE fallback does not support expert_map."
            )

        token_idx = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        flat_expert_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1).to(x.dtype)

        sorted_idx = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sorted_idx]
        sorted_token_idx = token_idx[sorted_idx]
        sorted_weights = flat_weights[sorted_idx]
        sorted_tokens = x[sorted_token_idx]

        out = torch.zeros_like(x)

        changes = torch.cat(
            [
                torch.tensor([True], device=x.device),
                sorted_expert_ids[1:] != sorted_expert_ids[:-1],
                torch.tensor([True], device=x.device),
            ]
        )
        boundary_positions = torch.where(changes)[0]
        unique_experts = sorted_expert_ids[boundary_positions[:-1]].tolist()
        counts = (boundary_positions[1:] - boundary_positions[:-1]).tolist()

        offset = 0
        for expert_id, count in zip(unique_experts, counts, strict=False):
            tokens = sorted_tokens[offset : offset + count]
            weights = sorted_weights[offset : offset + count]
            tgt_idx = sorted_token_idx[offset : offset + count]

            w13_input = (
                weights.unsqueeze(-1) * tokens
                if apply_router_weight_on_input
                else tokens
            )
            gate_up = self._gate_up_matmul(layer, w13_input, expert_id)
            if self.moe.has_bias:
                gate_up = gate_up + layer.w13_bias[expert_id]

            hidden = self._activation(gate_up, activation)

            expert_out = self._down_matmul(layer, hidden, expert_id)
            if self.moe.has_bias:
                expert_out = expert_out + layer.w2_bias[expert_id]
            weighted_out = (
                expert_out
                if apply_router_weight_on_input
                else weights.unsqueeze(-1) * expert_out
            )
            out.index_put_((tgt_idx,), weighted_out.to(out.dtype), accumulate=True)
            offset += count

        return out
