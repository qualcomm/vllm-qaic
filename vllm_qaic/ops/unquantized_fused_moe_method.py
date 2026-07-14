# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Register QAIC native forward for unquantized fused MoE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)


class QAicUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """QAIC OOT implementation for unquantized fused MoE."""

    def forward_oot(
        self,
        layer: FusedMoE,  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unquantized MoE on QAIC by batching tokens per expert."""
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        activation = layer.activation
        is_no_mul = activation.endswith("_no_mul")

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

            w13 = layer.w13_weight[expert_id]
            w2 = layer.w2_weight[expert_id]

            gate_up = tokens @ w13.t()
            if self.moe.has_bias:
                gate_up = gate_up + layer.w13_bias[expert_id]

            if is_no_mul:
                if activation == "silu_no_mul":
                    hidden = F.silu(gate_up)
                elif activation == "gelu_no_mul":
                    hidden = F.gelu(gate_up)
                else:
                    hidden = F.relu(gate_up).square()
            elif activation == "swigluoai":
                gate, up = gate_up[..., ::2], gate_up[..., 1::2]
                gate = gate.clamp(max=7.0)
                up = up.clamp(-7.0, 7.0)
                hidden = (up + 1) * (gate * torch.sigmoid(gate * 1.702))
            else:
                half = gate_up.shape[-1] // 2
                gate, up = gate_up[..., :half], gate_up[..., half:]
                if activation == "silu":
                    hidden = F.silu(gate) * up
                else:
                    hidden = F.gelu(gate) * up

            expert_out = hidden @ w2.t()
            if self.moe.has_bias:
                expert_out = expert_out + layer.w2_bias[expert_id]
            weighted_out = weights.unsqueeze(-1) * expert_out
            out.index_put_((tgt_idx,), weighted_out, accumulate=True)
            offset += count

        return out
