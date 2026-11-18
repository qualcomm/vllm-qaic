# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared helpers for the QAIC MoE method tests."""

from types import SimpleNamespace

import pytest
import torch

# e2m1 code -> value; low nibble holds the even-K element.
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    + [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def qaic_device_available() -> bool:
    try:
        import torch_qaic  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "qaic") and torch.qaic.is_available()


requires_qaic = pytest.mark.skipif(
    not qaic_device_available(), reason="needs a QAIC device and torch_qaic"
)


def make_method(cls, has_bias: bool):
    """Instantiate a MoE method without a FusedMoEConfig.

    The per-expert loop and matmul/activation hooks only read
    ``self.moe.has_bias``, so a stub is enough.
    """
    method = object.__new__(cls)
    method.moe = SimpleNamespace(has_bias=has_bias)
    return method


def make_layer(activation: str, apply_router_weight_on_input=False, **params):
    return SimpleNamespace(
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        expert_map=None,
        **params,
    )


def make_routing(num_tokens: int, num_experts: int, top_k: int):
    logits = torch.randn(num_tokens, num_experts)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    return topk_weights, topk_ids.to(torch.int32)


def ref_activation(gate_up: torch.Tensor, activation: str) -> torch.Tensor:
    """Independent fp32 reference for the activations the MoE methods support."""
    x = gate_up.float()
    if activation == "swigluoai":
        gate = x[..., ::2].clamp(max=7.0)
        up = x[..., 1::2].clamp(-7.0, 7.0)
        return (up + 1) * gate * torch.sigmoid(1.702 * gate)
    gate, up = x.chunk(2, dim=-1)
    if activation == "silu":
        return gate * torch.sigmoid(gate) * up
    if activation == "gelu":
        return torch.nn.functional.gelu(gate) * up
    raise NotImplementedError(activation)


def ref_moe(x, topk_weights, topk_ids, w13, w2, b13, b2, activation):
    """Dense fp32 MoE reference; w13 is [E, 2I, H], w2 is [E, H, I]."""
    x = x.float()
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[1]):
            e = int(topk_ids[t, k])
            h = x[t] @ w13[e].float().t()
            if b13 is not None:
                h = h + b13[e].float()
            h = ref_activation(h, activation)
            o = h @ w2[e].float().t()
            if b2 is not None:
                o = o + b2[e].float()
            out[t] += topk_weights[t, k].float() * o
    return out


def random_mxfp4(*lead: int, k: int):
    """Random packed MXFP4 operand: ([*lead, k/2] u8, [*lead, k/32] u8).

    Scales stay near 2^0 so dequantized magnitudes are O(1).
    """
    packed = torch.randint(0, 256, (*lead, k // 2), dtype=torch.uint8)
    scale = torch.randint(122, 127, (*lead, k // 32), dtype=torch.uint8)
    return packed, scale


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    lo = _E2M1_LUT[(packed & 0xF).long()]
    hi = _E2M1_LUT[(packed >> 4).long()]
    values = torch.stack([lo, hi], dim=-1).flatten(-2)
    s = scale.float()
    s = torch.where(scale == 0xFF, torch.zeros_like(s), torch.exp2(s - 127))
    return values * s.repeat_interleave(32, dim=-1)
