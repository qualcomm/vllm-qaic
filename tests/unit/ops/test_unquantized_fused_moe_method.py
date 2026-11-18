# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""QAicUnquantizedFusedMoEMethod: per-expert loop and fused kernels vs a
dense reference. This is the path regular (non-MXFP4) MoE models take."""

import pytest
import torch

from tests.unit.ops.moe_test_utils import (
    make_layer,
    make_method,
    make_routing,
    ref_activation,
    ref_moe,
    requires_qaic,
)
from vllm_qaic.ops.unquantized_fused_moe_method import QAicUnquantizedFusedMoEMethod

E, H, INTER, TOP_K = 8, 128, 64, 2


def _dense_layer(activation, has_bias, dtype):
    params = dict(
        w13_weight=torch.randn(E, 2 * INTER, H, dtype=dtype) / H**0.5,
        w2_weight=torch.randn(E, H, INTER, dtype=dtype) / INTER**0.5,
    )
    if has_bias:
        params["w13_bias"] = torch.randn(E, 2 * INTER, dtype=dtype) * 0.1
        params["w2_bias"] = torch.randn(E, H, dtype=dtype) * 0.1
    return make_layer(activation, **params)


def _ref(layer, x, topk_weights, topk_ids, activation):
    return ref_moe(
        x,
        topk_weights,
        topk_ids,
        layer.w13_weight,
        layer.w2_weight,
        getattr(layer, "w13_bias", None),
        getattr(layer, "w2_bias", None),
        activation,
    )


@pytest.mark.parametrize("activation", ["silu", "gelu", "swigluoai"])
def test_activation_hook_matches_reference(activation):
    method = make_method(QAicUnquantizedFusedMoEMethod, has_bias=False)
    gate_up = torch.randn(5, 2 * INTER) * 6  # wide enough to hit the clamps
    torch.testing.assert_close(
        method._activation(gate_up, activation), ref_activation(gate_up, activation)
    )


@pytest.mark.parametrize("activation", ["silu", "swigluoai"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_per_expert_cpu(activation, has_bias, num_tokens):
    torch.manual_seed(0)
    layer = _dense_layer(activation, has_bias, torch.float32)
    method = make_method(QAicUnquantizedFusedMoEMethod, has_bias)
    x = torch.randn(num_tokens, H)
    topk_weights, topk_ids = make_routing(num_tokens, E, TOP_K)

    out = method._forward_per_expert(layer, x, topk_weights, topk_ids)

    torch.testing.assert_close(
        out, _ref(layer, x, topk_weights, topk_ids, activation), atol=1e-4, rtol=1e-4
    )


def test_matmul_hooks_are_used_by_per_expert_loop():
    """Overriding the hooks must be enough to change the compute; this is the
    contract QaicGptOssMxfp4MoEMethod relies on."""

    class Doubling(QAicUnquantizedFusedMoEMethod):
        def _down_matmul(self, layer, hidden, expert_id):
            return 2 * super()._down_matmul(layer, hidden, expert_id)

    torch.manual_seed(0)
    layer = _dense_layer("silu", False, torch.float32)
    x = torch.randn(4, H)
    topk_weights, topk_ids = make_routing(4, E, TOP_K)
    base = make_method(QAicUnquantizedFusedMoEMethod, False)._forward_per_expert(
        layer, x, topk_weights, topk_ids
    )
    doubled = make_method(Doubling, False)._forward_per_expert(
        layer, x, topk_weights, topk_ids
    )
    torch.testing.assert_close(doubled, 2 * base)


@requires_qaic
@pytest.mark.parametrize("kernel", ["hmx", "hvx"])
@pytest.mark.parametrize("activation", ["silu", "swigluoai"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 16])
def test_fused_kernel_on_device(monkeypatch, kernel, activation, has_bias, num_tokens):
    monkeypatch.setenv("QAIC_UNQUANTIZED_FUSED_MOE_KERNEL", kernel)
    torch.manual_seed(0)
    layer = _dense_layer(activation, has_bias, torch.float16)
    x = torch.randn(num_tokens, H, dtype=torch.float16)
    topk_weights, topk_ids = make_routing(num_tokens, E, TOP_K)
    expected = _ref(layer, x, topk_weights, topk_ids, activation)

    dev = "qaic:0"
    for name in ("w13_weight", "w2_weight", "w13_bias", "w2_bias"):
        if hasattr(layer, name):
            setattr(layer, name, getattr(layer, name).to(dev))
    method = make_method(QAicUnquantizedFusedMoEMethod, has_bias)
    out = method.forward_oot(
        layer, x.to(dev), topk_weights.to(dev), topk_ids.to(dev)
    ).cpu()

    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
