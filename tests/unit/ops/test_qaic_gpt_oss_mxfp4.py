# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""QaicGptOssMxfp4MoEMethod: selection logic, and the packed-MXFP4 MoE forward
on device against a dequantize-then-dense reference."""

from types import SimpleNamespace

import pytest
import torch

from tests.unit.ops.moe_test_utils import (
    dequant_mxfp4,
    make_layer,
    make_method,
    make_routing,
    random_mxfp4,
    ref_moe,
    requires_qaic,
)
from vllm_qaic.quantization import qaic_gpt_oss_mxfp4_config as mxfp4_cfg
from vllm_qaic.quantization.qaic_gpt_oss_mxfp4_config import (
    QaicGptOssMxfp4Config,
    QaicGptOssMxfp4MoEMethod,
)

_GPT_OSS = SimpleNamespace(model_type="gpt_oss")


@pytest.fixture
def eager(monkeypatch):
    monkeypatch.setattr(mxfp4_cfg.current_platform, "is_aot_inference", lambda: False)
    monkeypatch.setattr(mxfp4_cfg, "check_qaic_mxfp4_ops_available", lambda: None)


@pytest.mark.parametrize("quant_method", ["mxfp4", "gpt_oss_mxfp4"])
def test_override_selects_for_gpt_oss_mxfp4(eager, quant_method):
    assert (
        QaicGptOssMxfp4Config.override_quantization_method(
            {"quant_method": quant_method}, None, hf_config=_GPT_OSS
        )
        == "gpt_oss_mxfp4"
    )


@pytest.mark.parametrize(
    "hf_quant_cfg, hf_config",
    [
        ({"quant_method": "fp8"}, _GPT_OSS),
        (None, _GPT_OSS),
        ({"quant_method": "mxfp4"}, SimpleNamespace(model_type="llama")),
        ({"quant_method": "mxfp4"}, None),
    ],
)
def test_override_declines_otherwise(eager, hf_quant_cfg, hf_config):
    assert (
        QaicGptOssMxfp4Config.override_quantization_method(
            hf_quant_cfg, None, hf_config=hf_config
        )
        is None
    )


def test_override_declines_in_aot(monkeypatch):
    monkeypatch.setattr(mxfp4_cfg.current_platform, "is_aot_inference", lambda: True)
    assert (
        QaicGptOssMxfp4Config.override_quantization_method(
            {"quant_method": "mxfp4"}, None, hf_config=_GPT_OSS
        )
        is None
    )


def test_override_raises_when_ops_missing(monkeypatch):
    monkeypatch.setattr(mxfp4_cfg.current_platform, "is_aot_inference", lambda: False)
    monkeypatch.setattr(mxfp4_cfg, "hasattr", lambda obj, name: False, raising=False)
    with pytest.raises(RuntimeError, match="qaic::mxfp4_mm"):
        QaicGptOssMxfp4Config.override_quantization_method(
            {"quant_method": "mxfp4"}, None, hf_config=_GPT_OSS
        )


def test_forward_oot_never_takes_fused_kernel(monkeypatch):
    """Packed uint8 weights must not reach the dense fused HMX/HVX kernel."""
    method = make_method(QaicGptOssMxfp4MoEMethod, has_bias=True)
    sentinel = object()
    monkeypatch.setattr(method, "_forward_per_expert", lambda *args, **kwargs: sentinel)
    x = torch.zeros(1, 64, dtype=torch.float16)
    assert method.forward_oot(make_layer("swigluoai"), x, None, None) is sentinel


# GPT-OSS-20B/120B: H=2880, I=2880, 32 experts (20B), top-4. The intermediate
# sizes are per-rank after TP and the MXFP4 round-up: TP=4 gives 720 -> 736 and
# TP=8 gives 360 -> 384. The last 16/24 rows of each padded expert are zeroed,
# as the loader leaves them, so the padding must be inert.
@requires_qaic
@pytest.mark.parametrize("inter, valid", [(736, 720), (384, 360)])
@pytest.mark.parametrize("num_tokens", [1, 2, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_mxfp4_moe_on_device(inter, valid, num_tokens, dtype):
    E, H, TOP_K = 32, 2880, 4
    torch.manual_seed(0)
    w13, w13_s = random_mxfp4(E, 2 * inter, k=H)
    w2, w2_s = random_mxfp4(E, H, k=inter)
    # Zero padded gate/up rows and the matching w2 input columns.
    w13[:, 2 * valid :] = 0
    w2[..., valid // 2 :] = 0
    b13 = torch.randn(E, 2 * inter, dtype=dtype) * 0.1
    b13[:, 2 * valid :] = 0
    b2 = torch.randn(E, H, dtype=dtype) * 0.1

    x = torch.randn(num_tokens, H, dtype=dtype) * 0.1
    topk_weights, topk_ids = make_routing(num_tokens, E, TOP_K)

    # Reference uses the dequantized weights rounded to the activation dtype,
    # matching the device's 16-bit crouton decode.
    expected = ref_moe(
        x,
        topk_weights,
        topk_ids,
        dequant_mxfp4(w13, w13_s).to(dtype),
        dequant_mxfp4(w2, w2_s).to(dtype),
        b13,
        b2,
        "swigluoai",
    )

    dev = "qaic:0"
    layer = make_layer(
        "swigluoai",
        w13_weight=w13.to(dev),
        w13_weight_scale=w13_s.to(dev),
        w2_weight=w2.to(dev),
        w2_weight_scale=w2_s.to(dev),
        w13_bias=b13.to(dev),
        w2_bias=b2.to(dev),
    )
    method = make_method(QaicGptOssMxfp4MoEMethod, has_bias=True)
    out = method.forward_oot(
        layer, x.to(dev), topk_weights.to(dev), topk_ids.to(dev)
    ).cpu()

    assert out.dtype == dtype
    rel = (out.float() - expected).norm() / expected.norm()
    assert rel < 1e-2, f"relative error {rel:.3e}"
