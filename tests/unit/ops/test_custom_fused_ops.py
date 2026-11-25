# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Hardware checks for torch.ops.qaic fused ops used by vllm_qaic layers."""

import pytest
import torch

from .moe_test_utils import requires_qaic


@requires_qaic
@pytest.mark.parametrize("op", ["gelu_and_mul", "gelu_tanh_and_mul"])
def test_gelu_and_mul(op):
    x = torch.randn(4, 128, dtype=torch.float16)
    approx = "tanh" if "tanh" in op else "none"
    ref = (
        torch.nn.functional.gelu(x[:, :64].float(), approximate=approx)
        * x[:, 64:].float()
    )
    out = getattr(torch.ops.qaic, op)(x.to("qaic")).cpu()
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)


@requires_qaic
@pytest.mark.parametrize("gemma", [False, True])
def test_add_rms_norm(gemma):
    h, eps = 1024, 1e-6
    x = torch.randn(8, h, dtype=torch.float16)
    r = torch.randn(8, h, dtype=torch.float16)
    w = torch.randn(h, dtype=torch.float16)
    s = x.float() + r.float()
    ww = w.float() + 1 if gemma else w.float()
    ref = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * ww
    out, res = torch.ops.qaic.add_rms_norm(
        x.to("qaic"), [h], w.to("qaic"), eps, r.to("qaic"), gemma
    )
    torch.testing.assert_close(out.cpu().float(), ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(res.cpu().float(), s, rtol=2e-3, atol=2e-3)


@requires_qaic
def test_topk_softmax():
    t, e, k = 16, 64, 4
    g = torch.randn(t, e, dtype=torch.float32)
    w = torch.empty(t, k, dtype=torch.float32, device="qaic")
    ids = torch.empty(t, k, dtype=torch.int32, device="qaic")
    torch.ops.qaic.topk_softmax(w, ids, g.to("qaic"), False, None)
    rw, ri = torch.topk(torch.softmax(g, -1), k)
    torch.testing.assert_close(w.cpu(), rw, rtol=1e-3, atol=1e-3)
    assert torch.equal(ids.cpu().long().sort(-1).values, ri.sort(-1).values)


@requires_qaic
# 1 and 8 take the index_add_ combine, 160 the argsort combine.
@pytest.mark.parametrize("tokens", [1, 8, 160])
def test_gpt_oss_mxfp4_fused_experts_matches_per_expert(tokens):
    from types import SimpleNamespace

    from vllm_qaic.quantization import qaic_gpt_oss_mxfp4_config as cfg

    e, h, i, k = 8, 256, 256, 4
    torch.manual_seed(0)

    def dev(t):
        return t.to("qaic")

    layer = SimpleNamespace(
        w13_weight=dev(torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8)),
        w13_weight_scale=dev(
            torch.randint(118, 122, (e, 2 * i, h // 32), dtype=torch.uint8)
        ),
        w2_weight=dev(torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8)),
        w2_weight_scale=dev(
            torch.randint(118, 122, (e, h, i // 32), dtype=torch.uint8)
        ),
        w13_bias=dev((torch.randn(e, 2 * i) * 0.1).half()),
        w2_bias=dev((torch.randn(e, h) * 0.1).half()),
        activation="swigluoai",
        apply_router_weight_on_input=False,
        expert_map=None,
    )
    m = object.__new__(cfg.QaicGptOssMxfp4MoEMethod)
    m.moe = SimpleNamespace(has_bias=True)
    x = dev((torch.randn(tokens, h) * 0.5).half())
    tw, ti = torch.topk(torch.softmax(torch.randn(tokens, e), -1), k)
    tw, ti = dev((tw / tw.sum(-1, keepdim=True)).half()), dev(ti)
    # Run per-expert once before and once after _fused_experts_supported()
    # reorders W13 for the fused kernel. A concat kernel gets in-place
    # reordered weights, and the per-expert path has to undo that.
    before = m._forward_per_expert(layer, x, tw, ti).cpu().float()
    assert m._fused_experts_supported(layer, x, "swigluoai")
    assert cfg._fused_w13_layout_cache in (cfg._W13_INTERLEAVED, cfg._W13_CONCAT)
    ref = m._forward_per_expert(layer, x, tw, ti).cpu().float()
    torch.testing.assert_close(ref, before, rtol=0, atol=0)
    out = m._forward_fused_experts(layer, x, tw, ti).cpu().float()
    # Several rows per token must all be accumulated (index_put_ would drop them).
    torch.testing.assert_close(out, ref, rtol=0, atol=5e-3 * ref.abs().max().item())


def _ref_topk_softmax(g, k, renorm, bias=None):
    p = torch.softmax(g.float(), -1)
    sel = p + bias if bias is not None else p
    ids = sel.topk(k, -1).indices
    w = p.gather(1, ids)
    if renorm:
        w = w / w.sum(-1, keepdim=True)
    return w, ids


def _run_router(g, k, renorm, bias=None, is_padding=None):
    from vllm_qaic.ops import topk_router

    t = g.shape[0]
    w = torch.empty(t, k, dtype=torch.float32, device="qaic")
    ids = torch.empty(t, k, dtype=torch.int32, device="qaic")
    tei = torch.empty(t, k, dtype=torch.int32, device="qaic")
    ok = topk_router._try_qaic_topk_softmax(
        w,
        ids,
        tei,
        g.to("qaic"),
        renorm,
        None if bias is None else bias.to("qaic"),
        None if is_padding is None else is_padding.to("qaic"),
    )
    return ok, w.cpu(), ids.cpu().long()


def _assert_same_selection(w, ids, rw, ri):
    o, ro = ids.argsort(-1), ri.argsort(-1)
    assert torch.equal(ids.gather(1, o), ri.gather(1, ro))
    torch.testing.assert_close(w.gather(1, o), rw.gather(1, ro), rtol=1e-3, atol=1e-3)


@requires_qaic
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("renorm", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_qaic_topk_softmax_router(dtype, renorm, with_bias):
    t, e, k = 32, 128, 8
    g = torch.randn(t, e).to(dtype)
    bias = torch.randn(e) * 0.05 if with_bias else None
    ok, w, ids = _run_router(g, k, renorm, bias)
    assert ok
    rw, ri = _ref_topk_softmax(g, k, renorm, bias)
    _assert_same_selection(w, ids, rw, ri)


@requires_qaic
def test_qaic_topk_softmax_router_is_padding():
    t, e, k = 16, 64, 4
    g = torch.randn(t, e)
    pad = torch.zeros(t, dtype=torch.bool)
    pad[-5:] = True
    ok, w, ids = _run_router(g, k, True, is_padding=pad)
    assert ok
    assert (ids[pad] == -1).all()
    rw, ri = _ref_topk_softmax(g, k, True)
    _assert_same_selection(w[~pad], ids[~pad], rw[~pad], ri[~pad])


@requires_qaic
def test_qaic_topk_softmax_router_rejects_bad_bias():
    g = torch.randn(4, 64)
    ok, _, _ = _run_router(g, 4, False, bias=torch.randn(32))  # wrong size
    assert not ok


def test_topk_softmax_epilogue_detection(monkeypatch):
    from vllm_qaic.ops import topk_router

    monkeypatch.setattr(topk_router, "_TOPK_SOFTMAX_HAS_EPILOGUE", None)
    has = topk_router._topk_softmax_has_epilogue()
    op = (
        getattr(torch.ops.qaic, "topk_softmax", None)
        if hasattr(torch.ops, "qaic")
        else None
    )
    if op is None:
        assert has is False
    else:
        names = {a.name for a in op.default._schema.arguments}
        assert has == ({"token_expert_indices", "is_padding"} <= names)


def test_topk_softmax_patch_falls_back_off_device():
    """CPU gating must skip the QAIC kernel and use the eager path."""
    import vllm._custom_ops as ops
    from vllm_qaic.ops.topk_router import register_qaic_topk_router

    register_qaic_topk_router()
    t, e, k = 8, 32, 4
    g = torch.randn(t, e)
    w = torch.empty(t, k)
    ids = torch.empty(t, k, dtype=torch.int32)
    tei = torch.empty(t, k, dtype=torch.int32)
    ops.topk_softmax(w, ids, tei, g, True)
    rw, ri = _ref_topk_softmax(g, k, True)
    _assert_same_selection(w, ids.long(), rw, ri)


@requires_qaic
@pytest.mark.parametrize("tokens", [1, 16, 1024])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.float16])
def test_gpt_oss_moe_align_matches_sort_by_expert(tokens, w_dtype):
    """qaic::moe_align_block_size must be bit-identical to the eager grouping."""
    from vllm_qaic.quantization import qaic_gpt_oss_mxfp4_config as cfg

    if not hasattr(torch.ops.qaic, "moe_align_block_size"):
        pytest.skip("torch_qaic has no moe_align_block_size")
    e, h, k = 32, 2880, 4
    m = cfg.QaicGptOssMxfp4MoEMethod
    x = torch.randn(tokens, h).half().to("qaic")
    ids = torch.randn(tokens, e).topk(k, -1).indices.to(torch.int32).to("qaic")
    w = torch.rand(tokens, k).to(w_dtype).to("qaic")
    aligned = m._moe_align(x, w, ids, e)
    assert aligned is not None, "moe_align_block_size raised; see warning"
    ref = m._sort_by_expert(x, w, ids, e)
    for name, a, r in zip(
        ("bnd", "token_idx", "x_sorted", "w_sorted"), aligned, ref, strict=True
    ):
        assert a.shape == r.shape, name
        assert torch.equal(a.cpu().to(r.dtype), r.cpu()), name


def test_interleaved_to_concat():
    from vllm_qaic.quantization import qaic_gpt_oss_mxfp4_config as cfg

    t = torch.arange(2 * 6 * 3).view(2, 6, 3)
    exp = torch.cat((t[:, 0::2], t[:, 1::2]), dim=1)
    assert torch.equal(cfg._interleaved_to_concat(t), exp)
    # Rows (0, 2, 4) are gate and (1, 3, 5) are up.
    assert cfg._interleaved_to_concat(t)[0, :, 0].tolist() == [0, 6, 12, 3, 9, 15]
