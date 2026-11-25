# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Hardware checks for the fused QK norm + RoPE path (qaic::qk_norm_rope_kv_insert
without its KV-cache insert).

The layer tests run the patched attention method and the upstream one on the
same layer object, with the real vLLM / vllm_qaic norm and RoPE modules, and
compare what each hands to attention.
"""

from types import SimpleNamespace

import pytest
import torch

from .moe_test_utils import requires_qaic

DEV = "qaic"
EPS = 1e-6


def _ref_norm_rope(x, w, offset, cache, pos, d, rd, norm):
    """fp32 CPU reference: per-head RMS norm (optional) + NeoX RoPE."""
    n = x.shape[0]
    h = x.view(n, -1, d).float()
    if norm:
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
        if w is not None:
            h = h * (w.float() + offset)
        h = h.to(x.dtype).float()
    cs = cache.float()[pos.long()]
    cos, sin = cs.chunk(2, dim=-1)
    cos, sin = cos[:, None], sin[:, None]
    half = rd // 2
    out = h.clone()
    out[..., :half] = h[..., :half] * cos - h[..., half:rd] * sin
    out[..., half:rd] = h[..., half:rd] * cos + h[..., :half] * sin
    return out.reshape(n, -1)


@requires_qaic
@pytest.mark.parametrize(
    "nq,nkv,d,rd,offset,norm_qk,with_v",
    [
        (4, 2, 256, 256, 0.0, True, True),  # Gemma4 sliding, TP4
        (4, 1, 512, 512, 0.0, True, True),  # Gemma4 full, TP4
        (4, 1, 256, 64, 1.0, True, False),  # Qwen3.6, TP4 (partial rotary)
        (64, 8, 64, 64, 0.0, False, False),  # GPT-OSS, TP1 (RoPE only)
    ],
)
@pytest.mark.parametrize("tokens", [1, 33])
def test_qk_norm_rope_matches_reference(
    nq, nkv, d, rd, offset, norm_qk, with_v, tokens
):
    from vllm_qaic.ops.qk_norm_rope import qk_norm_rope

    torch.manual_seed(0)
    qkv = torch.randn(tokens, (nq + 2 * nkv) * d, dtype=torch.float16)
    q, k, v = qkv.split([nq * d, nkv * d, nkv * d], dim=-1)
    qw = (torch.randn(d) * 0.1).half() if norm_qk else None
    kw = (torch.randn(d) * 0.1).half() if norm_qk else None
    cache = torch.randn(4096, rd).clamp(-1, 1).half()
    pos = torch.randint(0, 4096, (tokens,), dtype=torch.int64)

    dq, dk, dv = qkv.to(DEV).split([nq * d, nkv * d, nkv * d], dim=-1)
    out = qk_norm_rope(
        dq,
        dk,
        dv if with_v else None,
        qw.to(DEV) if norm_qk else None,
        kw.to(DEV) if norm_qk else None,
        cache.to(DEV),
        pos.to(DEV),
        d,
        rd,
        EPS,
        weight_offset=offset,
        norm_qk=norm_qk,
        norm_v=with_v,
    )
    assert out is not None, "fused op rejected a supported configuration"
    qo, ko, vo = (t.cpu().float() if t is not None else None for t in out)

    tol = {"rtol": 1e-2, "atol": 1e-2}
    torch.testing.assert_close(
        qo, _ref_norm_rope(q, qw, offset, cache, pos, d, rd, norm_qk), **tol
    )
    torch.testing.assert_close(
        ko, _ref_norm_rope(k, kw, offset, cache, pos, d, rd, norm_qk), **tol
    )
    if with_v:
        vf = v.view(tokens, -1, d).float()
        ref_v = (vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + EPS)).reshape(
            tokens, -1
        )
        torch.testing.assert_close(vo, ref_v, **tol)
    else:
        assert vo is None


@requires_qaic
def test_qk_norm_rope_rejects_unsupported():
    from vllm_qaic.ops.qk_norm_rope import qk_norm_rope

    q = torch.randn(2, 4 * 96, dtype=torch.float16, device=DEV)  # D % 64 != 0
    cache = torch.randn(16, 96, dtype=torch.float16, device=DEV)
    pos = torch.zeros(2, dtype=torch.int64, device=DEV)
    assert qk_norm_rope(q, None, None, None, None, cache, pos, 96, 96, EPS) is None
    # 2-D (MRoPE) positions are the caller's business.
    q = torch.randn(2, 4 * 64, dtype=torch.float16, device=DEV)
    cache = torch.randn(16, 64, dtype=torch.float16, device=DEV)
    pos3 = torch.zeros(3, 2, dtype=torch.int64, device=DEV)
    assert qk_norm_rope(q, None, None, None, None, cache, pos3, 64, 64, EPS) is None


@requires_qaic
@pytest.mark.parametrize("heads,d,rd", [(4, 256, 64), (16, 256, 64), (4, 128, 64)])
def test_apply_rotary_emb_strided_partial_rotary(heads, d, rd):
    """``x[..., :rotary_dim]`` of a multi-head tensor is a strided view; the QAIC
    rotation must match the reference on it (the raw op does not)."""
    from vllm_qaic.ops.rotary_embedding import QAicApplyRotaryEmb

    torch.manual_seed(0)
    n = 5
    x = torch.randn(n, heads, d, dtype=torch.float16)
    cos = torch.randn(n, rd // 2, dtype=torch.float16)
    sin = torch.randn(n, rd // 2, dtype=torch.float16)
    xd = x.to(DEV)[..., :rd]
    assert not xd.is_contiguous()
    out = QAicApplyRotaryEmb.forward_static(xd, cos.to(DEV), sin.to(DEV), True)

    xr = x[..., :rd].float()
    x1, x2 = xr.chunk(2, dim=-1)
    c, s = cos.float()[:, None], sin.float()[:, None]
    ref = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    torch.testing.assert_close(out.cpu().float(), ref, rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------
# Layer-level: patched method vs upstream method on the same layer object.
# --------------------------------------------------------------------------


class _Capture(torch.nn.Module):
    """Stands in for Attention / o_proj and returns what it was given."""

    def forward(self, q, k=None, v=None):
        return torch.cat([t for t in (q, k, v) if t is not None], dim=-1)


class _Proj(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self.out = out

    def forward(self, _x):
        return self.out, None


class _Ident(torch.nn.Module):
    def forward(self, x):
        return x, None


def _vllm_ctx():
    # CustomOp construction only reads ``compilation_config``; a full
    # VllmConfig() would run the QAIC platform hook, which needs a model config.
    from vllm.config import CompilationConfig, set_current_vllm_config

    return set_current_vllm_config(
        SimpleNamespace(compilation_config=CompilationConfig(custom_ops=["all"]))
    )


def _qaic_ops():
    from vllm_qaic.ops import register_qaic_customop

    register_qaic_customop()


def _upstream(fn):
    return getattr(fn, "__wrapped__", fn)


def _patched(fn, make):
    # The patched method if registration already ran, else wrap it here.
    return fn if hasattr(fn, "__wrapped__") else make(fn)


def _to_dev(mod):
    return mod.to(device=DEV, dtype=torch.float16)


@requires_qaic
@pytest.mark.parametrize("layer", ["sliding", "full", "kv_shared"])
@pytest.mark.parametrize("tokens", [1, 17])
def test_gemma4_attention_fused_matches_upstream(layer, tokens):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.gemma4 import Gemma4Attention

    from vllm_qaic.ops import qk_norm_rope as qnr

    _qaic_ops()
    torch.manual_seed(0)
    with _vllm_ctx():
        if layer == "full":
            d, nq, nkv = 512, 4, 1
            rope_params = {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1e6,
            }
        else:
            d, nq, nkv = 256, 4, 2
            rope_params = {"rope_type": "default", "rope_theta": 1e4}
        shared = layer == "kv_shared"
        width = nq * d if shared else (nq + 2 * nkv) * d
        qkv = torch.randn(tokens, width, dtype=torch.float16, device=DEV)
        mod = torch.nn.Module()
        mod.__dict__.update(
            num_heads=nq,
            num_kv_heads=nkv,
            head_dim=d,
            q_size=nq * d,
            kv_size=nkv * d,
            is_kv_shared_layer=shared,
        )
        mod.q_norm = _to_dev(RMSNorm(d, eps=EPS))
        mod.k_norm = _to_dev(RMSNorm(d, eps=EPS))
        mod.v_norm = _to_dev(RMSNorm(d, eps=EPS, has_weight=False))
        with torch.no_grad():
            mod.q_norm.weight.copy_(torch.randn(d) * 0.2 + 1)
            mod.k_norm.weight.copy_(torch.randn(d) * 0.2 + 1)
        mod.rotary_emb = get_rope(
            d,
            max_position=8192,
            rope_parameters=rope_params,
            is_neox_style=True,
            dtype=torch.float16,
        ).to(DEV)
        if shared:
            mod.q_proj = _Proj(qkv)
        else:
            mod.qkv_proj = _Proj(qkv)
        mod.attn = _Capture()
        mod.o_proj = _Ident()
        positions = torch.randint(0, 8192, (tokens,), dtype=torch.int64, device=DEV)

        assert qnr._gemma4_check(mod)
        ref = _upstream(Gemma4Attention.forward)(mod, positions, None)
        fused = _patched(Gemma4Attention.forward, qnr._make_gemma4_forward)
        out = fused(mod, positions, None)
    torch.testing.assert_close(
        out.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2
    )


@requires_qaic
@pytest.mark.parametrize("positions_2d", [False, True])
@pytest.mark.parametrize("tokens", [1, 17])
def test_qwen3_next_attention_fused_matches_upstream(positions_2d, tokens):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

    from vllm_qaic.ops import qk_norm_rope as qnr

    _qaic_ops()
    torch.manual_seed(0)
    d, nq, nkv = 256, 4, 1
    with _vllm_ctx():
        mod = torch.nn.Module()
        mod.__dict__.update(
            num_heads=nq,
            num_kv_heads=nkv,
            head_dim=d,
            q_size=nq * d,
            kv_size=nkv * d,
            attn_output_gate=True,
            dual_chunk_attention_config=None,
            use_fused_qk_norm_rope_gate=False,
            _qaic_text_only=True,
        )
        mod.q_norm = _to_dev(GemmaRMSNorm(d, eps=EPS))
        mod.k_norm = _to_dev(GemmaRMSNorm(d, eps=EPS))
        with torch.no_grad():
            mod.q_norm.weight.copy_(torch.randn(d) * 0.2)
            mod.k_norm.weight.copy_(torch.randn(d) * 0.2)
        mod.rotary_emb = get_rope(
            head_size=d,
            max_position=8192,
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "partial_rotary_factor": 0.25,
                "rope_theta": 1e7,
                "rope_type": "default",
            },
            dtype=torch.float16,
        ).to(DEV)
        qkv = torch.randn(
            tokens, (2 * nq + 2 * nkv) * d, dtype=torch.float16, device=DEV
        )
        pos1 = torch.randint(0, 8192, (tokens,), dtype=torch.int64)
        positions = (pos1.expand(3, -1).contiguous() if positions_2d else pos1).to(DEV)

        assert qnr._qwen3_next_check(mod)
        ref = _upstream(Qwen3NextAttention._project_qkv_gate)(mod, qkv, positions)
        fused = _patched(
            Qwen3NextAttention._project_qkv_gate,
            qnr._make_qwen3_next_project_qkv_gate,
        )
        out = fused(mod, qkv, positions)
    for o, r in zip(out, ref, strict=True):
        torch.testing.assert_close(
            o.cpu().float(), r.cpu().float(), rtol=1e-2, atol=1e-2
        )


def test_qwen3_next_positions_selection():
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    from vllm_qaic.ops import qk_norm_rope as qnr

    mrope = object.__new__(MRotaryEmbedding)  # only its type is inspected
    pos = torch.arange(12, dtype=torch.int64).view(3, 4)
    # 1-D positions pass through.
    text = SimpleNamespace(rotary_emb=mrope, _qaic_text_only=True)
    p0 = pos[0]
    assert qnr._qwen3_next_positions(text, p0) is p0
    # Text only: T / H / W rows are equal, the op uses row 0.
    assert torch.equal(qnr._qwen3_next_positions(text, pos), pos[0])
    # Multimodal: real MRoPE needed, stays on the unfused path.
    mm = SimpleNamespace(rotary_emb=mrope, _qaic_text_only=False)
    assert qnr._qwen3_next_positions(mm, pos) is None
    # 2-D positions without an MRoPE module are never collapsed.
    plain = SimpleNamespace(rotary_emb=object(), _qaic_text_only=True)
    assert qnr._qwen3_next_positions(plain, pos) is None


@requires_qaic
@pytest.mark.parametrize("tokens", [1, 17])
def test_oai_attention_rope_only_matches_upstream(tokens):
    from vllm.model_executor.layers.rotary_embedding import get_rope
    from vllm.model_executor.models.gpt_oss import OAIAttention

    from vllm_qaic.ops import qk_norm_rope as qnr

    _qaic_ops()
    torch.manual_seed(0)
    d, nq, nkv = 64, 64, 8
    with _vllm_ctx():
        mod = torch.nn.Module()
        mod.__dict__.update(head_dim=d, q_size=nq * d, kv_size=nkv * d)
        mod.rotary_emb = get_rope(
            d,
            max_position=131072,
            dtype=torch.float32,
            rope_parameters={
                "rope_theta": 150000,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
            },
            is_neox_style=True,
        ).to(DEV)
        qkv = torch.randn(tokens, (nq + 2 * nkv) * d, dtype=torch.float16, device=DEV)
        mod.qkv_proj = _Proj(qkv)
        mod.attn = _Capture()
        mod.o_proj = _Ident()
        positions = torch.randint(0, 131072, (tokens,), dtype=torch.int64, device=DEV)

        assert qnr._oai_check(mod)
        ref = _upstream(OAIAttention.forward)(mod, None, positions)
        fused = _patched(OAIAttention.forward, qnr._make_oai_forward)
        out = fused(mod, None, positions)
    torch.testing.assert_close(
        out.cpu().float(), ref.cpu().float(), rtol=1e-2, atol=1e-2
    )
