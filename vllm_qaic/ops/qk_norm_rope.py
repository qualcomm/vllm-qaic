# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Fused per-head QK RMS norm + NeoX RoPE (+ V norm) for QAIC attention layers.

``torch.ops.qaic.qk_norm_rope_kv_insert`` fuses, in one dispatch, what an
attention layer does between its QKV projection and attention: per-head RMS
norm of q and k, NeoX RoPE on both, and an optional plain RMS norm of v. It can
also write k / v into the paged KV cache, but that part is not used here: no
``key_cache`` / ``value_cache`` / ``slot_mapping`` is passed, so the op only
writes ``query_out`` / ``key_out`` / ``value_out``. The attention backend keeps
writing the cache with its own ``reshape_and_cache``, and the KV-cache layout is
unchanged.

Callers (patched in :func:`register_qaic_qk_norm_rope`):

- ``Gemma4Attention.forward``: q / k norm + RoPE + v norm (no weight); the
  KV-shared layers do q only.
- ``Qwen3NextAttention._project_qkv_gate`` (Qwen3.5 / Qwen3.6): Gemma-style q / k
  norm (``1 + w``, ``weight_offset=1``) + partial RoPE. Interleaved MRoPE
  reduces to 1-D RoPE on ``positions[0]`` when the model is text only (every
  token then has equal T / H / W positions).
- ``OAIAttention.forward`` (GPT-OSS): RoPE only (``norm_qk=False``); the YaRN
  factors are already in ``cos_sin_cache``.

The fused op is always used where it applies. Only inputs it cannot express
(e.g. a RoPE or norm variant it lacks, or real multimodal MRoPE positions) run
the layer's unfused PyTorch sequence; errors from the op are raised.
"""

from __future__ import annotations

import inspect

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

_PATCH_APPLIED_ATTR = "_qaic_qk_norm_rope_patch_applied"
_SUPPORTED_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_MAX_HEAD_SIZE = 512
# Per-module result of the shape / type checks that do not depend on the input.
_MODULE_OK_ATTR = "_qaic_qk_norm_rope_ok"


def _op_available() -> bool:
    return hasattr(torch.ops.qaic, "qk_norm_rope_kv_insert")


def qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    q_weight: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    head_size: int,
    rotary_dim: int,
    eps: float,
    weight_offset: float = 0.0,
    norm_qk: bool = True,
    norm_v: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None:
    """Per-head q / k RMS norm + NeoX RoPE (+ v RMS norm) in one QAIC dispatch.

    Args:
        query: [num_tokens, num_heads * head_size]; column views of a fused qkv
            tensor are fine (the row stride may exceed the width).
        key: [num_tokens, num_kv_heads * head_size], or None.
        value: [num_tokens, num_kv_heads * head_size], or None. Only used with
            ``norm_v`` (without a cache the op has nowhere to copy an
            un-normed v).
        q_weight / k_weight: [head_size] norm weights, or None for weight 1.
            Ignored when ``norm_qk`` is False.
        cos_sin_cache: [max_position, rotary_dim] = [cos | sin], query dtype.
        positions: [num_tokens], int32 or int64.
        weight_offset: the norm scales by ``w + weight_offset`` (1 for Gemma-style
            ``1 + w`` weights).
        norm_qk: False applies RoPE only.

    Returns:
        ``(query_out, key_out, value_out)`` (None for an absent key / value), or
        None when the op does not support the inputs; the caller then runs its
        unfused path. Errors from the op propagate.
    """
    if torch.compiler.is_compiling():
        return None
    dtype = query.dtype
    num_tokens = query.shape[0] if query.dim() == 2 else -1
    if (
        query.device.type != "qaic"
        or dtype not in _SUPPORTED_DTYPES
        or query.dim() != 2
        or positions.dim() != 1
        or positions.shape[0] != num_tokens
        or positions.dtype not in (torch.int32, torch.int64)
        or head_size % 64 != 0
        or head_size > _MAX_HEAD_SIZE
        or rotary_dim % 2 != 0
        or not 0 < rotary_dim <= head_size
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[-1] != rotary_dim
        or cos_sin_cache.dtype != dtype
        or (value is not None and not norm_v)
        or (value is not None and key is None)
    ):
        return None
    if not norm_qk:
        q_weight = k_weight = None
    for t in (key, value):
        if t is not None and (
            t.dim() != 2 or t.shape[0] != num_tokens or t.dtype != dtype
        ):
            return None
    for w in (q_weight, k_weight):
        if w is not None and (w.dtype != dtype or w.shape != (head_size,)):
            return None
    for t in (query, key, value):
        if t is not None and (t.shape[1] % head_size != 0 or t.stride(-1) != 1):
            return None

    if not _op_available():
        return None

    query_out = torch.empty(query.shape, dtype=dtype, device=query.device)
    key_out = (
        torch.empty(key.shape, dtype=dtype, device=key.device)
        if key is not None
        else query.new_empty(0)
    )
    value_out = (
        torch.empty(value.shape, dtype=dtype, device=value.device)
        if value is not None
        else query.new_empty(0)
    )
    torch.ops.qaic.qk_norm_rope_kv_insert(
        query,
        key,
        value,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        head_size,
        rotary_dim,
        eps,
        weight_offset,
        query_out,
        key_out,
        value_out,
        norm_qk=norm_qk,
        norm_v=norm_v,
    )
    return (
        query_out,
        key_out if key is not None else None,
        value_out if value is not None else None,
    )


# --------------------------------------------------------------------------
# Static (input-independent) checks on the layer's norm and RoPE modules.
# --------------------------------------------------------------------------


def _plain_neox_rope_types() -> tuple[type, ...]:
    """RoPE classes whose forward is the base NeoX rotation of the cos/sin table.

    Their only difference from ``RotaryEmbedding`` is how ``cos_sin_cache`` is
    built. Exact types are matched so a subclass that changes the rotation
    itself is never fused.
    """
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    types: list[type] = [RotaryEmbedding]
    try:
        from vllm.model_executor.layers.rotary_embedding.gemma4_rope import (
            Gemma4RotaryEmbedding,
        )

        types.append(Gemma4RotaryEmbedding)
    except ImportError:
        pass
    try:
        from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
            YaRNScalingRotaryEmbedding,
        )

        types.append(YaRNScalingRotaryEmbedding)
    except ImportError:
        pass
    from .rotary_embedding import QAicRotaryEmbedding, QAicYaRNScalingRotaryEmbedding

    types += [QAicRotaryEmbedding, QAicYaRNScalingRotaryEmbedding]
    return tuple(types)


def _mrope_types() -> tuple[type, ...]:
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    from .mrope import QAicMRotaryEmbedding

    return (MRotaryEmbedding, QAicMRotaryEmbedding)


def _rope_ok(rope, head_size: int, allow_mrope: bool = False) -> bool:
    allowed = _plain_neox_rope_types()
    if allow_mrope:
        allowed = allowed + _mrope_types()
    return (
        type(rope) in allowed
        and getattr(rope, "is_neox_style", False)
        and getattr(rope, "head_size", None) == head_size
        and isinstance(getattr(rope, "rotary_dim", None), int)
        and getattr(rope, "cos_sin_cache", None) is not None
    )


def _norm_weight_offset(norm, head_size: int, has_weight: bool = True):
    """Return the op's ``weight_offset`` for a vLLM norm module, or None.

    ``RMSNorm`` scales by ``w`` (offset 0), ``GemmaRMSNorm`` by ``1 + w``
    (offset 1). Anything else is not fused.
    """
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

    if isinstance(norm, GemmaRMSNorm):
        offset = 1.0
        norm_has_weight = True
    elif isinstance(norm, RMSNorm):
        if getattr(norm, "variance_size_override", None) is not None:
            return None
        offset = 0.0
        norm_has_weight = getattr(norm, "has_weight", True)
    else:
        return None
    if norm_has_weight != has_weight:
        return None
    if has_weight:
        weight = getattr(norm, "weight", None)
        if weight is None or tuple(weight.shape) != (head_size,):
            return None
    return offset


def _module_ok(module, check) -> bool:
    ok = module.__dict__.get(_MODULE_OK_ATTR)
    if ok is None:
        try:
            ok = bool(check(module))
        except Exception as e:  # never break the model on a failed check
            logger.debug("qk_norm_rope check failed for %s: %s", type(module), e)
            ok = False
        module.__dict__[_MODULE_OK_ATTR] = ok
    return ok


# --------------------------------------------------------------------------
# Gemma4Attention
# --------------------------------------------------------------------------


def _gemma4_check(self) -> bool:
    d = self.head_dim
    if _norm_weight_offset(self.q_norm, d) != 0.0:
        return False
    if not self.is_kv_shared_layer:
        if _norm_weight_offset(self.k_norm, d) != 0.0:
            return False
        if _norm_weight_offset(self.v_norm, d, has_weight=False) != 0.0:
            return False
        eps = self.q_norm.variance_epsilon
        if self.k_norm.variance_epsilon != eps or self.v_norm.variance_epsilon != eps:
            return False
    return _rope_ok(self.rotary_emb, d) and self.rotary_emb.rotary_dim == d


def _gemma4_norm_rope_eager(self, positions, q, k, v):
    # Upstream Gemma4Attention.forward between the projection and attention.
    q = q.unflatten(-1, (self.num_heads, self.head_dim))
    q = self.q_norm(q)
    q = q.flatten(-2, -1)
    if k is None:
        q, _ = self.rotary_emb(positions, q, None)
        return q, None, None
    k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
    k = self.k_norm(k)
    k = k.flatten(-2, -1)
    q, k = self.rotary_emb(positions, q, k)
    v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
    v = self.v_norm(v)
    v = v.flatten(-2, -1)
    return q, k, v


def _make_gemma4_forward(orig_forward):
    def forward(self, positions, hidden_states, **kwargs):
        if not _module_ok(self, _gemma4_check):
            return orig_forward(self, positions, hidden_states, **kwargs)
        if self.is_kv_shared_layer:
            q, _ = self.q_proj(hidden_states)
            k = v = None
        else:
            # For k_eq_v, K weights are loaded into both K and V slots of
            # qkv_proj, so V == K automatically.
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        rope = self.rotary_emb
        out = qk_norm_rope(
            q,
            k,
            v,
            self.q_norm.weight,
            None if k is None else self.k_norm.weight,
            rope._match_cos_sin_cache_dtype(q),
            positions,
            self.head_dim,
            rope.rotary_dim,
            self.q_norm.variance_epsilon,
            weight_offset=0.0,
            norm_qk=True,
            norm_v=True,
        )
        if out is None:
            out = _gemma4_norm_rope_eager(self, positions, q, k, v)
        q, k, v = out
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    return forward


# --------------------------------------------------------------------------
# Qwen3NextAttention (Qwen3-Next, Qwen3.5, Qwen3.6)
# --------------------------------------------------------------------------


def _qwen3_next_check(self) -> bool:
    d = self.head_dim
    offset = _norm_weight_offset(self.q_norm, d)
    return (
        offset is not None
        and _norm_weight_offset(self.k_norm, d) == offset
        and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
        and self.dual_chunk_attention_config is None
        and not getattr(self, "use_fused_qk_norm_rope_gate", False)
        and _rope_ok(self.rotary_emb, d, allow_mrope=True)
    )


def _qwen3_next_positions(self, positions):
    """1-D positions for the op, or None when the layer needs real MRoPE."""
    if positions.dim() == 1:
        return positions
    if (
        positions.dim() == 2
        and isinstance(self.rotary_emb, _mrope_types())
        and self.__dict__.get("_qaic_text_only", False)
    ):
        # Text only: the T / H / W rows are equal, so interleaved or sectioned
        # MRoPE is the 1-D rotation at positions[0].
        return positions[0]
    return None


def _make_qwen3_next_init(orig_init):
    sig = inspect.signature(orig_init)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        try:
            model_config = sig.bind(self, *args, **kwargs).arguments.get("model_config")
        except TypeError:
            model_config = None
        mm_config = getattr(model_config, "multimodal_config", None)
        text_only = model_config is not None and (
            mm_config is None or bool(getattr(mm_config, "language_model_only", False))
        )
        self.__dict__["_qaic_text_only"] = text_only

    return __init__


def _make_qwen3_next_project_qkv_gate(orig_project):
    def _project_qkv_gate(self, qkv, positions):
        if not _module_ok(self, _qwen3_next_check):
            return orig_project(self, qkv, positions)
        op_positions = _qwen3_next_positions(self, positions)
        if op_positions is None:
            return orig_project(self, qkv, positions)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        rope = self.rotary_emb
        out = qk_norm_rope(
            q,
            k,
            None,
            self.q_norm.weight,
            self.k_norm.weight,
            rope._match_cos_sin_cache_dtype(q),
            op_positions,
            self.head_dim,
            rope.rotary_dim,
            self.q_norm.variance_epsilon,
            weight_offset=_norm_weight_offset(self.q_norm, self.head_dim),
            norm_qk=True,
            norm_v=False,
        )
        if out is not None:
            q, k, _ = out
            return q, k, v, gate
        # Upstream eager path.
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    return _project_qkv_gate


# --------------------------------------------------------------------------
# OAIAttention (GPT-OSS): RoPE only
# --------------------------------------------------------------------------


def _oai_check(self) -> bool:
    return _rope_ok(self.rotary_emb, self.head_dim)


def _make_oai_forward(orig_forward):
    def forward(self, hidden_states, positions):
        if not _module_ok(self, _oai_check):
            return orig_forward(self, hidden_states, positions)
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        rope = self.rotary_emb
        out = qk_norm_rope(
            q,
            k,
            None,
            None,
            None,
            rope._match_cos_sin_cache_dtype(q),
            positions,
            self.head_dim,
            rope.rotary_dim,
            0.0,
            norm_qk=False,
            norm_v=False,
        )
        if out is None:
            q, k = rope(positions, q, k)
        else:
            q, k, _ = out
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    return forward


def _patch(cls, attr: str, make) -> None:
    orig = cls.__dict__.get(attr)
    if orig is None:
        logger.debug("qk_norm_rope: %s.%s not found; not patched", cls.__name__, attr)
        return
    patched = make(orig)
    patched.__wrapped__ = orig  # the upstream method, for fallbacks and tests
    setattr(cls, attr, patched)


def register_qaic_qk_norm_rope() -> None:
    """Route the norm + RoPE of the supported attention layers to the fused op."""
    import vllm.model_executor.layers.rotary_embedding as rope_mod

    if getattr(rope_mod, _PATCH_APPLIED_ATTR, False):
        return
    setattr(rope_mod, _PATCH_APPLIED_ATTR, True)

    try:
        from vllm.model_executor.models.gemma4 import Gemma4Attention

        _patch(Gemma4Attention, "forward", _make_gemma4_forward)
    except Exception as e:  # pragma: no cover - model module unavailable
        logger.debug("qk_norm_rope: Gemma4Attention not patched: %s", e)
    try:
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

        _patch(Qwen3NextAttention, "__init__", _make_qwen3_next_init)
        _patch(
            Qwen3NextAttention, "_project_qkv_gate", _make_qwen3_next_project_qkv_gate
        )
    except Exception as e:  # pragma: no cover
        logger.debug("qk_norm_rope: Qwen3NextAttention not patched: %s", e)
    try:
        from vllm.model_executor.models.gpt_oss import OAIAttention

        _patch(OAIAttention, "forward", _make_oai_forward)
    except Exception as e:  # pragma: no cover
        logger.debug("qk_norm_rope: OAIAttention not patched: %s", e)
