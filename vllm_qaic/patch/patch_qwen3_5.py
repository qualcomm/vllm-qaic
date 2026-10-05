# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from __future__ import annotations

import os
import time

import torch

from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

# Env-gated timing for diagnosing the GDN forward cost. Set VLLM_QAIC_GDN_TIMING=1.
_GDN_TIMING = os.environ.get("VLLM_QAIC_GDN_TIMING", "0") == "1"


def _build_qaic_gdn_class():
    """Import upstream symbols lazily and return the QAIC GDN subclass.

    Imports are deferred to call time so that merely importing this patch module
    (e.g. during plugin registration) does not pull in the heavy mamba/gdn stack
    on platforms/configs where Qwen3.5 is never loaded.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    from vllm_qaic._custom_ops import (
        gdn_gating_hexagon,
        gdn_conv1d_update_hexagon,
        gdn_conv1d_prefill_hexagon,
        gdn_recurrent_decode_hexagon,
        gdn_recurrent_prefill_hexagon,
    )

    class QaicQwenGDNAttention(QwenGatedDeltaNetAttention):
        """Qwen3.5 GDN linear-attention layer for QAIC eager mode.

        Reuses the parent layer's projections / conv1d / norm / out_proj and the
        hybrid conv+ssm KV state; the causal conv, gating, and gated delta-rule
        recurrence run on the on-device Hexagon (HVX/HMX) NSP kernels.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # Parent __init__ selects forward_cuda on non-CPU/XPU/ROCm platforms
            # (QAIC is "OOT"), which needs triton. Rebind to the QAIC path.
            self._forward_method = self.forward_qaic

        def forward_qaic(
            self,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            # --- Part 1: input projections (identical to forward_cpu) ---
            assert not hasattr(self, "in_proj_qkv"), (
                "LoRA (split in_proj_qkv) is not supported on QAIC GDN."
            )
            assert not self.gqa_interleaved_layout, (
                "QAIC GDN expects the Qwen3.5 non-interleaved [q,k,v,z]/[b,a] "
                "layout (gqa_interleaved_layout=False)."
            )

            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
            ba, _ = self.in_proj_ba(hidden_states)

            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b, a = ba.chunk(2, dim=-1)

            num_tokens = hidden_states.size(0)
            core_attn_out = torch.zeros(
                (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            # --- Part 2: core GDN attention (on-device Hexagon kernels) ---
            if _GDN_TIMING:
                _t0 = time.perf_counter()
            logger.info_once(
                "[GDN-PATH] layer=%s using ON-DEVICE HEXAGON kernels "
                "(csrc/gdn_core/kernel.cpp).",
                self.prefix,
            )
            self._gdn_core_kernel(
                mixed_qkv,
                b,
                a,
                core_attn_out,
                is_conv_state_dim_first=is_conv_state_dim_first,
                get_forward_context=get_forward_context,
                GDNAttentionMetadata=GDNAttentionMetadata,
                gdn_gating_hexagon=gdn_gating_hexagon,
                gdn_conv1d_update_hexagon=gdn_conv1d_update_hexagon,
                gdn_conv1d_prefill_hexagon=gdn_conv1d_prefill_hexagon,
                gdn_recurrent_decode_hexagon=gdn_recurrent_decode_hexagon,
                gdn_recurrent_prefill_hexagon=gdn_recurrent_prefill_hexagon,
            )
            if _GDN_TIMING:
                logger.info(
                    "[GDN-TIMING] layer=%s num_tokens=%d core=%.1f ms "
                    "path=HEXAGON(kernel.cpp)",
                    self.prefix,
                    num_tokens,
                    (time.perf_counter() - _t0) * 1e3,
                )

            # --- Part 3: RMSNormGated + output projection ---
            z_shape_og = z.shape
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)
            core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)
            out, _ = self.out_proj(core_attn_out)
            return out

        def _gdn_core_kernel(
            self,
            mixed_qkv: torch.Tensor,
            b: torch.Tensor,
            a: torch.Tensor,
            core_attn_out: torch.Tensor,
            *,
            is_conv_state_dim_first,
            get_forward_context,
            GDNAttentionMetadata,
            gdn_gating_hexagon,
            gdn_conv1d_update_hexagon,
            gdn_conv1d_prefill_hexagon,
            gdn_recurrent_decode_hexagon,
            gdn_recurrent_prefill_hexagon,
        ) -> None:
            """On-device Hexagon (HVX/HMX) GDN core.

            The causal conv, gating, and gated delta-rule recurrence run on the
            NSP kernels in csrc/gdn_core/kernel.cpp. The conv/ssm state stays
            resident on-device — the kernels read and update it in place,
            addressed by KV slot index.
            """
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if attn_metadata is None:
                # Profile / warmup run: no metadata, leave zeros.
                return
            assert isinstance(attn_metadata, dict)
            m = attn_metadata[self.prefix]
            assert isinstance(m, GDNAttentionMetadata)
            if m.num_actual_tokens == 0:
                return
            assert m.spec_sequence_masks is None and m.num_accepted_tokens is None, (
                "speculative decode is not supported in QAIC GDN attention."
            )

            state_indices_tensor = m.non_spec_state_indices_tensor
            query_start_loc = m.non_spec_query_start_loc
            assert state_indices_tensor is not None
            assert query_start_loc is not None

            # conv_state: [num_slots, conv_dim, kernel-1] (dim-first) — the kernels
            # expect the dim-first layout; transpose the SD layout to match.
            conv_state = self.kv_cache[0]
            if not is_conv_state_dim_first():
                conv_state = conv_state.transpose(-1, -2)
            conv_weights = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )
            ssm_state = self.kv_cache[1]  # [num_slots, HV, V, K]

            mixed_qkv = mixed_qkv.contiguous()
            a = a.contiguous()
            b = b.contiguous()

            num_decodes = m.num_decodes
            num_decode_tokens = m.num_decode_tokens
            num_prefills = m.num_prefills
            num_prefill_tokens = m.num_prefill_tokens

            scale = self.head_k_dim**-0.5
            activation = self.activation

            # --- decode requests (batched, 1 token each) ---
            if num_decodes > 0:
                decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
                decode_a = a[:num_decode_tokens]
                decode_b = b[:num_decode_tokens]
                decode_state_indices = state_indices_tensor[:num_decodes]

                # Causal conv update. The kernel indexes conv_state by slot
                # internally and rolls the state in place, so we pass the whole
                # cache + slot ids — no per-token gather/scatter. The
                # gather+scatter this replaces was ~1.2 s/layer/token in eager
                # mode; the kernel itself is ~15 ms.
                decode_conv_out = gdn_conv1d_update_hexagon(
                    x=decode_mixed_qkv,
                    conv_state=conv_state,
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    slot_ids=decode_state_indices,
                    activation=activation,
                )

                query, key, value = self.rearrange_mixed_qkv(decode_conv_out)
                g, beta = gdn_gating_hexagon(
                    decode_a, decode_b, self.A_log, self.dt_bias
                )

                attn_out = gdn_recurrent_decode_hexagon(
                    q=query.squeeze(0),
                    k=key.squeeze(0),
                    v=value.squeeze(0),
                    g=g,
                    beta=beta,
                    ssm_state=ssm_state,
                    slot_ids=decode_state_indices,
                    scale=scale,
                )
                core_attn_out[:num_decode_tokens] = attn_out.to(core_attn_out.dtype)

            # --- prefill requests (varlen) ---
            if num_prefills > 0:
                has_initial_state = m.has_initial_state
                assert has_initial_state is not None

                pstart = num_decode_tokens
                pend = pstart + num_prefill_tokens
                prefill_mixed_qkv = mixed_qkv[pstart:pend]
                prefill_a = a[pstart:pend]
                prefill_b = b[pstart:pend]
                prefill_state_indices = state_indices_tensor[
                    num_decodes : num_decodes + num_prefills
                ]
                prefill_query_start_loc = (
                    query_start_loc[num_decodes : num_decodes + num_prefills + 1]
                    - num_decode_tokens
                )
                prefill_has_initial_state = has_initial_state[
                    num_decodes : num_decodes + num_prefills
                ]

                prefill_conv_out = gdn_conv1d_prefill_hexagon(
                    x=prefill_mixed_qkv.transpose(0, 1),  # [conv_dim, tokens]
                    weight=conv_weights,
                    bias=self.conv1d.bias,
                    conv_states=conv_state,
                    query_start_loc=prefill_query_start_loc,
                    cache_indices=prefill_state_indices,
                    has_initial_state=prefill_has_initial_state,
                    activation=activation,
                ).transpose(0, 1)

                query, key, value = self.rearrange_mixed_qkv(prefill_conv_out)
                g, beta = gdn_gating_hexagon(
                    prefill_a, prefill_b, self.A_log, self.dt_bias
                )

                attn_out = gdn_recurrent_prefill_hexagon(
                    q=query.squeeze(0),
                    k=key.squeeze(0),
                    v=value.squeeze(0),
                    g=g,
                    beta=beta,
                    ssm_state=ssm_state,
                    query_start_loc=prefill_query_start_loc,
                    state_indices=prefill_state_indices,
                    has_initial_state=prefill_has_initial_state,
                    scale=scale,
                )
                core_attn_out[pstart:pend] = attn_out.to(core_attn_out.dtype)

            if not is_conv_state_dim_first():
                # conv_state is a transposed view of self.kv_cache[0]; in-place
                # index writes above already updated the underlying storage.
                pass

    return QaicQwenGDNAttention


_PATCH_APPLIED = False


def apply_qwen3_5_gdn_patch() -> None:
    """Swap the Qwen3.5 linear-attention layer for the QAIC Hexagon version.

    No-op on the AoT (QEfficient/QPC) path, which never instantiates the vLLM
    PyTorch model.  Idempotent.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    try:
        from vllm_qaic.platform_base import QaicPlatform
    except Exception:  # pragma: no cover - defensive
        QaicPlatform = None  # type: ignore[assignment]

    if QaicPlatform is not None and getattr(QaicPlatform, "is_aot", False):
        # AoT mode compiles via QEfficient; the vLLM eager model is not used.
        return

    from vllm.model_executor.models import qwen3_5

    qaic_cls = _build_qaic_gdn_class()
    qwen3_5.QwenGatedDeltaNetAttention = qaic_cls
    _PATCH_APPLIED = True
    logger.info_once(
        "Patched qwen3_5.QwenGatedDeltaNetAttention -> QaicQwenGDNAttention "
        "(Hexagon GDN kernels for QAIC eager mode)."
    )


apply_qwen3_5_gdn_patch()
