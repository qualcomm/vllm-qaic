# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Custom MRoPE implementation for QAIC platform with torch.compile support."""

import torch

from vllm.model_executor.layers.rotary_embedding.mrope import (
    MRotaryEmbedding,
    apply_interleaved_rope,
)

class QAicMRotaryEmbedding(MRotaryEmbedding):
    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """PyTorch-native MRoPE forward pass.

        Args:
            positions:
                [num_tokens,] for text-only inputs, or
                [3, num_tokens] for multimodal inputs (T/H/W positions).
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim in (1, 2)
        assert key is not None

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)

        # ------------------------------------------------------------------
        # Assemble cos/sin in PyTorch, but route the rotation itself through
        # ApplyRotaryEmb.forward_oot so it still reaches
        # torch.ops.qaic.rotary_embedding instead of falling all the way back
        # to the PyTorch decomposition.
        # ------------------------------------------------------------------
        if not self.is_neox_style or (positions.ndim == 2 and self.mrope_interleaved):
            num_tokens = positions.shape[-1]
            # [num_tokens, rotary_dim] for 1D positions,
            # [3, num_tokens, rotary_dim] for 3D (T/H/W) positions.
            cos_sin = cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            if positions.ndim == 2:
                # Collapse the leading T/H/W axis: each frequency band takes
                # its values from one of the 3 rows, giving [num_tokens, half].
                assert self.mrope_section
                if self.mrope_interleaved:
                    cos = apply_interleaved_rope(cos, self.mrope_section)
                    sin = apply_interleaved_rope(sin, self.mrope_section)
                else:
                    cos = torch.cat(
                        [
                            m[i]
                            for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
                        ],
                        dim=-1,
                    )
                    sin = torch.cat(
                        [
                            m[i]
                            for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
                        ],
                        dim=-1,
                    )

            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size)
            query_rot = query[..., : self.rotary_dim]
            query_pass = query[..., self.rotary_dim :]
            query_rot = self.apply_rotary_emb.forward_oot(
                query_rot, cos, sin, self.is_neox_style
            )
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = self.apply_rotary_emb.forward_oot(
                key_rot, cos, sin, self.is_neox_style
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

            return query, key
        # ------------------------------------------------------------------
        # Non-interleaved path: use the fused mrotary_embedding op.
        #
        # positions.ndim == 2 (multimodal, T/H/W axes, num_axes=3):
        #   The kernel assembles cos/sin from mrope_section slices internally.
        #
        # positions.ndim == 1 (text-only, num_axes=1):
        #   Triggers eager fallback (dispatcher returns VALIDATION_FAILED).
        #   mrope_section is unused (passed as [0, 0, 0]).
        # ------------------------------------------------------------------
        if positions.ndim == 2:
            assert self.mrope_section
            mrope_section_val = (list(self.mrope_section) + [0, 0, 0])[:3]
            num_axes = 3
        else:
            # 1D positions (text-only): mrope_section unused.
            # Note: num_axes=1 always triggers QAIC_ERROR_VALIDATION_FAILED in
            # the dispatcher (the VTCM+DMA kernel only supports 3D positions),
            # so this path falls back to eager computation in RegisterQAic.cpp.
            mrope_section_val = [0, 0, 0]
            num_axes = 1

        query_out = torch.empty_like(query)
        key_out = torch.empty_like(key)
        torch.ops.qaic.mrotary_embedding(
            query,
            key,
            positions,
            cos_sin_cache,
            num_axes,
            mrope_section_val[0],
            mrope_section_val[1],
            mrope_section_val[2],
            self.head_size,
            self.rotary_dim,
            self.is_neox_style,
            query_out,
            key_out,
        )

        return query_out, key_out
