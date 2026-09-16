# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Custom rotary embedding implementations for QAIC platform."""

import torch
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)


class QAicApplyRotaryEmb(ApplyRotaryEmb):
    """
    QAIC-specific ApplyRotaryEmbedding implementation.

    This is an out-of-tree (OOT) custom operator that replaces vLLM's default
    ``ApplyRotaryEmb`` implementation for QAIC devices. It delegates to the QAIC
    custom operator ``torch.ops.qaic.rotary_embedding``, which is dispatched to the
    native NSP kernel on QAIC hardware via the QAIC dispatcher.  For unsupported
    dtypes or non-contiguous inputs the custom op falls back to a CPU-side
    rotary_embedding decomposition.

    the staticmethod ensures that any calls to the static method of ApplyRotaryEmb
    are patched to QAicApplyRotaryEmb

    the forward_oot ensures that the regular forward calls on ApplyRotaryEmb
    instances are dispatched to the OOT implementation

    The op rotates each pair of channels of ``x`` by the angle encoded in
    ``cos``/``sin``. With ``d = x.shape[-1] // 2``, Neox style pairs channel ``i``
    with ``i + d``, while GPT-J style pairs adjacent channels ``2i`` and ``2i + 1``:

        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin

    Shapes:
        x: (num_tokens, num_heads, head_size) or
           (batch_size, seq_len, num_heads, head_size)
        cos, sin: (num_tokens, head_size // 2)
        return: same shape as x
    """

    @staticmethod
    def forward_static(
        x, cos, sin, is_neox_style=True, enable_fp32_compute=False
    ) -> torch.Tensor:
        """
        QAIC-specific forward implementation.

        Calls the QAIC custom  operator which dispatches to the
        NSP tiling kernel on QAIC hardware.

        Args:
            x: [batch_size (optional), seq_len, num_heads, head_size]
            cos: [seq_len, head_size // 2]
            sin: [seq_len, head_size // 2]
            is_neox_style: Whether to use the Neox-style or GPT-J-style.
            enable_fp32_compute: Temporarily convert x, cos, sin to FP32 dtype
                                 for higher accuracy.

        Returns:
            Output tensor of shape same as input x
        """
        if enable_fp32_compute:
            x = x.to(torch.float32)
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)
        out = torch.ops.qaic.rotary_embedding(x, cos, sin, is_neox_style)

        return out

    def forward_oot(self, x, cos, sin, is_neox_style=True, enable_fp32_compute=False):
        return self.forward_static(x, cos, sin, is_neox_style, enable_fp32_compute)


class QAicRotaryEmbedding(RotaryEmbedding):
    """
    QAIC-specific RotaryEmbedding implementation.

    This is an out-of-tree (OOT) custom operator that replaces vLLM's default
    ``RotaryEmbedding`` for QAIC devices. It keeps vLLM's cos/sin cache lookup and
    the rotary/pass-through split in PyTorch, and delegates only the rotation
    itself to :class:`QAicApplyRotaryEmb`, which calls the QAIC custom operator
    ``torch.ops.qaic.rotary_embedding``.

    Only the leading ``rotary_dim`` channels of each head are rotated; the
    remaining ``head_size - rotary_dim`` channels are passed through unchanged and
    concatenated back, so the query/key shapes seen by the caller are preserved.

    ``forward_static`` mirrors the base class staticmethod so that any code calling
    ``RotaryEmbedding.forward_static`` on a patched instance reaches the QAIC path,
    and ``forward_oot`` routes regular ``forward`` calls on the instance to that
    same implementation.
    """

    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None,
        head_size: int,
        rotary_dim: int,
        cos_sin_cache: torch.Tensor,
        is_neox_style: bool,
    ):
        """
        QAIC-specific forward implementation.

        Gathers the per-position cos/sin values from ``cos_sin_cache``, then applies
        the rotation to the first ``rotary_dim`` channels of query (and key, when
        provided) through :meth:`QAicApplyRotaryEmb.forward_static`, which dispatches
        to the NSP kernel on QAIC hardware.

        Args:
            positions: [num_tokens] or any shape that flattens to it; token
                       positions used to index ``cos_sin_cache``.
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size], or None when the layer has
                 no key to rotate (e.g. cross-layer KV sharing).
            head_size: Size of a single attention head.
            rotary_dim: Number of leading channels per head that are rotated.
            cos_sin_cache: [max_positions, rotary_dim] cache holding the
                           concatenated cos and sin tables.
            is_neox_style: Whether to use the Neox-style or GPT-J-style.

        Returns:
            Tuple of (query, key) with the same shapes as the inputs. ``key`` is
            None if None was passed in.
        """
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, head_size)
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        query_rot = QAicApplyRotaryEmb.forward_static(
            query_rot,
            cos,
            sin,
            is_neox_style,
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, head_size)
            key_rot = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]
            key_rot = QAicApplyRotaryEmb.forward_static(
                key_rot,
                cos,
                sin,
                is_neox_style,
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        OOT entry point used for regular ``forward`` calls on this instance.

        Casts the cos/sin cache to the query dtype and forwards to
        :meth:`forward_static` with the layer's own configuration.

        Args:
            positions: [num_tokens] or any shape that flattens to it.
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size], or None.

        Returns:
            Tuple of (query, key) with the same shapes as the inputs.
        """

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        return self.forward_static(
            positions,
            query,
            key,
            self.head_size,
            self.rotary_dim,
            cos_sin_cache,
            self.is_neox_style,
        )
