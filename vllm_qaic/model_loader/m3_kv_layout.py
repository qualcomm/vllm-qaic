# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""KV-handoff resharding for MiniMax M3 disaggregated serving.

Ported, oracle-verified reshape/transpose math -- reshards a CP=1 prefill KV
buffer into the CP=`cp` layout a decode QPC expects. `attention_cp1_to_cpn`
covers `past_key.*`/`past_value.*`; `index_cp1_to_cpn` covers the lightning
indexer's `index_key.*` (block-cyclic, granularity `INDEX_BLOCK_SIZE`).
"""

import numpy as np

# The one M3 model constant that can't be read back out of a QPC's own
# binding dims.
INDEX_BLOCK_SIZE = 128


def attention_cp1_to_cpn(
    logical: np.ndarray, cp: int, kv_heads: int, slots: int, head_dim: int
) -> np.ndarray:
    """Pack a CP=1 attention KV tensor into the CP=`cp` layout."""
    expected = (1, kv_heads, slots * cp, head_dim)
    if logical.shape != expected:
        raise ValueError(f"expected CP=1 shape {expected}, got {logical.shape}")
    grouped = logical.reshape(1, kv_heads, slots, cp, head_dim)
    interleaved = grouped.transpose(0, 3, 1, 2, 4)
    return np.ascontiguousarray(interleaved.reshape(1, cp * kv_heads, slots, head_dim))


def index_cp1_to_cpn(
    cache: np.ndarray, cp: int, block_size: int, prefix_tokens: int, head_dim: int
) -> np.ndarray:
    """Pack a CP=1 index KV tensor into the CP=`cp` block-cyclic layout."""
    if cache.shape[:2] != (1, 1) or cache.shape[-1] != head_dim:
        raise ValueError(f"unexpected CP=1 index cache shape: {cache.shape}")
    if prefix_tokens % (cp * block_size):
        raise ValueError(
            f"prefix_tokens {prefix_tokens} must be a multiple of "
            f"cp*block_size {cp * block_size}"
        )
    if cache.shape[2] < prefix_tokens:
        raise ValueError("prefix_tokens exceeds the CP=1 index cache")
    groups = prefix_tokens // (cp * block_size)
    grouped = cache[:, :, :prefix_tokens].reshape(1, groups, cp, block_size, head_dim)
    cp_rows = grouped.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(cp_rows.reshape(1, cp, prefix_tokens // cp, head_dim))
