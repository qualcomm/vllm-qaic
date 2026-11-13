# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""MiniMax M3 KV-handoff resharding math: attention_cp1_to_cpn/index_cp1_to_cpn.

The reshape/transpose logic in these functions was already validated
bit-identical against hand-written reference implementations in a sibling
repo before being ported here unmodified. These tests hand-compute small,
fully-worked examples to re-pin that same behavior locally, with no external
file dependencies.
"""

import numpy as np
import pytest

from vllm_qaic.model_loader.m3_kv_layout import (
    INDEX_BLOCK_SIZE,
    attention_cp1_to_cpn,
    index_cp1_to_cpn,
)


def test_index_block_size_constant():
    assert INDEX_BLOCK_SIZE == 128


def test_attention_cp1_to_cpn_hand_computed():
    # kv_heads=1, cp=2, slots=2 -> 4 raw tokens, head_dim=1.
    # Token t lands in way (t % cp): way0 gets tokens [0, 2], way1 gets [1, 3].
    logical = np.arange(4, dtype=np.int32).reshape(1, 1, 4, 1)
    out = attention_cp1_to_cpn(logical, cp=2, kv_heads=1, slots=2, head_dim=1)
    expected = np.array([[[[0], [2]], [[1], [3]]]], dtype=np.int32)
    assert out.shape == (1, 2, 2, 1)
    assert np.array_equal(out, expected)


def test_attention_cp1_to_cpn_multi_head():
    # kv_heads=2, cp=2, slots=2 -> row axis becomes cp*kv_heads=4, ordered
    # way-major (way0's heads, then way1's heads), matching the transpose
    # order (0, 3, 1, 2, 4) -> reshape(1, cp*kv_heads, slots, head_dim).
    logical = np.arange(2 * 4, dtype=np.int32).reshape(1, 2, 4, 1)
    out = attention_cp1_to_cpn(logical, cp=2, kv_heads=2, slots=2, head_dim=1)
    assert out.shape == (1, 4, 2, 1)
    # head0 way0, head1 way0, head0 way1, head1 way1
    assert np.array_equal(out[0, 0, :, 0], logical[0, 0, [0, 2], 0])
    assert np.array_equal(out[0, 1, :, 0], logical[0, 1, [0, 2], 0])
    assert np.array_equal(out[0, 2, :, 0], logical[0, 0, [1, 3], 0])
    assert np.array_equal(out[0, 3, :, 0], logical[0, 1, [1, 3], 0])


def test_attention_cp1_to_cpn_rejects_wrong_shape():
    bad = np.zeros((1, 1, 3, 1), dtype=np.int32)  # not slots*cp
    with pytest.raises(ValueError):
        attention_cp1_to_cpn(bad, cp=2, kv_heads=1, slots=2, head_dim=1)


def test_index_cp1_to_cpn_hand_computed():
    # cp=2, block_size=2, prefix_tokens=8 -> block-cyclic: block i (2 tokens)
    # goes to way (i % cp). way0 = tokens[0,1,4,5], way1 = tokens[2,3,6,7].
    cache = np.arange(8, dtype=np.int32).reshape(1, 1, 8, 1)
    out = index_cp1_to_cpn(cache, cp=2, block_size=2, prefix_tokens=8, head_dim=1)
    assert out.shape == (1, 2, 4, 1)
    assert np.array_equal(out[0, 0, :, 0], [0, 1, 4, 5])
    assert np.array_equal(out[0, 1, :, 0], [2, 3, 6, 7])


def test_index_cp1_to_cpn_rejects_non_divisible_prefix_tokens():
    cache = np.zeros((1, 1, 6, 1), dtype=np.int32)
    with pytest.raises(ValueError):
        # prefix_tokens=6 is not a multiple of cp*block_size=8
        index_cp1_to_cpn(cache, cp=2, block_size=4, prefix_tokens=6, head_dim=1)


def test_index_cp1_to_cpn_rejects_wrong_shape():
    bad = np.zeros((1, 2, 8, 1), dtype=np.int32)  # cache.shape[:2] must be (1, 1)
    with pytest.raises(ValueError):
        index_cp1_to_cpn(bad, cp=2, block_size=2, prefix_tokens=8, head_dim=1)
