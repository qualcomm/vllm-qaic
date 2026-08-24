# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for QAIC-specific TopK router helper functions.

Tests the pure Python guard functions that decide whether to dispatch
to the QAIC HVX kernel or fall back to the CPU reference implementation.
No hardware required.

Functions tested
----------------
From vllm_qaic.ops.grouped_topk_router:
  - _has_valid_expert_grouping(num_experts, num_expert_group, topk_group)
  - _supports_qaic_grouped_topk(gating_output, topk, num_expert_group, topk_group)

From vllm_qaic.ops.topk_router:
  - _supports_qaic_regular_topk(qaic_regular_topk, gating_output, topk, e_score_correction_bias)

Coverage areas
--------------
1. _has_valid_expert_grouping — valid and invalid combinations
2. _supports_qaic_grouped_topk — device type, dtype, shape, grouping constraints
3. _supports_qaic_regular_topk — device type, dtype, topk limit, bias constraint
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------
from vllm_qaic.ops.grouped_topk_router import (
    _has_valid_expert_grouping,
    _supports_qaic_grouped_topk,
)
from vllm_qaic.ops.topk_router import _supports_qaic_regular_topk


# ===========================================================================
# 1. _has_valid_expert_grouping()
# ===========================================================================

class TestHasValidExpertGrouping:
    """
    Valid grouping requires:
      - num_expert_group > 0
      - topk_group > 0
      - topk_group <= num_expert_group
      - num_experts > num_expert_group
      - num_experts % num_expert_group == 0
    """

    def test_valid_standard_moe_config(self):
        """Standard MoE: 64 experts, 8 groups, topk_group=2."""
        assert _has_valid_expert_grouping(
            num_experts=64, num_expert_group=8, topk_group=2
        ) is True

    def test_valid_deepseek_config(self):
        """DeepSeek-style: 256 experts, 8 groups, topk_group=3."""
        assert _has_valid_expert_grouping(
            num_experts=256, num_expert_group=8, topk_group=3
        ) is True

    def test_valid_minimal_config(self):
        """Minimal valid: 2 experts, 1 group, topk_group=1."""
        assert _has_valid_expert_grouping(
            num_experts=2, num_expert_group=1, topk_group=1
        ) is True

    def test_invalid_num_expert_group_zero(self):
        """num_expert_group=0 is invalid."""
        assert _has_valid_expert_grouping(
            num_experts=64, num_expert_group=0, topk_group=2
        ) is False

    def test_invalid_topk_group_zero(self):
        """topk_group=0 is invalid."""
        assert _has_valid_expert_grouping(
            num_experts=64, num_expert_group=8, topk_group=0
        ) is False

    def test_invalid_topk_group_exceeds_num_expert_group(self):
        """topk_group > num_expert_group is invalid."""
        assert _has_valid_expert_grouping(
            num_experts=64, num_expert_group=4, topk_group=8
        ) is False

    def test_invalid_num_experts_equals_num_expert_group(self):
        """num_experts == num_expert_group means no grouping (regular topk path)."""
        assert _has_valid_expert_grouping(
            num_experts=8, num_expert_group=8, topk_group=2
        ) is False

    def test_invalid_num_experts_less_than_num_expert_group(self):
        """num_experts < num_expert_group is invalid."""
        assert _has_valid_expert_grouping(
            num_experts=4, num_expert_group=8, topk_group=2
        ) is False

    def test_invalid_num_experts_not_divisible(self):
        """num_experts % num_expert_group != 0 is invalid."""
        assert _has_valid_expert_grouping(
            num_experts=65, num_expert_group=8, topk_group=2
        ) is False

    def test_topk_group_equals_num_expert_group(self):
        """topk_group == num_expert_group is valid (select all groups)."""
        assert _has_valid_expert_grouping(
            num_experts=64, num_expert_group=8, topk_group=8
        ) is True

    @pytest.mark.parametrize("num_experts,num_groups,topk_group,expected", [
        (64, 8, 2, True),
        (128, 8, 4, True),
        (256, 16, 4, True),
        (8, 8, 2, False),   # num_experts == num_expert_group
        (64, 0, 2, False),  # num_expert_group=0
        (64, 8, 0, False),  # topk_group=0
        (64, 9, 2, False),  # not divisible
    ])
    def test_parametrized_cases(self, num_experts, num_groups, topk_group, expected):
        result = _has_valid_expert_grouping(num_experts, num_groups, topk_group)
        assert result == expected


# ===========================================================================
# 2. _supports_qaic_grouped_topk()
# ===========================================================================

class TestSupportsQaicGroupedTopk:
    """
    QAIC grouped topk is supported when:
      - gating_output.device.type == "qaic"
      - gating_output.dtype == torch.float16
      - gating_output.dim() == 2
      - num_expert_group <= 128
      - topk_group <= 128
      - topk <= 64
      - gating_output.shape[-1] <= 1024
      - _has_valid_expert_grouping() is True
    """

    def _make_gating(self, device="cpu", dtype=torch.float16, shape=(4, 64)):
        """Create a mock gating_output tensor."""
        return torch.zeros(shape, dtype=dtype, device=device)

    def test_cpu_device_not_supported(self):
        """CPU device must not trigger QAIC kernel."""
        gating = self._make_gating(device="cpu")
        assert _supports_qaic_grouped_topk(gating, topk=2, num_expert_group=8, topk_group=2) is False

    def test_float32_not_supported(self):
        """float32 dtype must not trigger QAIC kernel (fp16 only)."""
        gating = self._make_gating(dtype=torch.float32)
        assert _supports_qaic_grouped_topk(gating, topk=2, num_expert_group=8, topk_group=2) is False

    def test_1d_tensor_not_supported(self):
        """1D tensor must not trigger QAIC kernel (requires 2D)."""
        gating = torch.zeros(64, dtype=torch.float16)
        assert _supports_qaic_grouped_topk(gating, topk=2, num_expert_group=8, topk_group=2) is False

    def test_topk_exceeds_64_not_supported(self):
        """topk > 64 must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 64))
        assert _supports_qaic_grouped_topk(gating, topk=65, num_expert_group=8, topk_group=2) is False

    def test_num_expert_group_exceeds_128_not_supported(self):
        """num_expert_group > 128 must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 256))
        assert _supports_qaic_grouped_topk(gating, topk=2, num_expert_group=129, topk_group=2) is False

    def test_shape_exceeds_1024_not_supported(self):
        """gating_output.shape[-1] > 1024 must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 1025))
        assert _supports_qaic_grouped_topk(gating, topk=2, num_expert_group=8, topk_group=2) is False

    def test_invalid_grouping_not_supported(self):
        """Invalid expert grouping must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 64))
        # num_experts == num_expert_group → invalid grouping
        assert _supports_qaic_grouped_topk(gating, topk=2, num_expert_group=64, topk_group=2) is False


# ===========================================================================
# 3. _supports_qaic_regular_topk()
# ===========================================================================

class TestSupportsQaicRegularTopk:
    """
    QAIC regular topk is supported when:
      - qaic_regular_topk is not None (kernel available)
      - e_score_correction_bias is None
      - gating_output.device.type == "qaic"
      - gating_output.dtype == torch.float16
      - gating_output.dim() == 2
      - topk <= 64
      - gating_output.shape[-1] <= 1024
    """

    def _make_gating(self, device="cpu", dtype=torch.float16, shape=(4, 64)):
        return torch.zeros(shape, dtype=dtype, device=device)

    def _mock_kernel(self):
        """Return a non-None mock kernel object."""
        return object()

    def test_none_kernel_not_supported(self):
        """None kernel means QAIC custom op not available."""
        gating = self._make_gating()
        assert _supports_qaic_regular_topk(None, gating, topk=2, e_score_correction_bias=None) is False

    def test_cpu_device_not_supported(self):
        """CPU device must not trigger QAIC kernel."""
        gating = self._make_gating(device="cpu")
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=2, e_score_correction_bias=None) is False

    def test_float32_not_supported(self):
        """float32 dtype must not trigger QAIC kernel."""
        gating = self._make_gating(dtype=torch.float32)
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=2, e_score_correction_bias=None) is False

    def test_1d_tensor_not_supported(self):
        """1D tensor must not trigger QAIC kernel."""
        gating = torch.zeros(64, dtype=torch.float16)
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=2, e_score_correction_bias=None) is False

    def test_topk_exceeds_64_not_supported(self):
        """topk > 64 must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 64))
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=65, e_score_correction_bias=None) is False

    def test_shape_exceeds_1024_not_supported(self):
        """gating_output.shape[-1] > 1024 must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 1025))
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=2, e_score_correction_bias=None) is False

    def test_e_score_correction_bias_not_supported(self):
        """Non-None e_score_correction_bias must not trigger QAIC kernel."""
        gating = self._make_gating(shape=(4, 64))
        bias = torch.zeros(64, dtype=torch.float16)
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=2, e_score_correction_bias=bias) is False

    def test_topk_at_limit_cpu_not_supported(self):
        """topk=64 (at limit) but CPU device → not supported."""
        gating = self._make_gating(device="cpu", shape=(4, 64))
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=64, e_score_correction_bias=None) is False

    def test_shape_at_limit_cpu_not_supported(self):
        """shape[-1]=1024 (at limit) but CPU device → not supported."""
        gating = self._make_gating(device="cpu", shape=(4, 1024))
        assert _supports_qaic_regular_topk(self._mock_kernel(), gating, topk=2, e_score_correction_bias=None) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
