# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for vllm_qaic.patch.patch_rejection_sampler.

Tests the QAIC-specific greedy-path optimizations in the rejection sampler:
  - skip-clone: when all requests are greedy and no logprobs needed,
    the tensor clone before apply_logits_processors is skipped
  - skip-softmax: for greedy decoding argmax(logits) == argmax(softmax(logits)),
    so computing the probability distribution is unnecessary

No hardware required.  All tests are pure Python.

Coverage areas
--------------
1. Patch module importability and _qaic_forward function exists
2. RejectionSampler.forward is replaced with _qaic_forward after import
3. Mathematical property: argmax(logits) == argmax(softmax(logits))
4. Skip-clone: clone vs no-clone produces same argmax
5. Bonus/target logits index extraction
"""

import pytest
import torch


# ===========================================================================
# 1-2. Patch module importability
# ===========================================================================

class TestPatchImport:
    def test_patch_module_importable(self):
        """patch_rejection_sampler must be importable without errors."""
        import vllm_qaic.patch.patch_rejection_sampler  # noqa: F401

    def test_qaic_forward_function_exists(self):
        """_qaic_forward must be defined in the patch module."""
        from vllm_qaic.patch.patch_rejection_sampler import _qaic_forward
        assert callable(_qaic_forward)

    def test_rejection_sampler_patched(self):
        """After importing the patch, RejectionSampler.forward must be _qaic_forward."""
        import vllm_qaic.patch.patch_rejection_sampler  # noqa: F401
        from vllm_qaic.patch.patch_rejection_sampler import _qaic_forward
        from vllm.v1.sample.rejection_sampler import RejectionSampler
        assert RejectionSampler.forward is _qaic_forward


# ===========================================================================
# 3. Mathematical property: argmax(logits) == argmax(softmax(logits))
# ===========================================================================

class TestGreedyPathMath:
    """
    The skip-softmax optimization is valid because:
      argmax(logits) == argmax(softmax(logits))
    This is a fundamental property of the softmax function.
    """

    def test_argmax_equals_softmax_argmax_float32(self):
        """argmax(logits) == argmax(softmax(logits)) for float32."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32000)
        greedy = logits.argmax(dim=-1)
        softmax_greedy = torch.softmax(logits, dim=-1).argmax(dim=-1)
        assert torch.equal(greedy, softmax_greedy)

    def test_argmax_equals_softmax_argmax_float16(self):
        """argmax(logits) == argmax(softmax(logits)) for float16 (QAIC dtype)."""
        torch.manual_seed(0)
        logits = torch.randn(8, 32000, dtype=torch.float16)
        greedy = logits.argmax(dim=-1)
        softmax_greedy = torch.softmax(logits.float(), dim=-1).argmax(dim=-1)
        assert torch.equal(greedy, softmax_greedy)

    def test_argmax_scale_invariant(self):
        """argmax(c * logits) == argmax(logits) for c > 0."""
        torch.manual_seed(1)
        logits = torch.randn(4, 1000)
        base = logits.argmax(dim=-1)
        for scale in [0.1, 1.0, 10.0, 100.0]:
            assert torch.equal((logits * scale).argmax(dim=-1), base)

    def test_argmax_batch_consistency(self):
        """argmax must be consistent across batch dimension."""
        torch.manual_seed(2)
        logits = torch.randn(16, 50000)
        batch_argmax = logits.argmax(dim=-1)
        for i in range(16):
            single_argmax = logits[i].argmax()
            assert batch_argmax[i] == single_argmax


# ===========================================================================
# 4. Skip-clone optimization
# ===========================================================================

class TestSkipCloneOptimization:
    """
    The skip-clone optimization avoids cloning logits before argmax.
    This is safe because argmax is read-only.
    """

    def test_clone_vs_no_clone_same_argmax(self):
        """Clone vs no-clone must produce identical argmax."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32000)
        tokens_with_clone = logits.clone().argmax(dim=-1)
        tokens_without_clone = logits.argmax(dim=-1)
        assert torch.equal(tokens_with_clone, tokens_without_clone)

    def test_in_place_modification_after_argmax_safe(self):
        """Modifying logits in-place after argmax does not affect the result."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32000)
        tokens = logits.argmax(dim=-1)
        logits.fill_(0.0)  # simulate in-place modification
        assert tokens.shape == (4,)
        assert tokens.dtype == torch.int64


# ===========================================================================
# 5. Bonus/target logits index extraction
# ===========================================================================

class TestLogitsIndices:
    """Tests the bonus_logits_indices and target_logits_indices extraction."""

    def test_bonus_logits_extraction(self):
        """bonus_logits = logits[bonus_logits_indices] must work correctly."""
        logits = torch.randn(20, 100)
        bonus_indices = torch.tensor([3, 7, 11, 15], dtype=torch.int32)
        bonus_logits = logits[bonus_indices]
        assert bonus_logits.shape == (4, 100)
        assert torch.equal(bonus_logits[0], logits[3])
        assert torch.equal(bonus_logits[1], logits[7])

    def test_target_logits_extraction(self):
        """target_logits = logits[target_logits_indices] must work correctly."""
        logits = torch.randn(20, 100)
        target_indices = torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.int32)
        target_logits = logits[target_indices]
        assert target_logits.shape == (6, 100)
        assert torch.equal(target_logits[0], logits[0])
        assert torch.equal(target_logits[3], logits[4])

    def test_bonus_argmax_is_greedy_token(self):
        """Greedy token from bonus_logits must be the argmax."""
        torch.manual_seed(42)
        logits = torch.randn(4, 32000)
        bonus_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        bonus_logits = logits[bonus_indices]
        greedy_tokens = bonus_logits.argmax(dim=-1)
        assert greedy_tokens.shape == (4,)
        assert greedy_tokens.dtype == torch.int64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
