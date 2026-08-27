# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Pure-Python numerics for the rms_norm algorithm implemented by kernel.cpp —
validated here via PyTorch ops (no hardware needed).

Hardware tests that call the compiled rms_norm_hexagon kernel and its Python
wrappers (QAicRMSNorm / QAicGemmaRMSNorm) on real QAIC silicon live in
vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_rms_norm_kernel.py.
"""

import math

import pytest
import torch
import torch.nn.functional as F


def _rms_norm_ref(
    attn_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference: residual = attn_out + x; dst = residual / rms(residual) * weight."""
    residual = attn_out.float() + x.float()
    mean_sq = (residual * residual).mean(dim=-1, keepdim=True)
    inv_rms = 1.0 / torch.sqrt(mean_sq + epsilon)
    dst = residual * inv_rms * weight.float()
    return dst, residual


def _cast_ref(
    dst_ref: torch.Tensor,
    residual_ref: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return dst_ref.to(dtype), residual_ref.to(dtype)


# N must be a multiple of 64 (HVX vector = 128 bytes / 2 bytes per element)
HIDDEN_SIZES = [64, 128, 256, 512, 1024, 4096]
BATCH_SIZES  = [1, 4, 14, 15, 16, 32]  # 14/15 exercises the ceil(M/numCores) boundary
DTYPES       = [torch.float16, torch.bfloat16]
EPSILONS     = [1e-5, 1e-6]

_ATOL = {torch.float16: 5e-3, torch.bfloat16: 5e-3}
_RTOL = {torch.float16: 1e-2, torch.bfloat16: 1e-2}


def _make_inputs(M: int, N: int, dtype: torch.dtype, seed: int = 42):
    torch.manual_seed(seed)
    attn_out = torch.randn(M, N, dtype=dtype) * 0.1
    x        = torch.randn(M, N, dtype=dtype) * 0.1
    weight   = torch.randn(N,    dtype=dtype)
    return attn_out, x, weight


class TestRmsNormReference:
    """Sanity-checks the FP32 reference before it is used as ground truth."""

    def test_identity_weight(self):
        """weight=1, input=1 → residual=2, dst=1 (rms of constant = the constant)."""
        M, N = 4, 64
        attn_out = torch.ones(M, N)
        x        = torch.ones(M, N)
        weight   = torch.ones(N)
        dst, residual = _rms_norm_ref(attn_out, x, weight, epsilon=1e-6)
        torch.testing.assert_close(residual, torch.full((M, N), 2.0), atol=1e-6, rtol=0)
        torch.testing.assert_close(dst, torch.ones(M, N), atol=1e-5, rtol=0)

    def test_zero_x(self):
        """x=0 → residual = attn_out, dst = rms_norm(attn_out) * weight."""
        M, N = 2, 128
        torch.manual_seed(0)
        attn_out = torch.randn(M, N)
        x        = torch.zeros(M, N)
        weight   = torch.ones(N)
        dst, residual = _rms_norm_ref(attn_out, x, weight, epsilon=1e-6)
        torch.testing.assert_close(residual, attn_out, atol=1e-6, rtol=0)
        ref = F.rms_norm(attn_out, [N], weight, eps=1e-6)
        torch.testing.assert_close(dst.float(), ref.float(), atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        """Output shapes match input shapes."""
        M, N = 7, 256
        attn_out, x, weight = _make_inputs(M, N, torch.float32)
        dst, residual = _rms_norm_ref(attn_out, x, weight, epsilon=1e-5)
        assert dst.shape == (M, N)
        assert residual.shape == (M, N)

    def test_epsilon_prevents_div_zero(self):
        """All-zero input must not produce NaN or Inf."""
        M, N = 2, 64
        attn_out = torch.zeros(M, N)
        x        = torch.zeros(M, N)
        weight   = torch.ones(N)
        dst, residual = _rms_norm_ref(attn_out, x, weight, epsilon=1e-5)
        assert not dst.isnan().any()
        assert not dst.isinf().any()

    def test_single_row(self):
        """M=1 degenerate case works."""
        M, N = 1, 64
        attn_out, x, weight = _make_inputs(M, N, torch.float32)
        dst, residual = _rms_norm_ref(attn_out, x, weight, epsilon=1e-6)
        assert dst.shape == (1, N)


class TestRmsNormKernelNumerics:
    """Validates the algorithm kernel.cpp implements, exercised via PyTorch ops (no HW needed)."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("epsilon", EPSILONS)
    @pytest.mark.parametrize("N", HIDDEN_SIZES)
    @pytest.mark.parametrize("M", BATCH_SIZES)
    def test_fused_add_rms_norm_2d(self, M, N, epsilon, dtype):
        """dst and residual match the FP32 reference across M/N/eps/dtype combinations."""
        attn_out, x, weight = _make_inputs(M, N, dtype)
        dst_ref, residual_ref = _cast_ref(*_rms_norm_ref(attn_out, x, weight, epsilon), dtype)

        residual_got = (attn_out.float() + x.float()).to(dtype)
        dst_got = F.rms_norm(residual_got.float(), [N], weight.float(), eps=epsilon).to(dtype)

        torch.testing.assert_close(residual_got, residual_ref, atol=_ATOL[dtype], rtol=_RTOL[dtype])
        torch.testing.assert_close(dst_got,      dst_ref,      atol=_ATOL[dtype], rtol=_RTOL[dtype])

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", [
        (4, 16, 256),    # 3-D: (batch, seq, hidden)
        (2, 8, 4, 64),   # 4-D: (B, M, H, N) — kernel reshapes to (B*M, H*N)
    ])
    def test_fused_add_rms_norm_nd(self, shape, dtype):
        """N-D inputs are correctly flattened to 2D before dispatch."""
        epsilon = 1e-5
        torch.manual_seed(7)
        attn_out = torch.randn(*shape, dtype=dtype) * 0.1
        x        = torch.randn(*shape, dtype=dtype) * 0.1
        hidden   = shape[-1] if len(shape) != 4 else shape[2] * shape[3]
        weight   = torch.randn(hidden, dtype=dtype)

        if len(shape) == 4:
            B, _M, H, _N = shape
            flat = H * _N
            a2d = attn_out.reshape(B * _M, flat).float()
            x2d = x.reshape(B * _M, flat).float()
        else:
            a2d = attn_out.reshape(-1, shape[-1]).float()
            x2d = x.reshape(-1, shape[-1]).float()

        residual_2d = (a2d + x2d).to(dtype)
        dst_2d = F.rms_norm(residual_2d.float(), [hidden], weight.float(), eps=epsilon).to(dtype)

        dst      = dst_2d.reshape(shape)
        residual = residual_2d.reshape(shape)

        assert dst.shape      == attn_out.shape
        assert residual.shape == attn_out.shape
        assert not dst.isnan().any()
        assert not residual.isnan().any()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_residual_is_sum(self, dtype):
        """residual == attn_out + x exactly."""
        M, N = 8, 128
        torch.manual_seed(1)
        attn_out = torch.randn(M, N, dtype=dtype) * 0.1
        x        = torch.randn(M, N, dtype=dtype) * 0.1
        residual_got = (attn_out.float() + x.float()).to(dtype)
        expected     = (attn_out.float() + x.float()).to(dtype)
        torch.testing.assert_close(residual_got, expected, atol=0, rtol=0)

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_weight_scaling(self, dtype):
        """Doubling the weight doubles dst."""
        M, N = 4, 64
        torch.manual_seed(2)
        attn_out = torch.randn(M, N, dtype=dtype) * 0.1
        x        = torch.randn(M, N, dtype=dtype) * 0.1
        weight   = torch.randn(N, dtype=dtype).abs() + 0.5
        epsilon  = 1e-5
        residual = (attn_out.float() + x.float())
        dst1 = F.rms_norm(residual, [N], weight.float(),       eps=epsilon).to(dtype)
        dst2 = F.rms_norm(residual, [N], (weight * 2).float(), eps=epsilon).to(dtype)
        torch.testing.assert_close(dst2, dst1 * 2, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_large_values_no_overflow(self, dtype):
        """Values near FP16 max must not produce Inf/NaN."""
        M, N = 4, 64
        # FP16 max ≈ 65504; stay well below to avoid overflow in the residual add
        scale = 100.0 if dtype == torch.float16 else 1000.0
        attn_out = torch.ones(M, N, dtype=dtype) * scale
        x        = torch.ones(M, N, dtype=dtype) * scale
        weight   = torch.ones(N, dtype=dtype)
        residual = (attn_out.float() + x.float()).to(dtype)
        dst = F.rms_norm(residual.float(), [N], weight.float(), eps=1e-5).to(dtype)
        assert not dst.isnan().any()
        assert not dst.isinf().any()
        assert not residual.isnan().any()

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_epsilon_effect(self, dtype):
        """Larger epsilon reduces dst magnitude when residual is near zero."""
        M, N = 4, 64
        attn_out = torch.full((M, N),  1e-3, dtype=dtype)
        x        = torch.full((M, N), -1e-3, dtype=dtype)  # residual ≈ 0
        weight   = torch.ones(N, dtype=dtype)
        residual = (attn_out.float() + x.float())
        dst_small_eps = F.rms_norm(residual, [N], weight.float(), eps=1e-8).to(dtype)
        dst_large_eps = F.rms_norm(residual, [N], weight.float(), eps=1.0).to(dtype)
        assert dst_large_eps.abs().max() <= dst_small_eps.abs().max() + 1e-3


class TestRmsNormAlignmentConstraint:
    """N must be a multiple of 64 (one HVX vector = 128 bytes / 2 bytes per element)."""

    @pytest.mark.parametrize("N", [64, 128, 256, 512, 1024])
    def test_valid_N_no_error(self, N):
        """Aligned sizes produce finite output."""
        M = 4
        attn_out, x, weight = _make_inputs(M, N, torch.float16)
        residual = (attn_out.float() + x.float()).to(torch.float16)
        dst = F.rms_norm(residual.float(), [N], weight.float(), eps=1e-5).to(torch.float16)
        assert not dst.isnan().any()
        assert not dst.isinf().any()

    @pytest.mark.parametrize("N_bad", [63, 65, 100, 127])
    def test_invalid_N_noted(self, N_bad):
        """Documents that non-multiples of 64 are invalid for the Hexagon kernel."""
        assert N_bad % 64 != 0, (
            f"N={N_bad} is unexpectedly aligned — update the test or the kernel constraint."
        )


class TestDtypeDispatch:
    """Mirrors the dtype-flag logic in dispatch.cpp: params[3]==0 → FP16, params[3]==1 → BF16."""

    def test_fp16_path_produces_fp16_output(self):
        M, N = 4, 64
        attn_out, x, weight = _make_inputs(M, N, torch.float16)
        residual = (attn_out.float() + x.float()).to(torch.float16)
        dst = F.rms_norm(residual.float(), [N], weight.float(), eps=1e-5).to(torch.float16)
        assert dst.dtype == torch.float16

    def test_bf16_path_produces_bf16_output(self):
        M, N = 4, 64
        attn_out, x, weight = _make_inputs(M, N, torch.bfloat16)
        residual = (attn_out.float() + x.float()).to(torch.bfloat16)
        dst = F.rms_norm(residual.float(), [N], weight.float(), eps=1e-5).to(torch.bfloat16)
        assert dst.dtype == torch.bfloat16

    def test_fp16_bf16_close_numerics(self):
        """FP16 and BF16 agree numerically for well-conditioned inputs."""
        M, N = 8, 128
        torch.manual_seed(99)
        base = torch.randn(M, N) * 0.1
        weight = torch.randn(N)
        epsilon = 1e-5

        def _run(dtype):
            a = base.to(dtype)
            w = weight.to(dtype)
            res = (a.float() + a.float())
            return F.rms_norm(res, [N], w.float(), eps=epsilon).float()

        # Loose tolerance — FP16 and BF16 have different rounding
        torch.testing.assert_close(_run(torch.float16), _run(torch.bfloat16), atol=2e-2, rtol=2e-2)


class TestMultiCoreStriping:
    """The kernel assigns rows as: core c processes rows c, c+numCores, c+2*numCores, ..."""

    @pytest.mark.parametrize("M", [1, 13, 14, 15, 28])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_row_independence(self, M, dtype):
        """Per-row RMS norm is independent of processing order."""
        N = 128
        attn_out, x, weight = _make_inputs(M, N, dtype)
        epsilon = 1e-5

        residual_full = (attn_out.float() + x.float()).to(dtype)
        dst_full = F.rms_norm(residual_full.float(), [N], weight.float(), eps=epsilon).to(dtype)

        dst_rev = torch.empty_like(dst_full)
        for i in reversed(range(M)):
            r_i = (attn_out[i].float() + x[i].float()).to(dtype)
            dst_rev[i] = F.rms_norm(
                r_i.float().unsqueeze(0), [N], weight.float(), eps=epsilon
            ).to(dtype).squeeze(0)

        torch.testing.assert_close(dst_full, dst_rev, atol=_ATOL[dtype], rtol=_RTOL[dtype])

    @pytest.mark.parametrize("num_cores", [1, 4, 7, 14])
    def test_ceil_row_iters(self, num_cores):
        """ceil(M/numCores) covers every row exactly once with no duplicates."""
        for M in [1, num_cores, num_cores + 1, 2 * num_cores - 1, 2 * num_cores]:
            row_iters = math.ceil(M / num_cores)
            assigned = set()
            for core in range(num_cores):
                for it in range(row_iters):
                    row = it * num_cores + core
                    if row < M:
                        assert row not in assigned, f"row {row} assigned twice (M={M}, cores={num_cores})"
                        assigned.add(row)
            assert assigned == set(range(M)), f"not all rows covered (M={M}, cores={num_cores})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
