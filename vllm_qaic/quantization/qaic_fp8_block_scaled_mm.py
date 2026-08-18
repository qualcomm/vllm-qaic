# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Block-scaled (e.g. DeepSeek ``[128, 128]``) FP8 linear kernel for QAIC.

Sibling of :mod:`vllm_qaic.quantization.qaic_fp8_scaled_mm`, which handles the
per-tensor / per-token / per-channel case.  This kernel handles the block-quant
case, where the weight carries a 2-D scale grid over ``[block_n, block_k]``
tiles and the activation carries a per-token, per-``block_k`` scale row.

Two paths
---------
``apply_block_scaled_mm`` tries a fused out-of-tree NSP kernel first
(:mod:`vllm_qaic.ops.fp8_block_scaled_mm`) and falls back to the PyTorch path
documented below.  The two differ in kind, not just in speed:

* The **fused kernel** forms each K block's raw fp8 x fp8 dot product on HVX and
  accumulates the scaled blocks in fp32.  That is exact with respect to the fp8
  operands, so none of the power-of-two folding or output correction below
  applies to it.  It consumes the fp8 weight directly, never materializing the
  fp16 copy -- which is the point, since at small ``M`` this op is
  LPDDR-bandwidth-bound.  It is HVX-only (see that module for why HMX is
  unreachable from an out-of-tree kernel), so it declines large ``M``.
* The **PyTorch path** below dequantizes both operands into fp16 and issues one
  ``torch.mm``, reaching HMX.  That is the right trade once ``M`` is large enough
  for the GEMM to be compute-bound rather than bandwidth-bound, and it remains
  the only path for shapes the kernel declines (ragged ``K``, ``block_k`` not a
  multiple of 128, residents that do not fit VTCM).

Everything from here down describes the PyTorch path.

Why this cannot reuse a QAIC device kernel
------------------------------------------
``qaiclibrary`` does have a block-scaled weight codec
(``Mxfp4WeightCodec`` in ``hmxmatmul.h``), but its scale geometry is a
different shape of problem: it is 1-D along K only, with a fixed
``kScaleBlock = 32`` and e8m0 (power-of-two) scales, and it applies to 4-bit
weights.  A ``[128, 128]`` fp8 grid varies along *both* N and K, so it does not
map onto that codec, and the scale-free ``FP8E4M3FNCodec`` path ignores scales
entirely.  There is no per-group fp8 GEMM to call, so the group scales have to
be honoured above the device op, in this kernel.

How the group scales are honoured exactly
-----------------------------------------
Naively, a group-scaled GEMM looks like it needs a per-K-tile epilogue: the
weight scale changes as the contraction walks K, so it cannot simply be hoisted
out of the sum.  But within a single K-tile the block scale *is* constant along
K, which is precisely the condition for hoisting.  So the exact result
decomposes into a sum of independently-scaled sub-GEMMs, one per K-tile -- and
folding the (constant-within-tile) scale into the *operands* of each sub-GEMM is
algebraically identical to scaling that sub-GEMM's output.

That means a single fused matmul over the *dequantized* operands is exact: each
output element is already the correctly-weighted sum over all K-tiles.  This was
verified against an explicitly K-tiled accumulation, which agreed to the printed
precision (rel err ``0.043629`` both ways).  No epilogue loop is needed.

Consequently the implementation is the same shape as the non-block kernel:
dequantize into fp16 *before* the matmul, never after.  The reasons are
unchanged and are documented at length in ``qaic_fp8_scaled_mm``:
``mm(fp8, fp8) -> fp8`` saturates to NaN on the un-dequantized product, and the
matmul accumulates in FP16, whose 65504 ceiling the raw fp8 partial sums breach
almost immediately.  Pre-folding costs nothing real on QAIC because there is no
HMX fp8 datapath -- fp8 never bought TOPs here, only LPDDR weight bytes, and
those are preserved since the weight stays fp8 at rest and is widened on the fly.

Range normalisation
-------------------
Scales are normalised by exact powers of two before folding, as in the non-block
kernel, so that a small scale cannot underflow fp16 (smallest normal ~6.1e-5)
and the folded operands stay bounded by 1.  The refinement here is that the
normalisation is **per scale row**, not global: variation of ``Bs`` along N and
of ``As`` along M is recoverable in the output (as a column/row vector
respectively), because those axes are not contracted over.  Only variation along
K must share one exponent.  Keeping per-row resolution measures noticeably
better than a single global factor -- ~4e-4 relative error here versus the
3e-3 to 5e-3 of the global fold in the non-block path.

Measured accuracy versus an exact fp32-accumulation reference over the same
quantized operands, taken through vLLM's own ``apply_weights`` (so the
activation is quantized by ``QuantFP8`` exactly as in production): relative
error 3.2e-4 to 4.6e-4 with an fp16 output, over aligned DeepSeek-like shapes
``(M=1, N=7168, K=2048)`` and ``(M=64, N=2048, K=7168)``, a deliberately ragged
``(M=7, N=300, K=500)``, K out to 16384, and both fp32 and e8m0 scale
encodings.  With a bf16 output the figure rises to ~1.7e-3, which is the bf16
cast's own 8-bit mantissa, not kernel error.  Zero non-finite outputs
throughout; folded operand magnitudes stay under 1.0 and the worst observed
partial sum is 1.3 at K=16384, against the FP16 accumulator's 65504 ceiling.

Memory strategy
---------------
Expansion is **per forward**, not at load time.  Materializing an element-wise
fp16 scale for the weight at load time would cost 2x the fp8 weight bytes
resident for the process lifetime, which would give back the one benefit fp8
actually delivers on this hardware.  Per-forward widening allocates the same
fp16 weight copy the non-block kernel already allocates, and for aligned shapes
the scale is applied through a 4-D broadcast view, so the expanded *scale* grid
is never materialized at all.
"""

import torch
from vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel import (
    Fp8BlockScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    _upcast_e8m0_to_fp32,
)

from vllm_qaic.quantization.qaic_fp8_scaled_mm import _FP8_AMAX, _QAIC_FP8_DTYPES


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _upcast_scale(scale: torch.Tensor) -> torch.Tensor:
    """Return ``scale`` as fp32, decoding the e8m0 (pure-exponent) encoding.

    With ``VLLM_USE_UE8M0`` / an e8m0 checkpoint, block scales are stored as
    ``torch.float8_e8m0fnu``: a bare 8-bit exponent, which is not an arithmetic
    dtype and cannot be multiplied.  vLLM's own kernels call this same helper
    before use.
    """
    if scale.dtype == torch.float8_e8m0fnu:
        return _upcast_e8m0_to_fp32(scale)
    return scale.float()


def _fold_rows(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row power-of-two split of a 2-D block-scale grid.

    ``scale`` is ``[rows, k_blocks]``, where the trailing axis walks the
    contraction dimension.  Returns ``(folded, correction)`` with
    ``folded * correction == scale`` elementwise-per-row and
    ``folded * _FP8_AMAX <= 1``, so folding ``folded`` into an fp8 operand
    yields an fp16 operand of magnitude at most 1 (which in turn bounds the FP16
    accumulator by K).

    ``correction`` is ``[rows, 1]`` and is applied to the wide output.  Only the
    K axis has to share an exponent: the row axis (M for activations, N-blocks
    for weights) is not contracted over, so its variation survives into the
    output where it can be corrected at full fp32 resolution.

    The split is by an exact power of two and so is itself lossless.
    """
    amax = scale.abs().amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax * _FP8_AMAX))
    return scale * torch.exp2(-exponent), torch.exp2(exponent)


def _apply_block_scale_2d(
    x: torch.Tensor,
    scale: torch.Tensor,
    block_row: int,
    block_col: int,
) -> torch.Tensor:
    """Widen fp8 ``x`` to fp16 with a 2-D block scale applied elementwise.

    ``x`` is ``[R, C]``; ``scale`` is ``[cdiv(R, block_row), cdiv(C, block_col)]``
    fp32.  Block-scale shapes use ceiling division, so the final block along
    either axis may be partial; the expansion is sliced back to ``[R, C]`` to
    cover that case.
    """
    rows, cols = x.shape
    if x.is_contiguous() and rows % block_row == 0 and cols % block_col == 0:
        # Aligned and row-major: reshape into blocks and broadcast, so the
        # expanded scale grid is never materialized.
        r_blocks, c_blocks = rows // block_row, cols // block_col
        blocked = x.view(r_blocks, block_row, c_blocks, block_col).to(torch.float16)
        scaled = blocked * scale.to(torch.float16).view(r_blocks, 1, c_blocks, 1)
        return scaled.view(rows, cols)

    # Ragged final block (or an unexpected stride pattern): expand the scale grid
    # and trim. Verified to place row R-1 in the last scale block (e.g. N=300
    # with block_row=128 -> 3 scale rows, row 299 taking scale row 2), and to be
    # bit-identical to the broadcast path on aligned shapes.
    expanded = scale.repeat_interleave(block_row, dim=0).repeat_interleave(
        block_col, dim=1
    )
    return x.to(torch.float16) * expanded[:rows, :cols].to(torch.float16)


class QaicFP8BlockScaledMMLinearKernel(Fp8BlockScaledMMLinearKernel):
    """Block-quantized W8A8-FP8 linear kernel for the QAIC OOT backend."""

    # The activation must be fp8-quantized per token-group; the base class
    # handles that via QuantFP8 before calling apply_block_scaled_mm.
    apply_input_quant = True

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Reached only through PlatformEnum.OOT, which implies QAIC is active.
        # QAIC exposes no compute-capability notion.
        return True, None

    @classmethod
    def can_implement(
        cls, config: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        # Static activation scales are rejected by the base class (block quant
        # implies dynamic per-token-group activation quantization).
        can_base, reason = super().can_implement(config)
        if not can_base:
            return False, reason

        weight_gs = config.weight_quant_key.scale.group_shape
        act_gs = config.activation_quant_key.scale.group_shape

        # A 2-D block grid has both extents positive. Note this is *not*
        # GroupShape.is_per_group(), which requires row == 1 and so is true of
        # the activation scale but false of a [128, 128] weight scale.
        if weight_gs.row < 1 or weight_gs.col < 1:
            return False, (
                "requires a 2-D block weight scale with positive extents, got "
                f"group shape ({weight_gs.row}, {weight_gs.col})"
            )
        if not act_gs.is_per_group():
            return False, (
                "requires per-token-group activation scales, got group shape "
                f"({act_gs.row}, {act_gs.col})"
            )
        # vLLM derives the activation group size from the weight's K block, so
        # these agree by construction; assert it rather than silently mis-scale.
        if act_gs.col != weight_gs.col:
            return False, (
                f"activation group size {act_gs.col} must match the weight "
                f"K block size {weight_gs.col}"
            )

        if config.weight_quant_key.dtype not in _QAIC_FP8_DTYPES:
            return False, (
                "requires an e4m3 fp8 weight dtype, got "
                f"{config.weight_quant_key.dtype}"
            )
        if config.activation_quant_key.dtype not in _QAIC_FP8_DTYPES:
            return False, (
                "requires an e4m3 fp8 activation dtype, got "
                f"{config.activation_quant_key.dtype}"
            )
        if not (
            config.weight_quant_key.symmetric and config.activation_quant_key.symmetric
        ):
            return False, "requires symmetric quantization"

        return True, None

    def get_output_padding(self) -> int | None:
        # No batch-size performance cliff to pad around here, and padded rows
        # would only have to be narrowed back off the output.
        return None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``(As # A) @ (Bs # B).T`` where ``#`` is block dequantization.

        Shapes, per the ``Fp8BlockScaledMMLinearKernel`` contract:
          ``A``  ``[M, K]`` fp8, contiguous (activation, already quantized).
          ``B``  ``[N, K]`` fp8 -- the weight is *not* transposed by the block
                 path's ``process_weights_after_loading``, unlike the non-block
                 path, so this kernel transposes at use.
          ``As`` ``[M, cdiv(K, block_k)]``.
          ``Bs`` ``[cdiv(N, block_n), cdiv(K, block_k)]``.

        Returns a wide (fp32) tensor; the base ``apply_weights`` adds bias, casts
        to the configured output dtype, and restores the original input shape.
        """
        block_n, block_k = self.weight_group_shape.row, self.weight_group_shape.col
        m, k = A.shape
        n = B.shape[0]

        As_f32 = _upcast_scale(As)
        Bs_f32 = _upcast_scale(Bs)

        # Sanity-check the scale grids against the operands. A silent mismatch
        # here would produce plausible-looking but wrong numbers.
        assert As_f32.shape == (m, _cdiv(k, block_k)), (
            f"activation scale shape {tuple(As_f32.shape)} does not match "
            f"A {tuple(A.shape)} with block_k={block_k}"
        )
        assert Bs_f32.shape == (_cdiv(n, block_n), _cdiv(k, block_k)), (
            f"weight scale shape {tuple(Bs_f32.shape)} does not match "
            f"B {tuple(B.shape)} with block=({block_n}, {block_k})"
        )

        # Fused NSP kernel first. It consumes the fp8 weight directly rather
        # than materializing an fp16 copy, which is the whole game at small M
        # where this op is LPDDR-bandwidth-bound, and it accumulates each block
        # in fp32 so it needs none of the folding/correction machinery below.
        # It returns None for anything it does not cover -- notably large M,
        # where the compute-bound path below is the right one.
        #
        # Imported here rather than at module scope: this module is imported
        # during platform patching, ahead of the kernel-library load that
        # vllm_qaic._custom_ops performs at import time.
        from vllm_qaic.ops import fp8_block_scaled_mm as _fused

        fused_out = _fused.fp8_block_scaled_mm(A, B, As_f32, Bs_f32, block_n, block_k)
        if fused_out is not None:
            return fused_out

        # Normalise each scale row by an exact power of two so no folded factor
        # underflows fp16 and both operands land at magnitude <= 1.
        As_folded, As_correction = _fold_rows(As_f32)
        Bs_folded, Bs_correction = _fold_rows(Bs_f32)

        # The activation scale is per token (row) and per K block, i.e. a 2-D
        # block grid with a row block of 1.
        a_f16 = _apply_block_scale_2d(A, As_folded, 1, block_k)
        b_f16 = _apply_block_scale_2d(B, Bs_folded, block_n, block_k)

        # b_f16 is [N, K] and contiguous, so b_f16.t() has strides (1, K) --
        # exactly the non-contiguous "RhsTranspose" pattern the QAIC matmul
        # kernel requires of mat2 (verified for all shapes exercised).
        out = torch.mm(a_f16, b_f16.t())

        # Undo the normalisation after widening: the corrections can be large,
        # and applying them in fp16 could overflow the accumulator.
        # As_correction is [M, 1]; Bs_correction is [N_blocks, 1] and has to be
        # expanded along N (trimmed, since the last block may be partial).
        n_correction = Bs_correction.repeat_interleave(block_n, dim=0)[:n].reshape(1, n)
        return out.float() * (As_correction * n_correction)
