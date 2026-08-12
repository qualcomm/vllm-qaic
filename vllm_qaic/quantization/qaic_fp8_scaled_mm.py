# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""FP8 scaled-mm kernel for the QAIC OOT backend.

torch_qaic has no ``torch._scaled_mm``; it only exposes an fp8 ``torch.mm``
(``aten::mm``) that internally decodes fp8 -> fp16, runs the unmodified FP16
HMX matmul, and re-encodes to fp8.  This kernel wraps that op so that the vLLM
layer above sees an ordinary ``FP8ScaledMMLinearKernel``: the fp8 -> fp16
widening and the scale application are entirely an implementation detail of
``apply_scaled_mm``.

Two properties of the QAIC fp8 matmul drive the implementation:

1. ``mm(fp8, fp8) -> fp8``.  All three tensors (self, mat2, out) must share one
   fp8 dtype for the device kernel to be selected, so the *output* is fp8 too.
   ``apply_scaled_mm`` must return bf16/fp16, and an fp8 output would also have
   to absorb the un-dequantized product, which saturates: fp8e4m3fn has no inf,
   so overflow becomes NaN.  Measured on a representative shape (M=4, N=8, with
   typical activation/weight magnitudes), re-encoding the raw product to fp8
   yields NaN in 31/32 to 32/32 output elements for K in {128, 1024, 4096}.  A
   raw fp8-in/fp8-out matmul is therefore unusable here regardless of layout.

2. The matmul accumulates in FP16, not FP32.  fp8 operands are integer-valued
   in [-448, 448], so the *un-dequantized* running sum leaves the fp16 range
   (65504) almost immediately -- measured worst-case partial sum 3.8e6 at K=512
   and 8.9e7 at K=16384.

Both problems have the same fix: dequantize *before* the matmul instead of
after.  Folding the scales into the fp16 operands restores their true
magnitudes, so the accumulator carries values of order 1e1-1e2 rather than
1e6-1e7 (measured worst-case partial sum 9.4 at K=512, 267 at K=16384 -- three
to five orders of magnitude of headroom recovered), and the result is produced
directly in the requested output dtype with no fp8 re-encode.

Because the operands are pre-scaled, the matmul that actually runs is a plain
fp16 mm, not the device fp8 kernel.  That costs nothing real on this hardware:
there is no HMX fp8 datapath, so fp8 never offered a TOPs uplift over FP16, and
the win that remains -- halved weight bytes in LPDDR -- is preserved, since the
weights are still stored as fp8 and only widened on the fly.

Accuracy versus an exact fp32 reference is 2-4% relative error, dominated by
the fp8 quantization of the inputs itself.  The divergence attributable to this
kernel (versus an exact fp8-input GEMM with fp32 accumulation) is 0.3-0.5%
across K in {512, 4096, 16384} and all four combinations of
per-tensor/per-channel weight scales with per-tensor/per-token activation
scales, with zero non-finite outputs.
"""

import torch
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

# e5m2 is not supported by the QAIC backend at all; only the two e4m3 variants
# have cast and matmul kernels.
_QAIC_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

# Largest finite magnitude an e4m3fn operand can hold (e4m3fnuz tops out lower,
# at 240, so this is a safe upper bound for both). Used as a static bound on the
# operand magnitude so the normalisation below needs no reduction over the
# (large) A and B tensors -- only over the (small) scale tensors.
_FP8_AMAX = 448.0


def _pow2_fold(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``scale`` into an fp16-safe factor and an output correction.

    Returns ``(folded, correction)`` with ``folded * correction == scale`` and
    ``folded * _FP8_AMAX <= 1``, i.e. folding ``folded`` into an fp8 operand
    yields an fp16 operand of magnitude at most 1.

    The split is by an exact power of two, so it introduces no rounding error of
    its own. Its purpose is dynamic range: a raw fp32 scale folded straight into
    an fp16 operand underflows to zero below ~6e-5 (fp16's smallest normal), and
    per-channel weight scales do reach that neighbourhood. Normalising keeps the
    folded factor near 1/448 regardless of the incoming scale, and the
    reciprocal is applied to the wide output instead.
    """
    # clamp guards log2(0) for an all-zero weight or activation shard.
    amax = scale.float().amax().clamp(min=torch.finfo(torch.float32).tiny)
    exponent = torch.ceil(torch.log2(amax * _FP8_AMAX))
    return scale.float() * torch.exp2(-exponent), torch.exp2(exponent)


class QaicFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    """W8A8-FP8 linear kernel backed by torch_qaic's fp8-capable ``torch.mm``."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # This kernel is only ever reached through PlatformEnum.OOT, which
        # implies the QAIC platform is active. QAIC has no compute-capability
        # notion, so there is nothing further to gate on here.
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        # Block/group-wise scales require a per-group epilogue; this kernel only
        # folds per-tensor / per-token / per-channel scales into the operands.
        # (Block-quant configs are routed to a separate kernel table anyway,
        # but guard here so a mis-route fails with a clear reason.)
        if c.activation_quant_key.scale.group_shape.is_per_group():
            return False, "does not support per-group (block) activation scales"
        if c.weight_quant_key.scale.group_shape.is_per_group():
            return False, "does not support per-group (block) weight scales"

        if c.weight_quant_key.dtype not in _QAIC_FP8_DTYPES:
            return False, (
                f"requires an e4m3 fp8 weight dtype, got {c.weight_quant_key.dtype}"
            )
        if c.activation_quant_key.dtype not in _QAIC_FP8_DTYPES:
            return False, (
                "requires an e4m3 fp8 activation dtype, got "
                f"{c.activation_quant_key.dtype}"
            )

        # Asymmetric quantization would need a zero-point correction term that
        # this kernel does not compute.
        if not (c.weight_quant_key.symmetric and c.activation_quant_key.symmetric):
            return False, "requires symmetric quantization"

        return True, None

    def get_output_padding(self) -> int | None:
        # torch._scaled_mm pads the token dim to work around a batch-size
        # performance cliff; no such cliff applies here, and padded rows would
        # only have to be narrowed back off the output.
        return None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # A symmetric quantized GEMM is by definition
        #   C = (As * A) @ (Bs * B) + bias
        # The scales are folded into the widened operands rather than applied to
        # the product. Applying them afterwards would require the matmul to
        # accumulate the raw fp8 product, which leaves the FP16 accumulator
        # range almost immediately (see the module docstring); folding restores
        # the true operand magnitudes and keeps the accumulator small.
        #
        # A:  [M, K] fp8, contiguous.
        # B:  [K, N] fp8. vLLM stores the weight as [N, K] and transposes it in
        #     process_weights_after_loading, so B arrives as a non-contiguous
        #     transposed view; `.to()` preserves those strides.
        # As: per-tensor scalar / [1], or per-token [M, 1].
        # Bs: per-tensor scalar / [1], per-channel [N, 1] (compressed-tensors),
        #     or 1-D [N] (ModelOpt FP8_PER_CHANNEL_PER_TOKEN). Normalize to a
        #     row so it broadcasts across B's trailing N dimension.
        if Bs.dim() == 1:
            Bs_row = Bs.view(1, -1)
        elif Bs.dim() == 2:
            Bs_row = Bs.t()
        else:
            Bs_row = Bs

        # Normalise both scales by exact powers of two so that neither folded
        # factor underflows fp16 and both operands land at magnitude <= 1, which
        # also bounds the FP16 accumulator by K.
        As_folded, As_correction = _pow2_fold(As)
        Bs_folded, Bs_correction = _pow2_fold(Bs_row)

        a_f16 = A.to(torch.float16) * As_folded.to(torch.float16)
        b_f16 = B.to(torch.float16) * Bs_folded.to(torch.float16)

        # Widen before undoing the normalisation: the correction can be large,
        # and applying it to the fp16 accumulator could overflow it.
        output = torch.mm(a_f16, b_f16).float() * (As_correction * Bs_correction)
        output = output.to(out_dtype)

        if bias is not None:
            output = output + bias

        # apply_weights flattened a possibly-3D input to 2D; restore the shape.
        return output.view(*output_shape)
