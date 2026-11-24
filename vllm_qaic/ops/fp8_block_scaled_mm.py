# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Fast-path dispatch for the fused block-scaled FP8 GEMM NSP kernel.

The kernel lives in ``csrc/fp8_block_scaled_mm/kernel.cpp`` and is exposed as
``qaic::fp8_block_scaled_mm`` by :mod:`vllm_qaic._custom_ops`.  This module owns
the *decision* to use it: it checks the shape/dtype/VTCM preconditions, converts
the scale grids to the contiguous fp32 layout the kernel DMAs, and returns
``None`` when any precondition fails so the caller can fall back to the PyTorch
path in :mod:`vllm_qaic.quantization.qaic_fp8_block_scaled_mm`.

Why a fused kernel at all
-------------------------
The PyTorch path materializes an fp16 copy of the weight on every forward and
runs several elementwise passes over LPDDR before the matmul.  For a decode-shaped
GEMM (``M=1, N=7168, K=2048``) that is roughly 73 MB of traffic against the
14.7 MB the fp8 weight actually occupies.  At small ``M`` the op is
LPDDR-bandwidth-bound rather than compute-bound, so consuming the fp8 weight
directly -- never materializing the fp16 copy -- is the whole win.  Above
``VLLM_QAIC_FP8_BLOCK_MM_MAX_M`` the GEMM becomes compute-bound and the
HMX-backed PyTorch path is the better choice, since this kernel is HVX-only.

Why eligibility is checked here and not in the kernel
-----------------------------------------------------
``torch_qaic``'s kernel dispatcher ``TORCH_CHECK``s on any non-success status, so
a kernel-side ``JIT_DEV_ERROR_INVALID_PARAMETER`` surfaces as a hard exception
rather than a graceful fallback.  Every precondition the kernel enforces is
therefore mirrored here, and the mirror is deliberately *stricter* (see
``_vtcm_budget_bytes``): if the two ever disagree the result is a fallback to a
correct-but-slower path, never a crash.
"""

import functools

import torch
from torch import Tensor

from vllm_qaic import envs
from vllm_qaic._custom_ops import fp8_block_scaled_mm as _fp8_block_scaled_mm_op
from vllm_qaic._custom_ops import fp8_dtype_id
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

# fp8 elements per HVX vector. One K block must be a whole number of vectors so
# the kernel's inner loop never needs cross-block masking.
_HVX_FP8_ELEMS = 128

# Mirrors the kernel's `constexpr uint32_t kAlign = 128`.
_VTCM_ALIGN = 128

# Withheld from the VTCM budget below what the device reports, so this host-side
# mirror of the kernel's sizing is strictly more pessimistic than the kernel's
# own `qshimQuery(DEV_ATTR_QSHIM_VTCM_SIZE)` view even if the runtime reserves a
# slice for itself.
_VTCM_RESERVE_BYTES = 64 * 1024

_SUPPORTED_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


@functools.cache
def _vtcm_budget_bytes() -> int:
    """Per-core VTCM the kernel may assume, minus a reserve.

    Returns 0 if the device cannot be queried, which disables the fast path.
    """
    try:
        import torch_qaic

        total = int(torch_qaic.qaic.get_device_info(0).per_core_vtcm_size_byte)
    except Exception as exc:  # pragma: no cover - depends on device presence
        logger.debug("fp8_block_scaled_mm: VTCM size unavailable (%s)", exc)
        return 0
    return max(0, total - _VTCM_RESERVE_BYTES)


def _vtcm_fits(m: int, k: int, k_blocks: int, n_blocks: int) -> bool:
    """Mirror of the kernel's VTCM sizing.

    Keep in sync with ``run()`` in ``csrc/fp8_block_scaled_mm/kernel.cpp``: the
    resident buffers are A as fp8 (decode staging), A as fp16, and both scale
    grids; the B row batch and the fp32 output columns are sized from whatever
    is left, and at least one B row must fit.
    """
    budget = _vtcm_budget_bytes()
    if budget <= 0:
        return False

    fixed = (
        _align_up(m * k, _VTCM_ALIGN)  # A, fp8 staging
        + _align_up(m * k * 2, _VTCM_ALIGN)  # A, decoded to fp16
        + _align_up(m * k_blocks * 4, _VTCM_ALIGN)  # As
        + _align_up(n_blocks * k_blocks * 4, _VTCM_ALIGN)  # Bs
        + _align_up(4, _VTCM_ALIGN)  # status word
        + 8 * _VTCM_ALIGN  # per-buffer alignment slack
    )
    if fixed >= budget:
        return False

    # Two ping/pong B slots plus one fp32 output column per row in the batch.
    per_row = 2 * k + m * 4
    return (budget - fixed) // per_row >= 1


def is_eligible(
    A: Tensor,
    B: Tensor,
    As: Tensor,
    Bs: Tensor,
    block_n: int,
    block_k: int,
) -> bool:
    """Whether this shape/dtype/config can be served by the fused NSP kernel.

    ``A`` is ``[M, K]`` fp8, ``B`` is ``[N, K]`` fp8 (not pre-transposed), and
    the scale grids are as documented on ``apply_block_scaled_mm``.  Ragged ``N``
    is fine -- the kernel's ``n / block_n`` indexing lands the tail rows in the
    final scale block -- but ragged ``K`` is not, and stays on the PyTorch path.
    """
    if envs.VLLM_QAIC_FP8_BLOCK_MM_DISABLE:
        return False

    if A.dim() != 2 or B.dim() != 2:
        return False
    m, k = A.shape
    n, k_b = B.shape
    if k_b != k:
        return False

    if m > envs.VLLM_QAIC_FP8_BLOCK_MM_MAX_M:
        return False

    if block_k <= 0 or block_n <= 0:
        return False
    if block_k % _HVX_FP8_ELEMS != 0 or k % block_k != 0:
        return False

    if A.dtype not in _SUPPORTED_FP8_DTYPES or B.dtype != A.dtype:
        return False

    # The kernel DMAs A and B as single linear blocks.
    if not A.is_contiguous() or not B.is_contiguous():
        return False

    k_blocks = k // block_k
    n_blocks = _cdiv(n, block_n)
    if tuple(As.shape) != (m, k_blocks):
        return False
    if tuple(Bs.shape) != (n_blocks, k_blocks):
        return False

    return _vtcm_fits(m, k, k_blocks, n_blocks)


def fp8_block_scaled_mm(
    A: Tensor,
    B: Tensor,
    As: Tensor,
    Bs: Tensor,
    block_n: int,
    block_k: int,
) -> Tensor | None:
    """Run the fused block-scaled FP8 GEMM, or return ``None`` if ineligible.

    Args:
        A: Activation, ``[M, K]`` fp8, contiguous.
        B: Weight, ``[N, K]`` fp8, contiguous.
        As: Activation scales, ``[M, K // block_k]``, already upcast to fp32.
        Bs: Weight scales, ``[cdiv(N, block_n), K // block_k]``, fp32.
        block_n: Weight scale block extent along N.
        block_k: Scale block extent along K.

    Returns:
        ``[M, N]`` fp32, or ``None`` when the caller should use the PyTorch path.
    """
    if not is_eligible(A, B, As, Bs, block_n, block_k):
        return None

    # The scale grids are tiny ([M, K/block_k] and [N/block_n, K/block_k]), so
    # normalising them to contiguous fp32 costs nothing and is usually a no-op.
    As = As.to(dtype=torch.float32).contiguous()
    Bs = Bs.to(dtype=torch.float32).contiguous()

    return _fp8_block_scaled_mm_op(
        A,
        B,
        As,
        Bs,
        block_n,
        block_k,
        fp8_dtype_id(A.dtype),
    )
