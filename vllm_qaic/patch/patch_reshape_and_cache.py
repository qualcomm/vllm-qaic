# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/attention/ops/triton_reshape_and_cache_flash.py

"""Patch triton_reshape_and_cache_flash to drop the CUDA compute-capability query.

Under the in-tree Triton attention backend (VLLM_QAIC_ENABLE_TRITON_ATTN=1) every
KV-cache write goes through
``vllm.v1.attention.ops.triton_reshape_and_cache_flash.triton_reshape_and_cache_flash``,
the Python wrapper that marshals strides and launch parameters for the
``reshape_and_cache_kernel_flash`` Triton kernel.

Its launch heuristic branches on the platform: ROCm and XPU get
8 warps x 4 stages, and *everything else* falls into the CUDA branch, which
takes 16 warps x 10 stages and then clamps TILE_SIZE for pre-Hopper GPUs via
``torch.cuda.get_device_capability(key.device)``. QAIC is neither ROCm nor XPU,
so it lands in that branch, and the query raises on a CUDA-less build --
failing the KV-cache write on every forward pass.

This is the upstream wrapper re-implemented for QAIC:
  * the compute-capability clamp is gone, so TILE_SIZE stays at
    ``min(2048, next_power_of_2(num_heads * head_size))``;
  * the ROCm/XPU vs CUDA branch is gone -- this wrapper is only installed on the
    QAIC platform, where both checks are always False -- replaced by a fixed
    16 warps x 4 stages launch config.
The kernel itself and all stride/dtype marshalling are upstream's, unchanged.
"""

import torch
import triton

from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.ops import (
    triton_reshape_and_cache_flash as reshape_and_cache_ops,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    _is_supported_kv_cache_dtype,
    reshape_and_cache_kernel_flash,
)

# QAIC launch config for reshape_and_cache_kernel_flash: 16 warps x 4 stages.
QAIC_NUM_WARPS = 16
QAIC_NUM_STAGES = 4


def _qaic_triton_reshape_and_cache_flash(
    key: torch.Tensor,  # [num_tokens, num_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_heads, head_size]
    # [num_blocks, block_size, num_heads, head_size]
    key_cache: torch.Tensor,
    # [num_blocks, block_size, num_heads, head_size]
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,  # "auto", "fp8"
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
):
    num_heads = key.shape[1]
    head_size = key.shape[2]

    use_head_major_layout = key_cache.ndim == 5
    if use_head_major_layout:
        block_size = key_cache.shape[3]
        x = key_cache.shape[4]
        head_stride = key_cache.stride(1)
        dim_stride_k = key_cache.stride(2)
        dim_stride_v = value_cache.stride(2)
    else:
        block_size = key_cache.shape[1]
        x = 1
        dim_stride_k = 0
        dim_stride_v = 0
        head_stride = key_cache.stride()[2]
    n = num_heads * head_size
    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    assert _is_supported_kv_cache_dtype(kv_cache_dtype), (
        f"unsupported kv_cache_dtype (str), got {kv_cache_dtype}."
    )
    kv_cache_torch_dtype = (
        current_platform.fp8_dtype()
        if is_quantized_kv_cache(kv_cache_dtype)
        else key_cache.dtype
    )

    if key_cache.dtype != kv_cache_torch_dtype and is_quantized_kv_cache(
        kv_cache_dtype
    ):
        # to avoid erounous implicit cast in triton kernel (tl.store to uint8)
        # (e.g. explicit cast to fp8e4m3fnuz is not supported in triton 3.4)
        key_cache = key_cache.view(kv_cache_torch_dtype)
        value_cache = value_cache.view(kv_cache_torch_dtype)
    assert kv_cache_dtype != torch.uint8, (
        "explicit fp8 cast and store to "
        "uint8 is not supported by triton reshape_and_cache_flash"
    )

    FP8_KV_CACHE = is_quantized_kv_cache(kv_cache_dtype)
    assert (not FP8_KV_CACHE) or kv_cache_torch_dtype in [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.uint8,
        torch.float8_e4m3fnuz,
    ], (
        "unsupported dtype of KV cache tensor, got "
        "{kv_cache_torch_dtype}. Supported kv cache dtypes: fp8e4m3fn, "
        "fp8e5m2, uint8, bfloat16, float16, float32, fp8e4m3fnuz."
    )

    # heuristics instead of autotuning
    TILE_SIZE = min(2048, triton.next_power_of_2(n))

    # TODO(ngl): maybe replace with static launch grid to avoid overhead if
    #   using cudagraphs
    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        # strides
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        head_stride=head_stride,
        dim_stride_k=dim_stride_k,
        dim_stride_v=dim_stride_v,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        x=x,
        USE_HEAD_MAJOR_LAYOUT=use_head_major_layout,
        FP8_KV_CACHE=FP8_KV_CACHE,
        # autotune parameters
        TILE_SIZE=TILE_SIZE,
        num_warps=QAIC_NUM_WARPS,
        num_stages=QAIC_NUM_STAGES,
    )


reshape_and_cache_ops.triton_reshape_and_cache_flash = (
    _qaic_triton_reshape_and_cache_flash
)
