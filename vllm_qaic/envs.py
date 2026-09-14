# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.envs import (
    environment_variables,
    maybe_convert_bool,
    maybe_convert_int,
)


if TYPE_CHECKING:
    VLLM_QAIC_COMPILER_ARGS: str | None = None
    VLLM_QAIC_DFS_EN: bool = True
    VLLM_QAIC_MAX_CPU_THREADS: int | None = None
    VLLM_QAIC_MOS: int | None = None
    VLLM_QAIC_NUM_CORES: int | None = None
    VLLM_QAIC_QPC_PATH: str | None = None
    VLLM_TORCH_QAIC_PROFILER_DIR: str | None = None
    # Route the decoder paged-attention backend to vLLM's in-tree Triton
    # attention (unified_attention + triton_reshape_and_cache_flash)
    VLLM_QAIC_ENABLE_TRITON_ATTN: bool = False
    # Per-operator opt-in flags: dispatch the CustomOp forward_oot to the
    # corresponding vLLM Triton kernel
    VLLM_QAIC_TRITON_RMS_NORM: bool = False
    VLLM_QAIC_TRITON_MM_ENCODER_ATTENTION: bool = False
    VLLM_QAIC_TRITON_MROTARY_EMBEDDING: bool = False
    # Repurpose vLLM's SWIGLUSTEP Triton kernel (swiglustep_and_mul_triton) as
    # plain SiluAndMul (limit=+inf); routes the SwiGLU MLP activation to Triton.
    VLLM_QAIC_TRITON_SILU_AND_MUL: bool = False
    # Route the unquantized fused-MoE forward to vLLM's hand-written Triton
    # fused-MoE kernel (fused_experts -> fused_moe_kernel)
    VLLM_QAIC_TRITON_FUSED_MOE: bool = False
    # Route the unquantized linear/GEMM family (QKV/Column/Row/MergedColumn
    # ParallelLinear, ParallelLMHead, LogitsProcessor matmul) to vLLM's
    # batch-invariant Triton GEMM (linear_batch_invariant) via patch_linear.
    VLLM_QAIC_TRITON_LINEAR: bool = False
    # Route the sampler's top-k/top-p logits masking to vLLM's Triton kernel
    # (apply_top_k_top_p_triton) via patch_topk_topp_sampler; when unset the
    # PyTorch sort implementation is used regardless of batch size.
    VLLM_QAIC_TRITON_TOPK_TOPP: bool = False

# --8<-- [start:env-vars-definition]
qaic_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_QAIC_COMPILER_ARGS": lambda: os.getenv("VLLM_QAIC_COMPILER_ARGS", None),
    "VLLM_QAIC_DFS_EN": lambda: maybe_convert_bool(os.getenv("VLLM_QAIC_DFS_EN", None)),
    "VLLM_QAIC_MAX_CPU_THREADS": lambda: maybe_convert_int(
        os.getenv("VLLM_QAIC_MAX_CPU_THREADS", None)
    ),
    "VLLM_QAIC_MOS": lambda: maybe_convert_int(os.getenv("VLLM_QAIC_MOS", None)),
    "VLLM_QAIC_NUM_CORES": lambda: maybe_convert_int(
        os.getenv("VLLM_QAIC_NUM_CORES", None)
    ),
    "VLLM_QAIC_QPC_PATH": lambda: os.getenv("VLLM_QAIC_QPC_PATH", None),
    "VLLM_TORCH_QAIC_PROFILER_DIR": lambda: os.getenv(
        "VLLM_TORCH_QAIC_PROFILER_DIR", None
    ),
    "VLLM_QAIC_ENABLE_TRITON_ATTN": lambda: os.getenv(
        "VLLM_QAIC_ENABLE_TRITON_ATTN", "0"
    )
    == "1",
    "VLLM_QAIC_TRITON_RMS_NORM": lambda: os.getenv("VLLM_QAIC_TRITON_RMS_NORM", "0")
    == "1",
    "VLLM_QAIC_TRITON_MM_ENCODER_ATTENTION": lambda: os.getenv(
        "VLLM_QAIC_TRITON_MM_ENCODER_ATTENTION", "0"
    )
    == "1",
    "VLLM_QAIC_TRITON_MROTARY_EMBEDDING": lambda: os.getenv(
        "VLLM_QAIC_TRITON_MROTARY_EMBEDDING", "0"
    )
    == "1",
    "VLLM_QAIC_TRITON_SILU_AND_MUL": lambda: os.getenv(
        "VLLM_QAIC_TRITON_SILU_AND_MUL", "0"
    )
    == "1",
    "VLLM_QAIC_TRITON_FUSED_MOE": lambda: os.getenv("VLLM_QAIC_TRITON_FUSED_MOE", "0")
    == "1",
    "VLLM_QAIC_TRITON_LINEAR": lambda: os.getenv("VLLM_QAIC_TRITON_LINEAR", "0") == "1",
    "VLLM_QAIC_TRITON_TOPK_TOPP": lambda: os.getenv("VLLM_QAIC_TRITON_TOPK_TOPP", "0")
    == "1",
}
environment_variables.update(qaic_environment_variables)
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """
    Gets environment variables lazily.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
