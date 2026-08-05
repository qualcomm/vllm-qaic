# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

# -------------------------------------------------------------------
# This module manage the patch for vllm. Once a new patch is added in
# vllm-qaic, please add the patch description into this file
# -------------------------------------------------------------------

# What's Patched and how it works:
# --------------------------------
# ** 1. File: patch_config.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.config.cache.CacheConfig
#    Why:
#       Need to add an additional QAIC‑supported CacheDtype: `mxint8`.
#    How:
#       Extend CacheConfig and update _validate_cache_dtype.
#    2. vllm.config.cache.DeviceConfig
#     Why:
#       For QAIC in AOT mode, the torch device should be `cpu` instead of `qaic`.
#     How:
#       Extend DeviceConfig and update __post_init__.
#
# ** 2. File: patch_parallel_state.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.distributed.parallel_state.GroupCoordinator
#     Why:
#       For QAIC in AOT mode, the torch device should be `cpu` instead of `qaic`.
#     How:
#       Extend GroupCoordinator and update __init__.
#
# ** 3. File: patch_utils.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.utils.torch_utils.STR_DTYPE_TO_TORCH_DTYPE
#    Why:
#       Need to add an additional QAIC‑supported CacheDtype: `mxint8`.
#    How:
#       Add the corresponding key/value to STR_DTYPE_TO_TORCH_DTYPE.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ** 4. File: patch_rejection_sampler.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.v1.sample.rejection_sampler.RejectionSampler.forward
#    Why:
#       Skip unnecessary tensor clone and softmax in the greedy-sampling
#       path of the rejection sampler.  For greedy requests with no logprobs,
#       (a) the clone before apply_logits_processors is not needed because
#       raw_target_logits is never read again, and (b) softmax can be skipped
#       because argmax(logits) == argmax(softmax(logits)).
#    How:
#       Replace RejectionSampler.forward with a QAIC-optimized version.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ** 5. File: patch_graph_pickler.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. torch.fx._graph_pickler.Options
#   2. torch.fx._graph_pickler.GraphPickler.dumps
#    Why:
#       QAIC uses torch==2.7.0 (CPU-only build), but vllm.compilation.caching
#       imports `Options` from torch.fx._graph_pickler and passes it to
#       GraphPickler.dumps — both of which were only added in torch 2.8.0.
#       Without this shim the EngineCore subprocess crashes at import time.
#    How:
#       If `Options` is absent, inject a compatible dataclass into
#       torch.fx._graph_pickler and wrap GraphPickler.dumps to accept and
#       silently ignore the options argument. No-op on torch >= 2.8.
#    Note:
#       This patch must be applied before vllm.compilation.caching is imported.
#       Because vLLM defaults to fork-based subprocesses, patching here (in the
#       main process before fork) is sufficient.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ** 6. File: patch_triton_import.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.triton_utils.HAS_TRITON / triton / tl / tldevice
#    Why:
#       Triton is required to run kernels on the QAIC Triton Backend, but
#       upstream's HAS_TRITON check can end up False even when a usable
#       Triton install is present (e.g. its Triton-CPU-backend check reads
#       `importlib.metadata.version("vllm")`, which raises when vllm is
#       installed under a different distribution name).
#    How:
#       Re-import triton and check for QAIC Triton Backend in
#       triton.backends.backends; if found, re-enable HAS_TRITON and rebind
#       triton/tl/tldevice on vllm.triton_utils to the real modules.
#    Note:
#       Must run before any vllm module does
#       `from vllm.triton_utils import ...`, since that binds triton/tl/
#       HAS_TRITON into the importing module's own namespace at that
#       moment. Applied here (during pre_register_and_update) is early
#       enough, since all such imports live under model_executor/layers/
#       and v1/worker/, which only load during model/worker construction.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ** 7. File: patch_linear.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.model_executor.layers.linear.UnquantizedLinearMethod.apply
#   2. vllm.model_executor.layers.vocab_parallel_embedding.
#        UnquantizedEmbeddingMethod.apply
#    Why:
#       The linear/GEMM family (QKV/Column/Row/MergedColumn ParallelLinear) and
#       the LM-head/logits matmul do not inherit CustomOp, so they have no
#       forward_oot hook. To route their GEMM to vLLM's Triton kernel
#       (linear_batch_invariant) we must patch their `apply` method.
#    How:
#       Reassign both `apply` methods to a wrapper that calls
#       linear_batch_invariant when VLLM_QAIC_TRITON_LINEAR is set, else the
#       original dispatch_unquantized_gemm path.
#    Note:
#       Imported only when VLLM_QAIC_TRITON_LINEAR is set (see below). Requires
#       the QAIC platform to implement num_compute_units (matmul_persistent
#       reads it) — added in platform_base.py.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =================

import vllm_qaic.patch.patch_config  # noqa
import vllm_qaic.patch.patch_parallel_state  # noqa
import vllm_qaic.patch.patch_utils  # noqa
import vllm_qaic.patch.patch_rejection_sampler  # noqa
import vllm_qaic.patch.patch_graph_pickler  # noqa
import vllm_qaic.patch.patch_mem_utils  # noqa

import vllm_qaic.envs as qaic_envs

# The numpy slot-mapping patch bypasses vLLM's Triton slot-mapping kernel. When
# the in-tree Triton attention backend is enabled (VLLM_QAIC_ENABLE_TRITON_ATTN),
# the real Triton slot-mapping kernel must stay live to feed slot_mapping into
# triton_reshape_and_cache_flash, so skip this patch in that mode.
if not qaic_envs.VLLM_QAIC_ENABLE_TRITON_ATTN:
    import vllm_qaic.patch.patch_block_table  # noqa

# Route the unquantized linear/GEMM family to vLLM's Triton GEMM only when the
# opt-in flag is set; otherwise leave the default dispatch_unquantized_gemm path.
if qaic_envs.VLLM_QAIC_TRITON_LINEAR:
    import vllm_qaic.patch.patch_linear  # noqa
