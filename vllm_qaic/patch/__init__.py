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
# ** 6. File: patch_qaic_triton_import.py **
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
#       moment. The patch is applied in QAicPlatform::pre_register_and_update()
#       after checking that the inference mode is PyT and not AoT.
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
#
# ** 8. File: patch_topk_topp_sampler.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.v1.sample.ops.topk_topp_sampler.apply_top_k_top_p
#    Why:
#       Upstream selects the Triton top-k/top-p kernel implicitly, from
#       HAS_TRITON plus a CUDA-tuned batch-size heuristic (Triton at batch >= 8,
#       PyTorch sort below). On QAIC HAS_TRITON is force-enabled by
#       patch_qaic_triton_import, so the Triton sampler kernel is selected with
#       no way to opt out and no way to exercise it below batch 8.
#    How:
#       Replace apply_top_k_top_p with a version gated solely on
#       VLLM_QAIC_TRITON_TOPK_TOPP (read per call via triton_op_enabled):
#       set to 1 -> apply_top_k_top_p_triton at any batch size, otherwise ->
#       apply_top_k_top_p_pytorch.
#    Note:
#       Applied first, because apply_top_k_top_p is a module-level function that
#       vllm.v1.sample.rejection_sampler (imported by patch_rejection_sampler
#       below) and vllm.v1.worker.gpu.sample.{sampler,states} pull in with
#       `from ... import apply_top_k_top_p`, which copies the object into their
#       namespace at import time. None of them is imported yet at this point, so
#       patching the definition site first is enough for all of them.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ** 9. File: patch_reshape_and_cache.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.v1.attention.ops.triton_reshape_and_cache_flash.
#        triton_reshape_and_cache_flash
#    Why:
#       The wrapper that launches reshape_and_cache_kernel_flash picks its launch
#       parameters per platform: ROCm/XPU get 8 warps x 4 stages, everything else
#       falls into the CUDA branch, which clamps TILE_SIZE via
#       torch.cuda.get_device_capability(). QAIC is neither ROCm nor XPU, so it
#       lands in that branch and the query raises without a CUDA runtime,
#       failing every KV-cache write under the Triton attention backend.
#    How:
#       Re-implement the wrapper without the compute-capability clamp and
#       without the platform branch (both checks are always False on QAIC),
#       using a fixed 16 warps x 4 stages launch config. The kernel and all
#       stride/dtype marshalling are upstream's, unchanged.
#    Note:
#       Imported only when VLLM_QAIC_ENABLE_TRITON_ATTN is set, since that
#       backend is the only caller of this wrapper on QAIC.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ** 10. File: patch_gemma4_dtype.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.model_executor.models.gemma4_mm.Gemma4ForConditionalGeneration.__init__
#    Why:
#       The vision/audio towers are built with AutoModel.from_config(sub_config),
#       which uses the sub-config's own dtype (bf16) and ignores --dtype.
#    How:
#       Set dtype/torch_dtype on the config and its sub-configs to
#       model_config.dtype before calling the original __init__.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ** 11. File: patch_mem_utils.py (addition) **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. torch.accelerator.get_memory_info
#    Why:
#       Model code (gemma4_mm video encoder) calls it directly and it asserts a
#       PyTorch DeviceAllocator, which QAIC does not use.
#    How:
#       Rebind it to current_platform.mem_get_info.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ** 12. File: patch_gemma4_router.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.model_executor.models.gemma4.gemma4_routing_function_torch
#    Why:
#       Gemma4MoE builds a routing_function closure (top-k, softmax,
#       renormalisation, per-expert scale) and hands it to the fused-MoE layer
#       as custom_routing_function. The closure calls the Triton kernel
#       gemma4_fused_routing_kernel_triton only when current_platform is
#       CUDA-alike or XPU, and otherwise calls gemma4_routing_function_torch.
#       QAIC is neither, so Triton is unreachable (not a CustomOp, so
#       forward_oot can't dispatch it either).
#    How:
#       The closure resolves gemma4_routing_function_torch as a module global at
#       call time, so rebind that name to a wrapper gated solely on
#       VLLM_QAIC_TRITON_GEMMA4_ROUTER (read per call via triton_op_enabled):
#       set to 1 -> upstream's gemma4_fused_routing_kernel_triton, otherwise ->
#       the original torch function, saved before the rebind.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ** 13. File: patch_fused_moe_activation.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm.model_executor.layers.fused_moe.fused_moe.apply_moe_activation
#    Why:
#       fused_experts_impl (reached when VLLM_QAIC_TRITON_FUSED_MOE is set)
#       applies the activation between the two expert GEMMs through
#       apply_moe_activation, which for GELU_TANH (Gemma 4) calls the compiled
#       op torch.ops._C.gelu_tanh_and_mul directly. It is not a CustomOp, so
#       forward_oot can't dispatch it, and the op is not built for QAIC.
#    How:
#       fused_moe resolves apply_moe_activation as a module global at call
#       time, so rebind that name to a wrapper that runs GELU_TANH with
#       upstream's PyTorch math (that of GeluAndMul.forward_native,
#       approximate="tanh", computed inline since a CustomOp cannot be built
#       at forward time) and passes every other activation to the original
#       function.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ** 14. File: patch_fused_moe_sum.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm._custom_ops.moe_sum
#    Why:
#       fused_experts_impl (reached when VLLM_QAIC_TRITON_FUSED_MOE is set)
#       reduces the per-expert outputs over the top-k dim with ops.moe_sum,
#       which calls the compiled op torch.ops._moe_C.moe_sum directly. It is
#       not a CustomOp, so forward_oot can't dispatch it, and the op is not
#       built for QAIC.
#    How:
#       fused_moe resolves ops.moe_sum on the _custom_ops module at call time,
#       so rebind that attribute to a wrapper reproducing the CUDA op's
#       semantics (fp32 accumulation, output overwritten, topk_ids/expert_map
#       slot skipping). PyTorch by default; VLLM_QAIC_TRITON_MOE_SUM routes it
#       to upstream's Triton masked_moe_sum kernel.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ** 15. File: patch_fused_moe_align.py **
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   1. vllm._custom_ops.moe_align_block_size
#    Why:
#       fused_experts_impl (reached when VLLM_QAIC_TRITON_FUSED_MOE is set)
#       groups the top-k assignments by expert, padded to the GEMM block size,
#       through moe_align_block_size, whose wrapper calls the compiled op
#       torch.ops._moe_C.moe_align_block_size directly. It is not a CustomOp,
#       so forward_oot can't dispatch it, and the op is not built for QAIC.
#       vLLM has no Triton version of it.
#    How:
#       The wrapper resolves ops.moe_align_block_size on the _custom_ops module
#       at call time, so rebind that attribute to a PyTorch version reproducing
#       the CUDA op's semantics (invalid/unmapped ids dropped, per-expert
#       padding to block_size, sorted_token_ids padded with numel, expert_ids -1
#       past the padded total, num_tokens_post_pad set). The wrapper's buffer
#       allocation and expert_map remapping stay upstream's.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# =================

# NOTE: patch_topk_topp_sampler must stay first; it rebinds a module-level
# function whose consumers (vllm.v1.sample.rejection_sampler, imported by
# patch_rejection_sampler) copy it into their own namespace on import.
import vllm_qaic.patch.patch_topk_topp_sampler  # noqa
import vllm_qaic.patch.patch_config  # noqa
import vllm_qaic.patch.patch_parallel_state  # noqa
import vllm_qaic.patch.patch_utils  # noqa
import vllm_qaic.patch.patch_rejection_sampler  # noqa
import vllm_qaic.patch.patch_graph_pickler  # noqa
import vllm_qaic.patch.patch_mem_utils  # noqa
import vllm_qaic.patch.patch_gemma4_dtype  # noqa
import vllm_qaic.patch.patch_gemma4_router  # noqa
import vllm_qaic.patch.patch_fused_moe_activation  # noqa
import vllm_qaic.patch.patch_fused_moe_sum  # noqa
import vllm_qaic.patch.patch_fused_moe_align  # noqa

import vllm_qaic.envs as qaic_envs

# The numpy slot-mapping patch bypasses vLLM's Triton slot-mapping kernel. When
# the in-tree Triton attention backend is enabled (VLLM_QAIC_ENABLE_TRITON_ATTN),
# the real Triton slot-mapping kernel must stay live to feed slot_mapping into
# triton_reshape_and_cache_flash, so skip this patch in that mode -- and instead
# patch that wrapper, whose CUDA compute-capability query raises on QAIC.
if not qaic_envs.VLLM_QAIC_ENABLE_TRITON_ATTN:
    import vllm_qaic.patch.patch_block_table  # noqa
else:
    import vllm_qaic.patch.patch_reshape_and_cache  # noqa

# Route the unquantized linear/GEMM family to vLLM's Triton GEMM only when the
# opt-in flag is set; otherwise leave the default dispatch_unquantized_gemm path.
if qaic_envs.VLLM_QAIC_TRITON_LINEAR:
    import vllm_qaic.patch.patch_linear  # noqa
