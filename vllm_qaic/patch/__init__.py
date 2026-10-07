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
# =================

import vllm_qaic.patch.patch_config  # noqa
import vllm_qaic.patch.patch_parallel_state  # noqa
import vllm_qaic.patch.patch_utils  # noqa
import vllm_qaic.patch.patch_rejection_sampler  # noqa
import vllm_qaic.patch.patch_graph_pickler  # noqa
import vllm_qaic.patch.patch_mem_utils  # noqa
import vllm_qaic.patch.patch_block_table  # noqa
