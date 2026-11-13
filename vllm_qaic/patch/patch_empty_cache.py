# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Redirect torch.accelerator.empty_cache to the eager QAIC allocator.

The generic API reaches PyTorch's CachingDeviceAllocator before vLLM's
platform abstraction can intervene. QAIC uses qaicrt, so model code calling
this API must use torch.qaic.empty_cache instead.
"""

import torch

_EMPTY_CACHE_PATCHED_ATTR = "_qaic_empty_cache_patch_applied"
_ORIGINAL_EMPTY_CACHE_ATTR = "_qaic_original_empty_cache"


def _patch_accelerator_empty_cache() -> None:
    """Route generic accelerator cache cleanup through torch-qaic.

    QAIC uses qaicrt rather than PyTorch's CachingDeviceAllocator. Calling the
    generic implementation therefore raises an internal allocator assertion.
    The torch-qaic API has the same no-argument cleanup contract and owns the
    allocator that backs QAIC tensors.
    """
    accelerator = torch.accelerator
    if getattr(accelerator, _EMPTY_CACHE_PATCHED_ATTR, False):
        return

    qaic = getattr(torch, "qaic", None)
    qaic_empty_cache = getattr(qaic, "empty_cache", None)
    if not callable(qaic_empty_cache):
        # AOT mode has no torch-qaic device allocator and does not execute eager
        # QAIC model code, so leave the generic API untouched.
        return

    setattr(accelerator, _ORIGINAL_EMPTY_CACHE_ATTR, accelerator.empty_cache)

    def _qaic_empty_cache() -> None:
        qaic_empty_cache()

    accelerator.empty_cache = _qaic_empty_cache
    setattr(accelerator, _EMPTY_CACHE_PATCHED_ATTR, True)


_patch_accelerator_empty_cache()
