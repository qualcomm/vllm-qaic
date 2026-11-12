# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""
This patch is required for enabling Triton with the QAIC backend.
It imports vLLM's Triton utilities and re-enables ``HAS_TRITON`` without
importing the optional ``triton_kernels`` package. GPT-OSS enables and imports
``triton_kernels`` explicitly through ``enable_qaic_triton_kernels``.
"""

# What's Patched and how it works:
# --------------------------------
#   1. vllm.triton_utils.HAS_TRITON / triton / tl / tldevice
#    Why:
#       Triton is required to run kernels on the QAIC Triton Backend, but
#       upstream's HAS_TRITON check can end up False even when a usable
#       Triton install is present.
#    How:
#       Re-import triton and check for QAIC Triton Backend in
#       triton.backends.backends; if found, re-enable HAS_TRITON and rebind
#       triton/tl/tldevice on vllm.triton_utils to the real modules.
#    Note:
#       Must run before any vllm module does
#       `from vllm.triton_utils import ...`, since that binds triton/tl/
#       HAS_TRITON into the importing module's own namespace at that
#       moment.
import sys

from vllm import triton_utils
from vllm.utils import import_utils

QAIC_TRITON_BACKEND_KEY = "qcom_hexagon_backend"

_TRITON_KERNEL_AVAILABILITY_CONSUMERS = (
    "vllm.model_executor.layers.fused_moe.config",
    "vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe",
    "vllm.model_executor.layers.fused_moe.oracle.mxfp4",
    "vllm.model_executor.layers.quantization.utils.mxfp4_utils",
)


def _qaic_has_triton_kernels_disabled() -> bool:
    """Keep optional Triton kernels hidden during non-GPT-OSS startup."""
    return False


def _qaic_has_triton_kernels() -> bool:
    """Load Triton kernels after the GPT-OSS path explicitly enables them.

    ``import_triton_kernels`` also aliases vLLM's vendored package as the
    top-level ``triton_kernels`` module, which is how the GPT-OSS kernels
    import it.
    """

    if "triton_kernels" not in sys.modules:
        # Retry if vLLM cached an import attempt made before QAIC registered
        # its Triton backend. import_triton_kernels() calls _has_module(), so
        # both caches must be cleared before retrying.
        import_utils._has_module.cache_clear()
        import_utils.import_triton_kernels.cache_clear()
        import_utils.import_triton_kernels()
    return "triton_kernels" in sys.modules


def _refresh_triton_kernel_availability_consumers() -> None:
    """Refresh modules that imported ``has_triton_kernels`` by value.

    Replacing ``import_utils.has_triton_kernels`` does not update modules that
    previously used ``from vllm.utils.import_utils import has_triton_kernels``.
    Only update consumers that have already been imported; importing them here
    would defeat the lazy GPT-OSS setup and could introduce import cycles.
    """
    for module_name in _TRITON_KERNEL_AVAILABILITY_CONSUMERS:
        if module := sys.modules.get(module_name):
            # ModuleType cannot describe this known upstream module attribute.
            module.has_triton_kernels = _qaic_has_triton_kernels  # type: ignore[attr-defined]


def enable_qaic_triton_kernels() -> None:
    """Enable and patch ``triton_kernels`` for the GPT-OSS execution path."""
    import_utils.has_triton_kernels = _qaic_has_triton_kernels
    if not _qaic_has_triton_kernels():
        raise ImportError(
            "QAIC GPT-OSS requires a compatible triton_kernels package, but "
            "the package could not be imported."
        )
    _refresh_triton_kernel_availability_consumers()

    import vllm_qaic.patch.patch_triton_kernels_layout  # noqa: F401
    import vllm_qaic.patch.patch_triton_kernels_opt_flags  # noqa: F401


try:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    if QAIC_TRITON_BACKEND_KEY in triton.backends.backends:
        triton_utils.HAS_TRITON = True
        triton_utils.triton = triton
        triton_utils.tl = tl
        triton_utils.tldevice = tldevice
        import_utils.has_triton_kernels = _qaic_has_triton_kernels_disabled
except Exception:
    triton_utils.HAS_TRITON = False
