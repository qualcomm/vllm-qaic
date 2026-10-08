# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Adapt the Triton MXFP4 layout helpers to vLLM's expected API.

The ``triton_kernels`` version bundled with Hexagon Triton returns a layout
object directly from ``make_default_matmul_mxfp4_w_{,scale_}layout``.  The
vLLM MXFP4 swizzle code expects newer helpers that return ``(layout, opts)``.

For QAIC bring-up the existing helpers already select ``StridedLayout(-2)``,
which is the conservative layout we want.  Keep that layout and only adapt
the return shape.  CUDA and HIP calls continue to use the upstream helpers
without modification.
"""

import triton

from triton_kernels.tensor_details import layout

QAIC_TRITON_BACKEND_KEY = "qcom_hexagon_backend"

_upstream_make_value_layout = layout.make_default_matmul_mxfp4_w_layout
_upstream_make_scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout


def _qaic_backend_is_active() -> bool:
    return (
        triton.runtime.driver.active.get_current_target().backend
        == QAIC_TRITON_BACKEND_KEY
    )


def _with_layout_options(result):
    """Normalize an old layout result to the newer ``(layout, opts)`` API."""
    if isinstance(result, tuple):
        return result
    return result, {}


def _qaic_make_default_matmul_mxfp4_w_layout(*args, **kwargs):
    result = _upstream_make_value_layout(*args, **kwargs)
    if not _qaic_backend_is_active():
        return result
    return _with_layout_options(result)


def _qaic_make_default_matmul_mxfp4_w_scale_layout(*args, **kwargs):
    result = _upstream_make_scale_layout(*args, **kwargs)
    if not _qaic_backend_is_active():
        return result
    return _with_layout_options(result)


layout.make_default_matmul_mxfp4_w_layout = _qaic_make_default_matmul_mxfp4_w_layout
layout.make_default_matmul_mxfp4_w_scale_layout = (
    _qaic_make_default_matmul_mxfp4_w_scale_layout
)
