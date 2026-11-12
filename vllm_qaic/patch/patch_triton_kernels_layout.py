# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Adapt Triton MXFP4 layout handling for the QAIC backend.

The ``triton_kernels`` version bundled with Hexagon Triton returns a layout
object directly from ``make_default_matmul_mxfp4_w_{,scale_}layout``.  The
vLLM MXFP4 swizzle code expects newer helpers that return ``(layout, opts)``.

For QAIC bring-up the existing helpers already select ``StridedLayout(-2)``,
which is the conservative layout we want.  Keep that layout and only adapt
the return shape.

This module is imported only by the QAIC plugin.  Do not query Triton's active
target here: platform initialization can import the patch before the active
driver target is finalized, which would leave the old return API in place.

Hexagon Triton's layout conversion also performs a canonical-layout round trip
when the source and destination layouts are identical.  That round trip issues
device-side uint8-to-uint8 ``copy_`` operations, which the QAIC PyTorch backend
does not support.  Skip only this semantic no-op; conversions between different
layouts continue through the upstream implementation.
"""

from triton_kernels import tensor
from triton_kernels.tensor_details import layout

_upstream_make_value_layout = layout.make_default_matmul_mxfp4_w_layout
_upstream_make_scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout
_upstream_convert_layout = tensor.convert_layout


def _with_layout_options(result):
    """Normalize an old layout result to the newer ``(layout, opts)`` API."""
    if isinstance(result, tuple):
        return result
    return result, {}


def _qaic_make_default_matmul_mxfp4_w_layout(*args, **kwargs):
    result = _upstream_make_value_layout(*args, **kwargs)
    return _with_layout_options(result)


def _qaic_make_default_matmul_mxfp4_w_scale_layout(*args, **kwargs):
    result = _upstream_make_scale_layout(*args, **kwargs)
    return _with_layout_options(result)


def _qaic_convert_layout(value, target_layout, **layout_transformation_kwargs):
    """Avoid an unsupported QAIC uint8 copy for an unchanged layout."""
    if value.storage.layout == target_layout and not layout_transformation_kwargs:
        return value
    return _upstream_convert_layout(
        value,
        target_layout,
        **layout_transformation_kwargs,
    )


layout.make_default_matmul_mxfp4_w_layout = _qaic_make_default_matmul_mxfp4_w_layout
layout.make_default_matmul_mxfp4_w_scale_layout = (
    _qaic_make_default_matmul_mxfp4_w_scale_layout
)
tensor.convert_layout = _qaic_convert_layout
