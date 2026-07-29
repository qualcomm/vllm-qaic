# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/models/qwen2_vl.py

"""Patch qwen2_vl._create_qwen2vl_field_factory for the QAIC disagg path.

Applies to both AoT and PyT modes (device-agnostic mm input processing).

In the QAIC E-PD / disagg path, _merge_embeds (vllm.entrypoints.chat_utils)
adds a leading batch dim to image_grid_thw ([N, 3] -> [1, N, 3]) and then
calls _get_mm_fields_config directly, bypassing get_data_parser (which
normally squeezes it in QaicQwen2VLMultiModalDataParser._parse_image_data).
The upstream Qwen field factory then computes a 2-D image_grid_thw.prod(-1)
and MultiModalFieldConfig.flat_from_sizes raises "size_per_item should be a
1-D tensor". The exception is caught and vLLM falls back, but it logs a noisy
traceback on every multimodal request.

Wrap _create_qwen2vl_field_factory so the returned field config squeezes
ndim==3 image_grid_thw / video_grid_thw back to ndim==2 before delegating,
and re-bind the by-name copies imported into qwen3_vl and friends. Mirrors
the fix the monolithic v0.15.0 fork applied inline in qwen2_vl.py.
"""

import importlib

from packaging.version import Version as _Version
from transformers import __version__ as _transformers_version

from vllm.model_executor.models import qwen2_vl as _qwen2_vl

# transformers v5.5.4 image processor makes _reduce_data return ndim==3 grids.
_TRANSFORMERS_NEW_IMAGE_PROCESSOR = _Version(_transformers_version) == _Version("5.5.4")

_original_create_factory = _qwen2_vl._create_qwen2vl_field_factory


def _create_qwen2vl_field_factory_qaic(spatial_merge_size: int):
    inner = _original_create_factory(spatial_merge_size)

    def _qwen2vl_field_config_qaic(hf_inputs):
        if _TRANSFORMERS_NEW_IMAGE_PROCESSOR and hasattr(hf_inputs, "get"):
            patched = None
            for key in ("image_grid_thw", "video_grid_thw"):
                grid = hf_inputs.get(key)
                if grid is not None and getattr(grid, "ndim", None) == 3:
                    if patched is None:
                        patched = dict(hf_inputs)
                    patched[key] = grid.squeeze(0)
            if patched is not None:
                hf_inputs = patched
        return inner(hf_inputs)

    return _qwen2vl_field_config_qaic


_qwen2_vl._create_qwen2vl_field_factory = _create_qwen2vl_field_factory_qaic

# Re-bind the by-name copies imported into other model modules.
for _mod_name in (
    "vllm.model_executor.models.qwen3_vl",
    "vllm.model_executor.models.glm4_1v",
    "vllm.model_executor.models.opencua",
    "vllm.model_executor.models.mimo_v2_omni",
):
    try:
        _mod = importlib.import_module(_mod_name)
    except Exception:
        continue
    if hasattr(_mod, "_create_qwen2vl_field_factory"):
        _mod._create_qwen2vl_field_factory = _create_qwen2vl_field_factory_qaic
