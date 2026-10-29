# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import torch
from vllm.model_executor.models import gemma4_mm
from vllm_qaic.logger import init_logger
from contextlib import suppress

logger = init_logger(__name__)

_SUB_CONFIG_NAMES = ("text_config", "vision_config", "audio_config")


def _force_config_dtype(config, dtype: torch.dtype) -> None:
    # transformers keeps the checkpoint dtype (bf16) on each sub-config and
    # AutoModel.from_config() honours it, ignoring the requested --dtype.
    for cfg in (config, *(getattr(config, n, None) for n in _SUB_CONFIG_NAMES)):
        if cfg is None:
            continue
        for attr in ("dtype", "torch_dtype"):
            with suppress(Exception):  # read-only alias on some transformers versions
                setattr(cfg, attr, dtype)


_orig_init = gemma4_mm.Gemma4ForConditionalGeneration.__init__


def _patched_init(self, *, vllm_config, prefix: str = "", **kwargs):
    _force_config_dtype(
        vllm_config.model_config.hf_config, vllm_config.model_config.dtype
    )
    _orig_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)


gemma4_mm.Gemma4ForConditionalGeneration.__init__ = _patched_init
