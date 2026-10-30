# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Lightweight registration for QAIC GPT-OSS MXFP4 quantization.

This module is imported during platform pre-registration for every model, so
it must not import the GPT-OSS MoE implementation or ``triton_kernels``. The
implementation is loaded only after vLLM selects this config for a GPT-OSS
checkpoint and asks for a layer's quantization method.
"""

import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.platforms import current_platform

from vllm_qaic.utils import QAIC_GPT_OSS_MXFP4_METHOD


@register_quantization_config(QAIC_GPT_OSS_MXFP4_METHOD)
class QaicGptOssMxfp4Config(QuantizationConfig):
    """Lazy QAIC eager-mode MXFP4 config for GPT-OSS checkpoints."""

    def get_name(self) -> str:
        return "gpt_oss_mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "QaicGptOssMxfp4Config":
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None, **kwargs
    ) -> str | None:
        if current_platform.is_aot_inference():
            return None
        if not (
            isinstance(hf_quant_cfg, dict)
            and hf_quant_cfg.get("quant_method") in ("mxfp4", "gpt_oss_mxfp4")
        ):
            return None
        if getattr(hf_config, "model_type", None) != "gpt_oss":
            return None
        return "gpt_oss_mxfp4"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm_qaic.patch.patch_qaic_triton_import import (
            enable_qaic_triton_kernels,
        )

        enable_qaic_triton_kernels()

        from vllm_qaic.quantization.qaic_gpt_oss_mxfp4_config import (
            QaicGptOssMxfp4Config as QaicGptOssMxfp4Implementation,
        )

        return QaicGptOssMxfp4Implementation().get_quant_method(layer, prefix)
