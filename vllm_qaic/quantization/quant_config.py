# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/model_executor/layers/quantization/base_config.py

from typing import Any

import torch
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm_qaic.utils import QAIC_QUANTIZATION_LIST, QAIC_QUANTIZATION_METHOD


@register_quantization_config(QAIC_QUANTIZATION_METHOD)
class QaicQuantConfig(QuantizationConfig):
    """MxFP6 Quantization Config class for QAIC Backend."""

    def __init__(self, quantize_method: dict[str, Any] | None = None):
        super().__init__()
        self.quantize_method = quantize_method

    def __repr__(self) -> str:
        return "QaicQuantConfig:\n" + super().__repr__()

    def get_name(self) -> str:
        return "qaic_quant"

    @classmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "This function should not be called with QAIC Backend"
        )

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QaicQuantConfig":
        quantize_method = cls.get_from_keys_or(
            config, ["quant_method", "quantize_method"], None
        )
        return cls(quantize_method=quantize_method)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> None:
        return None

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> str | None:
        quantize_method = hf_quant_cfg.get("quant_method", "").lower()
        if (
            quantize_method in QAIC_QUANTIZATION_LIST
            and user_quant == QAIC_QUANTIZATION_METHOD
        ):
            return user_quant
        return None

    def get_scaled_act_names(self) -> list[str]:
        return []
