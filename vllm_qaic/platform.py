# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from vllm_qaic.logger import init_logger
from .platform_base import QaicPlatform as QaicPlatformBase

from .utils import (
    QAIC_KV_CACHE_DTYPE,
    QAIC_QUANTIZATION_METHOD,
)

logger = init_logger(__name__)


class QaicPlatform(QaicPlatformBase):
    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        cls.device_type = "qaic"
        # Adapt the patch here.
        from vllm_qaic import patch  # noqa: F401

        if parser is not None:  # For synchronous vLLM engine
            # disable prefix caching as QAIC backend plugin does not support it
            parser.set_defaults(enable_prefix_caching=False)
            # add mxfp6 to quantization methods list
            quant_action = parser._option_string_actions.get("--quantization")
            if (
                quant_action
                and hasattr(quant_action, "choices")
                and quant_action.choices
                and QAIC_QUANTIZATION_METHOD not in quant_action.choices
            ):
                quant_action.choices.append(QAIC_QUANTIZATION_METHOD)
            kv_cache_dtype_action = parser._option_string_actions.get(
                "--kv-cache-dtype"
            )
            if (
                kv_cache_dtype_action
                and hasattr(kv_cache_dtype_action, "choices")
                and kv_cache_dtype_action.choices
                and QAIC_KV_CACHE_DTYPE not in kv_cache_dtype_action.choices
            ):
                kv_cache_dtype_action.choices.append(QAIC_KV_CACHE_DTYPE)

        from vllm_qaic.quantization.quant_config import (  # noqa: F401
            QaicQuantConfig,
        )
