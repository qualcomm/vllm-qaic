# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/config/cache.py


from typing import Literal

import torch
import vllm.config
import vllm.engine.arg_utils
from pydantic import ConfigDict, field_validator
from pydantic.dataclasses import dataclass
from vllm.config.cache import CacheConfig, CacheDType
from vllm.config.device import DeviceConfig
from vllm.config.utils import config
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

QaicCacheDType = CacheDType | Literal["mxint8"]


@config
@dataclass
class QaicCacheConfig(CacheConfig):
    cache_dtype: QaicCacheDType = "auto"

    @field_validator("cache_dtype", mode="after")
    @classmethod
    def _validate_cache_dtype(cls, cache_dtype: QaicCacheDType) -> QaicCacheDType:
        if cache_dtype.startswith("fp8"):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor."
            )
        return cache_dtype


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class QaicDeviceConfig(DeviceConfig):
    def __post_init__(self):
        if self.device == "auto":
            # Automated device type detection
            from vllm.platforms import current_platform

            self.device_type = current_platform.device_type
            if not self.device_type:
                raise RuntimeError(
                    "Failed to infer device type, please set "
                    "the environment variable `VLLM_LOGGING_LEVEL=DEBUG` "
                    "to turn on verbose logging to help debug the issue."
                )
        else:
            # Device type is assigned explicitly
            if isinstance(self.device, str):
                self.device_type = self.device
            elif isinstance(self.device, torch.device):
                self.device_type = self.device.type

        # Some device types require processing inputs on CPU
        if self.device_type in ["tpu"]:
            self.device = None
        elif self.device_type in ["qaic"]:
            self.device = torch.device("cpu")
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


vllm.config.CacheConfig = QaicCacheConfig
vllm.config.DeviceConfig = QaicDeviceConfig

vllm.engine.arg_utils.DeviceConfig = QaicDeviceConfig
vllm.engine.arg_utils.CacheConfig = QaicCacheConfig
