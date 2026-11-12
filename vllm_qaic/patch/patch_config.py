# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/config/cache.py and vllm/vllm/config/vllm.py

from typing import Literal, TypeAlias

import torch
import vllm.config
import vllm.engine.arg_utils
from pydantic import ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass
from vllm.config.cache import CacheConfig, CacheDType
from vllm.config.device import DeviceConfig
from vllm.config.utils import config
from vllm.config.vllm import VllmConfig
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

QaicCacheDType: TypeAlias = CacheDType | Literal["mxint8"]


@config
@dataclass
class QaicCacheConfig(CacheConfig):
    cache_dtype: QaicCacheDType = "auto"
    """Data type for storing the KV cache. Extends the upstream
    ``CacheConfig.cache_dtype`` choices with the QAIC-only ``mxint8`` option."""

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
    device: str | torch.device | None = "auto"
    """Device type for vLLM execution."""
    device_type: str = ""
    """Device type resolved from the current platform."""

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


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class QaicVllmConfig(VllmConfig):
    # Default to QaicDeviceConfig: the CLI builds defaults from this class
    # after the platform sets device_type="qaic".
    device_config: DeviceConfig = Field(default_factory=QaicDeviceConfig)
    """Device configuration."""

    @property
    def max_concurrent_batches(self) -> int:
        # PP requires PP-size concurrent batches to fill the pipeline.
        # Async scheduling requires 2 concurrent batches to overlap.
        pp_size = self.parallel_config.pipeline_parallel_size
        if self.scheduler_config.async_scheduling:
            # QAIC prefill runs its pipeline stages on-device, so pp_size is 1.
            # The kv producer needs max_num_seqs + 1 concurrent batches to
            # fill all the stages.
            override_qaic_config = (self.additional_config or {}).get(
                "override_qaic_config"
            ) or {}
            is_kv_producer = bool(
                self.kv_transfer_config
                and self.kv_transfer_config.kv_role == "kv_producer"
            )
            if is_kv_producer and int(override_qaic_config.get("stages", 1)) > 1:
                return self.scheduler_config.max_num_seqs + 1
            if self.use_v2_model_runner:
                return pp_size + 1
            # V1 Model Runner does not fully support async scheduling with PP.
            if pp_size <= 1:
                return 2
        return pp_size


vllm.config.CacheConfig = QaicCacheConfig
vllm.config.DeviceConfig = QaicDeviceConfig
vllm.config.VllmConfig = QaicVllmConfig

vllm.engine.arg_utils.DeviceConfig = QaicDeviceConfig
vllm.engine.arg_utils.CacheConfig = QaicCacheConfig
vllm.engine.arg_utils.VllmConfig = QaicVllmConfig
