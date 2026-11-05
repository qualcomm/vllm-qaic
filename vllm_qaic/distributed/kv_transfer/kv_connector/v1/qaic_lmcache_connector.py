# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from typing import TYPE_CHECKING

from lmcache.integration.vllm.qaic_vllm_v1_adapter import QaicLMCacheConnectorV1Impl

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
    LMCacheKVEvents,
)
from vllm_qaic.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class QaicLMCacheConnectorV1(LMCacheConnectorV1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # Bypass LMCacheConnectorV1.__init__ to avoid creating LMCacheConnectorV1Impl
        KVConnectorBase_V1.__init__(
            self, vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        # Create the QAIC-specific engine instead
        self._lmcache_engine = QaicLMCacheConnectorV1Impl(vllm_config, role, self)

        self._kv_cache_events: LMCacheKVEvents | None = None

    def wait_for_save(self, **kwargs):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        Overrides parent to accept **kwargs, required for async scheduling.
        """
        self._lmcache_engine.wait_for_save(**kwargs)
