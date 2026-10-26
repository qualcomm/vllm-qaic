# SPDX-License-Identifier: BSD-3-Clause-Clear
"""QAIC NIXL Push connector with QPC-slot completion accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlPushConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
    from vllm.v1.kv_cache_interface import KVCacheConfig


class QaicNixlPushConnectorWorker(NixlPushConnectorWorker):
    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta) -> None:
        super()._xfer_blocks_for_req(req_id, meta)
        with self._sending_transfers_lock:
            if self._sending_transfers.get(req_id):
                self._recving_metadata[req_id] = meta

    def get_transfer_results(self):
        results = super().get_transfer_results()
        for req_id in results.finished_sending:
            self._recving_metadata.pop(req_id, None)
        return results


class QaicNixlPushConnector(NixlPushConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            assert self.connector_worker is not None
            self.connector_worker.shutdown()
            self.connector_worker = QaicNixlPushConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
