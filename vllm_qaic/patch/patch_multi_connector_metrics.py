# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Allow MultiConnector children without Prometheus metrics."""

from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorPromMetrics,
)

_PATCH_ATTR = "_qaic_unknown_connector_metrics_patch_applied"


if not getattr(MultiKVConnectorPromMetrics, _PATCH_ATTR, False):
    _observe = MultiKVConnectorPromMetrics.observe

    def observe(
        self: MultiKVConnectorPromMetrics,
        transfer_stats_data: dict[str, Any],
        engine_idx: int = 0,
    ) -> None:
        supported = {
            connector_id: stats
            for connector_id, stats in transfer_stats_data.items()
            if connector_id in self._prom_metrics
        }
        if supported:
            _observe(self, supported, engine_idx)

    MultiKVConnectorPromMetrics.observe = observe
    setattr(MultiKVConnectorPromMetrics, _PATCH_ATTR, True)
