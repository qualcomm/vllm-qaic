# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.platforms import current_platform


# Register Kv connectors
def register_connector():
    if not current_platform.is_aot_inference():
        return

    KVConnectorFactory.register_connector(
        "QaicConnector",
        "vllm_qaic.distributed.kv_transfer.kv_connector.v1.qaic_connector",
        "QaicConnector",
    )

    KVConnectorFactory.register_connector(
        "QaicLMCacheConnectorV1",
        "vllm_qaic.distributed.kv_transfer.kv_connector.v1.qaic_lmcache_connector",
        "QaicLMCacheConnectorV1",
    )
