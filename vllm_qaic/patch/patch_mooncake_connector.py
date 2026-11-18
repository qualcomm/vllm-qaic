# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector,
)

MooncakeConnectorWorker = mooncake_connector.MooncakeConnectorWorker
_PATCH_APPLIED_ATTR = "_qaic_mooncake_connector_worker_init_patch_applied"


if not getattr(MooncakeConnectorWorker, _PATCH_APPLIED_ATTR, False):
    _original_init = MooncakeConnectorWorker.__init__

    def _qaic_mooncake_connector_worker_init(self, *args, **kwargs):
        original_current_device_index = (
            mooncake_connector.torch.accelerator.current_device_index
        )
        mooncake_connector.torch.accelerator.current_device_index = lambda: 0
        try:
            return _original_init(self, *args, **kwargs)
        finally:
            mooncake_connector.torch.accelerator.current_device_index = (
                original_current_device_index
            )

    MooncakeConnectorWorker.__init__ = _qaic_mooncake_connector_worker_init
    setattr(MooncakeConnectorWorker, _PATCH_APPLIED_ATTR, True)
