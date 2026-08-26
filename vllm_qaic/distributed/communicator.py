# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

# Customization point for distributed calls. Most defaults redispatch to
# torch.dist.* equivalents
# which uses QCCL native/via-gloo implementation for qaic platform.

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


# See `DeviceCommunicatorBase` for available functions that can be overridden.
# By default, all distributed ops are implemented using torch.distributed
# impl (i.e., qccl).
class QAicCommunicator(DeviceCommunicatorBase):
    pass
