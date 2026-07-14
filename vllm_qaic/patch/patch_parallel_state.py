# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/distributed/parallel_state.py
"""This patch is required for AOT, as the torch device must be set to CPU."""

import torch
import vllm.distributed.parallel_state
from torch.distributed import Backend
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    _get_unique_name,
    _register_group,
)
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import suppress_stdout


class QaicGroupCoordinator(GroupCoordinator):
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,  # whether to use device communicator
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        from vllm.platforms import current_platform

        if current_platform.is_aot_inference():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"{current_platform.device_name}:{local_rank}")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            device_comm_cls = resolve_obj_by_qualname(
                current_platform.get_device_communicator_cls()
            )
            self.device_communicator = device_comm_cls(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

        from vllm.platforms import current_platform

        self.use_custom_op_call = False
        # (
        #     current_platform.is_cuda_alike() or current_platform.is_tpu()
        # )

        self.use_cpu_custom_send_recv = False
        # current_platform.is_cpu() and hasattr(
        #     torch.ops._C, "init_shm_manager"
        # )


vllm.distributed.parallel_state.GroupCoordinator = QaicGroupCoordinator
