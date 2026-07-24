# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
from functools import cached_property

from vllm.v1.executor.uniproc_executor import UniProcExecutor


class QaicUniProcExecutor(UniProcExecutor):
    @cached_property
    def max_concurrent_batches(self) -> int:
        additional_config = self.vllm_config.additional_config or {}
        override_qaic_config = additional_config.get("override_qaic_config") or {}
        stages = int(override_qaic_config.get("stages", 1))
        if self.scheduler_config.async_scheduling and stages >= 1:
            return self.scheduler_config.max_num_seqs + 1
        return 2 if self.scheduler_config.async_scheduling else 1
