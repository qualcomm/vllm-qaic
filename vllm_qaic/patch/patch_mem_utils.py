# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""Patch mem_utils to bypass torch.accelerator calls that crash on QAIC.

torch.accelerator.* functions assert that the allocator is a PyTorch
DeviceAllocator. QAIC uses qaicrt, not PyTorch's allocator, so all
torch.accelerator calls must be bypassed.

MemorySnapshot.measure() and memory_profiling() are patched back to the
v15 current_platform-based implementation so QAIC avoids torch.accelerator
allocator calls while still reporting device memory for KV cache sizing.
"""

import gc
import time
from contextlib import contextmanager
from typing import Generator

import psutil
import vllm.utils.mem_utils
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemoryProfilingResult, MemorySnapshot


def _qaic_measure(self) -> None:
    device = self.device_

    # we measure the torch peak memory usage via allocated_bytes,
    # rather than `torch.cuda.memory_reserved()` .
    # After `torch.cuda.reset_peak_memory_stats()`,
    # `torch.cuda.memory_reserved()` will keep growing, and only shrink
    # when we call `torch.cuda.empty_cache()` or OOM happens.
    self.torch_peak = current_platform.memory_stats(device).get(
        "allocated_bytes.all.peak", 0
    )

    self.free_memory, self.total_memory = current_platform.mem_get_info(device)
    shared_sysmem_device_mem_sms = ((8, 7), (11, 0), (12, 1))  # Orin, Thor, Spark
    if (
        current_platform.is_cuda()
        and current_platform.get_device_capability(device.index)
        in shared_sysmem_device_mem_sms
    ):
        # On UMA (Orin, Thor and Spark) platform,
        # where both CPU and GPU rely on system memory,
        # the cudaMemGetInfo function shows the amount of free system memory
        # rather than what’s actually available.
        # In the case,
        # torch.cuda.mem_get_info() only reports "free" memory,
        # which can be lower than what is actually
        # available due to not including cache memory.
        # There’s also a comprehensive reference page
        # that explains how you can compute the proper value yourself.
        # https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/
        # #estimating-total-allocatable-device-memory-on-an-integrated-gpu-device
        self.free_memory = psutil.virtual_memory().available

    self.cuda_memory = self.total_memory - self.free_memory

    # torch.cuda.memory_reserved() is how many bytes
    # PyTorch gets from cuda (by calling cudaMalloc, etc.)
    # this is used to measure the non-torch memory usage
    self.torch_memory = current_platform.memory_reserved(device)

    self.non_torch_memory = self.cuda_memory - self.torch_memory
    self.timestamp = time.time()


@contextmanager
def _qaic_memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    gc.collect()
    current_platform.empty_cache()
    current_platform.reset_peak_memory_stats(baseline_snapshot.device_)

    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        weights_memory=weights_memory,
    )
    result.before_profile.measure()

    yield result

    gc.collect()
    current_platform.empty_cache()
    result.after_profile.measure()

    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    non_torch_memory = result.non_torch_increase
    peak_activation_memory = result.torch_peak_increase
    result.non_kv_cache_memory = (
        non_torch_memory + peak_activation_memory + result.weights_memory
    )


MemorySnapshot.measure = _qaic_measure
vllm.utils.mem_utils.memory_profiling = _qaic_memory_profiling
