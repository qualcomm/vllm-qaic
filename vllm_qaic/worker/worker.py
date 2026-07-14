# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/worker/gpu_worker.py
"""A QAIC worker class."""

import gc
import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import vllm.envs as envs
from vllm.config import CUDAGraphMode, VllmConfig, set_current_vllm_config
from vllm.config.compilation import CompilationMode
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    ensure_kv_transfer_shutdown,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.profiler.wrapper import TorchProfilerWrapper
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, format_gib, memory_profiling
from vllm.utils.network_utils import get_distributed_init_method, get_ip, get_open_port
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.utils import compute_iteration_details
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.worker_base import WorkerBase

from vllm_qaic.worker.model_runner import QaicModelRunnerAoT, QaicModelRunnerPyt

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler_output import SchedulerOutput


class QaicWorker(WorkerBase):
    """A worker base class that executes the model on a group of qaic devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.profiler = None

        if self.speculative_config:
            assert self.model_config.hf_config.model_type not in [
                "mlp_speculator",
                "medusa",
                "eagle",
            ], (
                "qaic backend currently doesn't support mlp_speculator or "
                "medusa or eagle models"
            )

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on QAIC, ignore it.")
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        logger.warning("sleep mode is not supported on QAIC, ignore it.")
        pass

    def _configure_thread_parallelism(self):
        # Configure thread parallelism
        #
        # Avoid oversubscription of CPU threads, during multi-instance execution.
        # By default there is no limit, if user set an environment variable
        # VLLM_QAIC_MAX_CPU_THREADS, then number of cpu thread running pytorch
        # sampling on cpu is limited, to avoid over-subscription.
        # The contention is amplified when running in a container where CPU limits
        # can cause throttling.
        default_limit = min(torch.get_num_threads(), 8)
        thread_limit = os.environ.get("VLLM_QAIC_MAX_CPU_THREADS", default_limit)
        if thread_limit:
            logger.warning(
                "Reducing Torch parallelism from %s threads to %s"
                " to avoid unnecessary CPU contention."
                " Set VLLM_QAIC_MAX_CPU_THREADS to tune this value as needed.",
                torch.get_num_threads(),
                thread_limit,
            )
            torch.set_num_threads(int(thread_limit))
            if "OMP_NUM_THREADS" not in os.environ:
                os.environ["OMP_NUM_THREADS"] = str(thread_limit)

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def shutdown(self) -> None:
        # has_kv_transfer_group can be None during interpreter shutdown.
        if ensure_kv_transfer_shutdown is not None:
            ensure_kv_transfer_shutdown()
        if self.profiler is not None:
            self.profiler.shutdown()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()


class QaicWorkerPyt(QaicWorker):
    def __init__(
        self, vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        assert self.model_config.enforce_eager
        self.parallel_config.disable_custom_all_reduce = True

        # configure float32 matmul precision according to vLLM env.
        precision = envs.VLLM_FLOAT32_MATMUL_PRECISION
        torch.set_float32_matmul_precision(precision)

        # Torch profiler. Enabled and configured through env vars:
        if os.environ.get("VLLM_TORCH_PROFILER_DIR"):
            torch_profiler_trace_dir = os.environ["VLLM_TORCH_PROFILER_DIR"]
            worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
            logger.info(
                "Profiling enabled. Traces will be saved to: %s",
                torch_profiler_trace_dir,
            )
            profiler_config = vllm_config.profiler_config
            assert profiler_config.profiler == "torch"
            self.profiler = TorchProfilerWrapper(
                profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
                activities=["CPU"],
            )

        from vllm_qaic.ops import register_qaic_customop

        register_qaic_customop()

    def annotate_profile(self, scheduler_output):
        # adapted from v1/worker/gpu_worker.py
        # add trace annotation so that we can easily distinguish
        # new/cached request numbers in each iteration
        # add trace annotation so that we can easily distinguish
        # context/generation request numbers in each iteration.
        # A context request is a request that has not yet generated any tokens
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        iteration_details = compute_iteration_details(scheduler_output)

        annotation = "".join(
            [
                "execute_context_",
                str(iteration_details.num_ctx_requests),
                "(",
                str(iteration_details.num_ctx_tokens),
                ")_generation_",
                str(iteration_details.num_generation_requests),
                "(",
                str(iteration_details.num_generation_tokens),
                ")",
            ]
        )
        return self.profiler.annotate_context_manager(annotation)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # adapted from gpu_worker.py
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        return

    def init_device(self):
        """Initialize qaic device
        Adapted from gpu_worker.py init_device()"""
        self.device = self.device_config.device

        device = self.device_config.device
        if isinstance(device, torch.device) and device.type == "qaic":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            if (
                self.parallel_config.data_parallel_size > 1
                and self.parallel_config.data_parallel_size_local > 0
                and self.parallel_config.distributed_executor_backend
                not in ["ray", "external_launcher"]
                and self.vllm_config.parallel_config.data_parallel_backend != "ray"
                and self.vllm_config.parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_rank

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert self.local_rank < torch.qaic.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds."
                )
                visible_device_count = (
                    torch.qaic.device_count() if torch.qaic.is_available() else 0
                )
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) "
                    f"must be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
            self.device = torch.device(f"qaic:{self.local_rank}")
            current_platform.set_device(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures QCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after QCCL is initialized
            gc.collect()
            torch.qaic.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (
                self.init_snapshot.total_memory
                * self.cache_config.gpu_memory_utilization
            )
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) is less than "
                    f"desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB)."
                )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")
        set_random_seed(self.model_config.seed)

        self._configure_thread_parallelism()

        # Construct the model runner
        self.model_runner: QaicModelRunnerPyt = QaicModelRunnerPyt(
            self.vllm_config, self.device
        )

    @torch.inference_mode
    def determine_available_memory(self) -> int:
        """Profiles peak memory usage to determine how much can be used for KV cache."""
        GiB = lambda b: b / GiB_bytes
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()

            msg = (
                f"Initial free memory {GiB(self.init_snapshot.free_memory):.2f} "
                f"GiB, reserved {GiB(kv_cache_memory_bytes):.2f} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        torch.qaic.empty_cache()
        torch.qaic.reset_peak_memory_stats()

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), "
            "%.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        gc.collect()

        return int(self.available_kv_cache_memory_bytes)

    def reload_weights(self) -> None:
        # invoke reload_weights from GPUModelRunner
        return self.model_runner.reload_weights()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""
        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        self.model_runner.initialize_kv_cache(kv_cache_config)

    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    def load_model(self) -> None:
        # adapted from gpu_worker.py
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        with self._maybe_get_memory_pool_context(
            tag="weights"
        ) and set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model(eep_scale_up=eep_scale_up)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def compile_or_warm_up_model(self) -> None:
        # Adapted from gpu_worker.py
        warmup_sizes = []

        if self.vllm_config.compilation_config.mode == CompilationMode.VLLM_COMPILE:
            # warm up sizes that are not in cudagraph capture sizes,
            # but users still want to compile for better performance,
            # e.g. for the max-num-batched token size in chunked prefill.
            compile_sizes = self.vllm_config.compilation_config.compile_sizes
            warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []
            cg_capture_sizes: list[int] = []

            if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
                cg_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
                cg_capture_sizes = [] if cg_sizes is None else cg_sizes
                warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]

            compile_ranges = self.vllm_config.compilation_config.get_compile_ranges()
            # For each compile_range, if none of the batch sizes
            # in warmup_sizes or cudagraph_capture_sizes are in the range,
            # add the end of the range to ensure compilation/warmup.
            all_sizes = set(cg_capture_sizes)
            all_sizes.update([x for x in warmup_sizes if isinstance(x, int)])
            for compile_range in compile_ranges:
                if not any(x in compile_range for x in all_sizes):
                    warmup_sizes.append(compile_range.end)

        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            self.model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        kernel_warmup(self)

        cuda_graph_memory_bytes = 0
        # FIXME need better handling of AoT v/s torch compile mode when
        # that needs to be supported
        # if not self.model_config.enforce_eager:
        #     cuda_graph_memory_bytes = self.model_runner.capture_model()

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({format_gib(self.init_snapshot.free_memory)}/"
                f"{format_gib(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{format_gib(self.requested_memory)} GiB). "
                f"Actual usage is {format_gib(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {format_gib(self.peak_activation_memory)} GiB "
                f"for peak activation, {format_gib(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {format_gib(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({format_gib(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{format_gib(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics
            hidden_states, last_hidden_states = self.model_runner._dummy_run(
                num_tokens=max_num_reqs,
                skip_eplb=True,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            )
            if self.model_runner.is_pooling_model:
                self.model_runner._dummy_pooler_run(hidden_states)
            else:
                self.model_runner._dummy_sampler_run(hidden_states=last_hidden_states)

            # Reset the seed to ensure that the random state is not affected by
            # the model initialization and profiling.
            set_random_seed(self.model_config.seed)
        # self.model_runner.warming_up_model()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        # adapted from v1/worker/gpu_worker.py
        with self.annotate_profile(scheduler_output):
            output = self.model_runner.execute_model(scheduler_output)
        return output

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Please set --profiler-config to enable "
                "profiling. Example: "
                "'--profiler-config.profiler=torch --profiler-config.torch_profiler_dir"
                "=YOUR_DIR_PATH_TO_DUMP_TRACE'"
            )
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        self.model_runner._dummy_run(1, uniform_decode=True)


class QaicWorkerAoT(QaicWorker):
    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        # disable sliding window
        self.cache_config.sliding_window = None

        assert num_cpu_blocks == 0
        if not self.cache_config.enable_prefix_caching:
            self.cache_config.num_gpu_blocks = num_gpu_blocks
            # Sanity check: AOT requires exact block count; eager is flexible
            assert num_gpu_blocks == self.scheduler_config.max_num_seqs + 1
            return
        else:
            raise NotImplementedError("prefix caching is not supported on QAIC in V1")

    def init_device(self):
        """Initialize qaic device
        Adapted from gpu_worker.py init_device()"""
        self.device = self.device_config.device

        self._init_qaic_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )

        # Device ID check
        try:
            from qaicrt import Util as qaic_util
        except ImportError:
            import platform
            import sys

            sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
            from qaicrt import Util as qaic_util

        _devices_available = set(qaic_util().getDeviceIds()[1])
        _device_group = (self.vllm_config.additional_config or {}).get("device_group")
        # overwrite device_group with the one from override_qaic_config
        # , if specified
        override_qaic_config = (self.vllm_config.additional_config or {}).get(
            "override_qaic_config"
        )
        if override_qaic_config and "device_group" in override_qaic_config:
            _device_group = override_qaic_config["device_group"]
        if _device_group is not None:
            for device_id in _device_group:
                if device_id not in _devices_available:
                    logger.error("Device id %s not available!!", device_id)

        set_random_seed(self.model_config.seed)

        self._configure_thread_parallelism()

        # Construct the model runner
        self.model_runner: QaicModelRunnerAoT = QaicModelRunnerAoT(
            self.vllm_config, self.device
        )

    def load_model(self) -> None:
        """Load model from QEfficient Transformer library"""
        self.model_runner.load_model()

    def compile_or_warm_up_model(self) -> None:
        self.model_runner._qaic_dummy_run()

    def reload_weights(self) -> None:
        logger.warning("reloading weights is not supported on QAIC in AoT mode.")
        pass

    def determine_available_memory(self) -> int:
        num_gpu_blocks = (
            self.cache_config.num_gpu_blocks_override
            if self.cache_config.num_gpu_blocks_override
            else self.scheduler_config.max_num_seqs
        ) + 1
        # adapted from get_uniform_page_size
        page_sizes = set(
            layer.page_size_bytes for layer in self.get_kv_cache_spec().values()
        )
        assert len(page_sizes) == 1
        page_size = page_sizes.pop()
        return (
            num_gpu_blocks
            * page_size
            * self.model_config.get_num_layers(self.parallel_config)
        )

    def _init_qaic_worker_distributed_environment(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str | None = None,
        local_rank: int = -1,
    ) -> None:
        """Qaic uses tensor parallelism using device-group argument

        vLLM still needs the environment inited when TP/PP > 1
        """
        # Spawning a large number of workers may lead to race condition
        # in using the same port in distributed_init_method
        # Retry logic is similar to https://github.com/vllm-project/vllm/pull/20151
        max_retries = 5
        _distributed_init_method = distributed_init_method
        for i in range(max_retries):
            try:
                init_distributed_environment(
                    world_size=1,
                    rank=rank,
                    local_rank=local_rank,
                    distributed_init_method=_distributed_init_method,
                    backend=current_platform.dist_backend,
                )
                self.distributed_init_method = _distributed_init_method
                break
            except torch.distributed.DistNetworkError as e:
                if "EADDRINUSE" in str(e):
                    if i < max_retries - 1:
                        logger.warning(
                            "Address %s already in use. Retrying with a new port.",
                            _distributed_init_method,
                        )
                        _distributed_init_method = get_distributed_init_method(
                            get_ip(), get_open_port()
                        )
                        continue
                    else:
                        logger.error(
                            "Failed to initialize distributed environment"
                            " after %d attempts: %s",
                            i + 1,
                            e,
                        )
                raise e
        ensure_model_parallel_initialized(
            1,
            1,
        )
        ensure_kv_transfer_initialized(vllm_config)

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        return self.model_runner.execute_model(scheduler_output)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        logger.warning("profile is not supported with QAIC AoT mode.")
        pass
