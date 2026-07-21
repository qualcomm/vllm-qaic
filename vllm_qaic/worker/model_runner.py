# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/worker/gpu_model_runner.py

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from queue import Queue
from threading import Event
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.forward_context import get_forward_context, set_forward_context
from vllm_qaic.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, SupportedTask
from vllm.utils.import_utils import PlaceholderModule
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ECConnectorOutput,
    KVConnectorOutput,
    ModelRunnerOutput,
    SamplerOutput,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import InputBatch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner, AsyncGPUPoolingModelRunnerOutput

try:
    import torch_qaic.profile as qaic_profile
    import torch_qaic.qaic as qaic
except ImportError:
    qaic = PlaceholderModule("qaic")
    qaic_profile = PlaceholderModule("qaic_profile")

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.spec_decode.qaic_draft_model import QaicDraftModelProposer

logger = init_logger(__name__)


class QaicExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None.

    Additional field are added on top of ExecuteModelState for Qaic Async scheduling.
    """

    scheduler_output: SchedulerOutput
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: QaicSpecDecodeCommonAttnMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None
    cudagraph_stats: CUDAGraphStat | None
    slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None
    hidden_states_decode: np.ndarray | None
    hidden_states_prefill: np.ndarray | None
    pending_prefill_exec_queue: Queue | None
    num_decodes_executed: (
        int | None
    )  # for kv_consumer this also include the last prompt token
    discard_request_mask_np: np.ndarray | None  # used to calculate partial prefills


class QaicAsyncPoolingModelRunnerOutput(AsyncModelRunnerOutput):
    """
    Async output wrapper for QAIC pooling / embedding models.

    Defers QAIC hardware wait, output-buffer processing, and pooling until
    get_output() is called, enabling the scheduler to prepare the next batch
    while the current inference is still in flight.

    IMPORTANT: execute_model() returns this object directly (not None) because
    the engine core's step_with_batch_queue() uses exec_future directly for
    pooling models and never calls sample_tokens().  Returning None would cause
    a RuntimeError in the engine core.

    The async pattern mirrors _run_decode:
      execute_model() → _run_encode() submits inference, returns Queue
                      → returns QaicAsyncPoolingModelRunnerOutput directly
      get_output()    → complete_all_inf(Queue) → process_encode_output_buffer()
                        → _pool() → ModelRunnerOutput
    """

    def __init__(
        self,
        model_runner: QaicModelRunnerAoT,
        pending_prefill_exec_queue: Queue,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput,
    ):
        self._model_runner = model_runner
        self._pending_prefill_exec_queue = pending_prefill_exec_queue
        self._num_scheduled_tokens = num_scheduled_tokens
        self._num_scheduled_tokens_np = num_scheduled_tokens_np
        self._kv_connector_output = kv_connector_output

    def get_output(self) -> ModelRunnerOutput:
        self._model_runner.complete_all_inf(self._pending_prefill_exec_queue, 0)

        hidden_states = self._model_runner.model._process_encode_output(
            self._model_runner.model._encode_prefill_cum_sum,
            self._model_runner.model._encode_prompt_lens,
            self._model_runner.model._encode_output_key,
        )

        result = self._model_runner._pool(
            hidden_states,
            self._num_scheduled_tokens,
            self._num_scheduled_tokens_np,
            self._kv_connector_output,
        )

        self._model_runner.model_runner_output_event.set()

        return result


class QaicAsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    """
    Async output wrapper for QAIC. Defers hardware wait, hidden-state
    computation, sampling and bookkeeping until get_output() is called.
    """

    def __init__(
        self,
        model_runner: QaicModelRunnerAoT,
        state: QaicExecuteModelState,
        grammar_output: GrammarOutput | None,
        kv_connector_output: KVConnectorOutput,
        kv_connector_metadata=None,
        input_batch_req_ids=None,
        input_batch_req_id_to_index=None,
    ):
        self._model_runner = model_runner
        self._state = state
        self._grammar_output = grammar_output

        # For Disagg
        self._kv_connector_output = kv_connector_output
        self._kv_connector_metadata = kv_connector_metadata
        self._input_batch_req_ids = input_batch_req_ids
        self._input_batch_req_id_to_index = input_batch_req_id_to_index

    def get_output(self) -> ModelRunnerOutput:
        state = self._state
        mr = self._model_runner

        # Disagg prefill: unblock the next batch early in sample_tokens() and
        # return a dummy sampler output since sampled tokens are not needed.
        if mr.is_async_kv_producer:
            # 1. Wait for inference completion
            mr.complete_all_inf(
                state.pending_prefill_exec_queue, state.num_decodes_executed
            )
            # 2. Create dummy sampler output
            sampler_output = mr._make_sampler_output(
                torch.zeros((len(self._input_batch_req_ids), 1), dtype=torch.int64)
            )
            # 3. Discard samped tokens for partial prefills
            kv_connector_output = self._kv_connector_output
            discard_sampled_tokens_req_indices = np.nonzero(
                state.discard_request_mask_np
            )[0]
            valid_sampled_token_ids = sampler_output.sampled_token_ids.tolist()
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()
            # 4. Save KV Cache to connector
            if has_kv_transfer_group():
                kv_connector = get_kv_transfer_group()
                # Partial or full KV are handled by the connector metadata
                QaicModelRunnerAoT._save_kv_and_finalize_connector_output(
                    kv_connector,
                    kv_connector_output,
                    state.scheduler_output.finished_req_ids,
                    kv_cache_info=mr.kv_cache_info,  # type: ignore[has-type]
                    connector_metadata=self._kv_connector_metadata,
                )
            if (
                mr.model.on_device_sampling_en
                and state.scheduler_output.finished_req_ids
            ):
                mr._ods_frequency_penalty_warned_req_ids.difference_update(
                    state.scheduler_output.finished_req_ids
                )
            return ModelRunnerOutput(
                req_ids=self._input_batch_req_ids,
                req_id_to_index=self._input_batch_req_id_to_index,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=kv_connector_output,
                num_nans_in_logits=None,
            )

        # 1. Wait for inference completiong
        mr.complete_all_inf(
            state.pending_prefill_exec_queue, state.num_decodes_executed
        )
        # 2. Compute hidden states + logits
        if mr.model.on_device_sampling_en:
            hidden_states = torch.empty((0, 0), dtype=torch.float32)
            logits = torch.zeros((mr.input_batch.num_reqs, 1), dtype=torch.float32)
        else:
            hidden_states, logits = mr._compute_hidden_states_and_logits(
                state.hidden_states_decode,
                state.hidden_states_prefill,
                state.num_decodes_executed,
                spec_decode_metadata=state.spec_decode_metadata,
            )
        # Apply structured output bitmasks if present.
        if self._grammar_output is not None and not mr.model.on_device_sampling_en:
            apply_grammar_bitmask(
                state.scheduler_output, self._grammar_output, mr.input_batch, logits
            )

        sampling_metadata = None
        if mr.model.on_device_sampling_en:
            sampling_metadata = mr.input_batch.sampling_metadata
            if sampling_metadata is not None:
                nondefault_slots = torch.nonzero(
                    sampling_metadata.frequency_penalties != 0.0,
                    as_tuple=False,
                ).flatten()
                req_ids = mr.input_batch.req_ids
                for slot_index in nondefault_slots.tolist():
                    req_id = req_ids[slot_index]
                    if req_id in mr._ods_frequency_penalty_warned_req_ids:
                        continue
                    logger.warning(
                        "Request %s specified non-default frequency_penalty, "
                        "but frequency_penalty is ignored when on-device "
                        "sampling is enabled.",
                        req_id,
                    )
                    mr._ods_frequency_penalty_warned_req_ids.add(req_id)

        if mr.model.on_device_sampling_en:
            if state.spec_decode_metadata is not None:
                raise ValueError(
                    "On-device sampling requires speculative decoding to be "
                    "disabled; spec_decode_metadata must be None."
                )
            mr.input_batch.update_async_output_token_ids()
            sampler_output = mr.model.sample(logits, sampling_metadata)
        else:
            sampler_output = mr._sample(logits, state.spec_decode_metadata)
        mr.input_batch.prev_sampled_token_ids = None

        # 3. Book keep to update input batch
        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = mr._bookkeeping_sync(
            state.scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            state.scheduler_output.total_num_scheduled_tokens,
        )
        if mr.model.on_device_sampling_en:
            num_nans_in_logits = None
        if (
            mr.model.on_device_sampling_en
            and state.scheduler_output.finished_req_ids
        ):
            mr._ods_frequency_penalty_warned_req_ids.difference_update(
                state.scheduler_output.finished_req_ids
            )

        # 4. Set event to unblock future batches
        mr.execute_model_state = None
        mr.model_runner_output_event.set()

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=self._kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        max_gen_len = sampled_token_ids.shape[-1]

        # Release the device tensors once the copy has completed.
        if max_gen_len == 1:
            valid_sampled_token_ids = sampled_token_ids.tolist()
            for i in invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if logprobs_tensors is not None:
                logprobs_lists = logprobs_tensors.tolists()
        else:
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                sampled_token_ids,
                mr.input_batch.vocab_size,
                invalid_req_indices,
                logprobs_tensors=logprobs_tensors,
            )

        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists

        return output


@dataclass
class QaicSpecDecodeCommonAttnMetadata:
    """Minimal metadata needed for GPU-like drafter fit checks on QAIC."""

    max_seq_len: int


class QaicModelRunnerPyt(GPUModelRunner):
    """PyTorch eager-mode model runner for QAIC.

    Wraps GPUModelRunner with QAIC-specific device setup (qaic.Event /
    qaic.Stream), warm-up compilation trigger, and optional per-iteration
    profiling via torch_qaic.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.profiling_iter = 0
        with _torch_qaic_wrapper():
            super().__init__(vllm_config, device)

    def _init_device_properties(self) -> None:
        self.device_properties = torch.qaic.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ):
        # Propagate updated req_ids (post-reorder) to attention metadata
        # builders that support per-sequence KV caching.
        req_ids = list(self.input_batch.req_ids)
        for kv_cache_gid_groups in self.attn_groups:
            for attn_group in kv_cache_gid_groups:
                builder = attn_group.get_metadata_builder()
                if hasattr(builder, "update_req_ids"):
                    builder.update_req_ids(req_ids)
        return super()._prepare_inputs(scheduler_output, num_scheduled_tokens)

    def _sync_device(self) -> None:
        if not isinstance(qaic, PlaceholderModule):
            qaic.synchronize()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        with optional_qaic_profiling(
            profiling_dir=getattr(envs, "VLLM_TORCH_PROFILER_DIR", None),
            profiling_wrapper=qaic_profile.ProfileForwardWithSampling,
            model=self.model,  # type: ignore[has-type]
            n_samples=10,
            skip_first_n=0,
            device_name=str(self.device),
            profiling_iter=self.profiling_iter,
        ) as self.model:  # type: ignore[has-type]
            output = super().execute_model(
                scheduler_output=scheduler_output,
                intermediate_tensors=intermediate_tensors,
            )
        self.profiling_iter += 1
        return output


class QaicModelRunnerAoT(GPUModelRunner):
    """AoT (Ahead-of-Time) compiled model runner for QAIC.

    Runs models that have been pre-compiled for the QAIC accelerator.
    Manages separate decode / prefill execution objects, async scheduling,
    and disaggregated-serving KV transfer.
    """

    # Widen drafter type to include the QAIC-specific proposer. The parent
    # GPUModelRunner declares it without QaicDraftModelProposer; assigning
    # a QaicDraftModelProposer in load_model() would otherwise taint every
    # subsequent isinstance check and method call with [has-type] errors.
    # QaicDraftModelProposer is imported only under TYPE_CHECKING to avoid
    # a runtime circular/conditional import (it is loaded dynamically).
    drafter: (  # type: ignore[override]
        NgramProposer
        | SuffixDecodingProposer
        | EagleProposer
        | DraftModelProposer
        | QaicDraftModelProposer
        | None
    )

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.profiling_iter = 0
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        assert device == torch.device("cpu")
        # --- Disaggregated serving flags (must precede drafter gating) ---
        self.is_kv_producer: bool = bool(
            vllm_config.kv_transfer_config
            and vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = bool(
            vllm_config.kv_transfer_config
            and vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        self.is_async_kv_producer: bool = (
            self.is_kv_producer and self.use_async_scheduling
        )
        # KV producer (prefill node) must never run the drafter; clear
        # anything the parent __init__ installed for ngram/suffix SpD.
        if self.is_kv_producer:
            self.drafter = None
        if self.speculative_config and self.speculative_config.method not in (
            "ngram",
            "suffix",
            "draft_model",
        ):
            raise ValueError(
                "Speculative decoding method "
                f"{self.speculative_config.method} "
                "is not yet supported on QAIC backend."
            )
        if (
            self.speculative_config
            and self.speculative_config.uses_draft_model()
            and not self.is_kv_producer
        ):
            # The parent's __init__ creates a GPU DraftModelProposer; override with
            # the QAIC-specific proposer. Model weights are loaded in load_model().
            from vllm.config.utils import replace as config_replace  # noqa: PLC0415

            from vllm_qaic.spec_decode.qaic_draft_model import (  # noqa: PLC0415
                QaicDraftModelProposer,
            )

            # Build a VllmConfig for the draft model using the target's config
            # as a base, overriding with draft-specific model/parallel config.
            spec_cfg = self.speculative_config
            draft_vllm_config = config_replace(
                self.vllm_config,
                model_config=spec_cfg.draft_model_config,
                quant_config=None,
            )
            _draft_override = (self.vllm_config.additional_config or {}).get(
                "draft_override_qaic_config"
            )
            if _draft_override is not None:
                # Give the draft model its own additional_config dict; cannot mutate
                # the shared parent dict.
                draft_vllm_config = config_replace(
                    draft_vllm_config,
                    additional_config={
                        **(draft_vllm_config.additional_config or {}),
                        "override_qaic_config": _draft_override,
                    },
                )
            self.drafter = QaicDraftModelProposer(draft_vllm_config)
        # Extract configuration params
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.execute_model_state: QaicExecuteModelState | None = None
        self.kv_caches: list[list] = [
            [] for _ in range(vllm_config.scheduler_config.max_num_seqs)
        ]
        self.num_decode_tokens = 0
        self.max_decode_tokens = 1 + self.num_spec_tokens
        # Variable-K decode specializations: for ngram/suffix we compile two
        # kernels (K=0 and K=max_k) and select the cheapest one each step.
        _method = self.speculative_config.method if self.speculative_config else None
        self.decode_ks: list[int] = (
            [0, self.num_spec_tokens]
            if _method in ("ngram", "suffix") and self.max_decode_tokens > 1
            else [self.num_spec_tokens]
        )
        # active_k is updated each step; defaults to max K until first dispatch.
        self.active_k: int = self.decode_ks[-1]
        # spec dec vars
        self.signal_num_scheduled_tokens: np.ndarray | None = None
        self.arange_mdt: np.ndarray | None = None
        self.tile_arange_mdts: list[np.ndarray] | None = None
        if self.max_decode_tokens > 1:
            self.arange_mdt = np.arange(self.max_decode_tokens)
            self.signal_num_scheduled_tokens = np.zeros(
                (self.max_num_reqs,), dtype=np.int32
            )
            self.tile_arange_mdts = [
                np.tile(self.arange_mdt, bsz) for bsz in range(1, self.max_num_reqs + 1)
            ]
        # Latent state: updated by _prepare_qaic_inputs on every step that has
        # max_decode_tokens > 1 (i.e. when SpD is active). Reads before the
        # first _prepare_qaic_inputs call return 0.
        self.spec_decode_max_seq_len = 0
        self.model_runner_output_event: Event = Event()
        # Correctness invariant: in the targeted vLLM engine version,
        # scheduler_output.finished_req_ids must be exhaustive for request
        # teardown (normal completion, abort, and error). If a future version
        # misses a teardown path, warned request IDs can become stale in this
        # set and accumulate without bound.
        self._ods_frequency_penalty_warned_req_ids: set[str] = set()
        # Post-process tensors
        self._postprocess_tensors()

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        # AoT mode: QAIC inference is already synchronized via complete_all_inf()
        # before _pool() is called.  No CUDA synchronization is needed or possible.
        pass

    def _postprocess_tensors(self) -> None:
        # Cast below tensors from `int32` -> `int64`; AOT-mode only.
        self.input_ids_cpu = self.input_ids.cpu.to(torch.int64)
        # Alias so that non-SpD code paths referencing self.input_ids.cpu still work
        self.input_ids.cpu = self.input_ids_cpu
        # transform empty values of 0 to -1 to represent padding values in SpD
        token_ids_cpu_tensor = self.input_batch.token_ids_cpu_tensor.to(torch.int64) - 1
        if self.max_decode_tokens > 1:
            # Enlarge token ids table so that spec dec has enough padded tokens
            # for position `max_model_len-1`
            zeros = (
                torch.zeros(
                    (self.max_num_reqs, self.num_spec_tokens),
                    device="cpu",
                    dtype=torch.int64,
                    pin_memory=False,
                )
                - 1
            )
            token_ids_cpu_tensor = torch.cat([token_ids_cpu_tensor, zeros], dim=1)
        self.input_batch.token_ids_cpu_tensor = token_ids_cpu_tensor

        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy()
        self.input_batch.block_table[0].block_table.cpu = self.input_batch.block_table[
            0
        ].block_table.cpu.to(torch.int64)
        self.input_batch.block_table[0].block_table.np = self.input_batch.block_table[
            0
        ].block_table.cpu.numpy()
        self.positions_np = self.positions.numpy()
        if self.uses_mrope:
            self.mrope_positions_np = self.mrope_positions.np

    def _may_reorder_batch(self, scheduler_output: SchedulerOutput) -> None:
        _, num_decodes, num_decode_tokens = reorder_batch_to_split_decodes_and_prefills(
            self.input_batch, scheduler_output
        )
        self.num_decodes = num_decodes
        self.num_decode_tokens = num_decode_tokens

    def _prepare_input_ids(
        self,
        scheduler_output: SchedulerOutput,
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens need
        to be copied into the corresponding slots into input_ids."""
        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids.to(torch.int64)

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.cpu[:num_commmon_tokens].copy_(
                prev_sampled_token_ids[:num_commmon_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.cpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        self.input_ids.cpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=prev_sampled_token_ids[prev_common_req_indices_tensor, 0],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:  # type: ignore
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)  # type: ignore[has-type]
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)  # type: ignore

        self.input_ids.cpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _determine_active_k(self, scheduler_output: SchedulerOutput) -> int:
        """Return K to use for this decode step.

        With multi-spec (``decode_ks = [0, max_k]``): select K=0 when no
        decode request has proposals for this step; select max_k otherwise.
        With single-spec: always returns the sole K (no-op).
        """
        if len(self.decode_ks) <= 1 or self.num_decodes == 0:
            return self.decode_ks[-1]
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        if spec_tokens and any(len(v) > 0 for v in spec_tokens.values()):
            return self.decode_ks[-1]  # proposals exist → full SpD kernel
        return 0  # no proposals → cheap fallback kernel
    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        if not self.model.is_qaic_pooler and self.model.task not in (
            "score",
            "classify",
        ):  # for CPU based embed pooling, use GPU model runner's _pool
            # Force synchronous path: AsyncGPUPoolingModelRunnerOutput requires
            # CUDA streams which are not available on QAIC hardware.  The QAIC
            # async scheduling for pooling is handled at a higher level by
            # QaicAsyncPoolingModelRunnerOutput, so _pool() itself must always
            # be synchronous.
            orig_async = self.use_async_scheduling
            self.use_async_scheduling = False
            try:
                result = super()._pool(
                    hidden_states,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    kv_connector_output,
                )
            finally:
                self.use_async_scheduling = orig_async
            return result

        # For QAIC based pooler OR score/classify tasks (CPU or QAIC),
        # model outputs are already pooled — wrap directly in ModelRunnerOutput.
        num_reqs_qaicpooler = self.input_batch.num_reqs
        assert num_reqs_qaicpooler == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )
        hidden_states_qaicpooler = hidden_states[:num_reqs_qaicpooler]
        seq_lens_qaicpooler = self.seq_lens.cpu[:num_reqs_qaicpooler]

        pooling_metadata_qaicpooler = self.input_batch.get_pooling_metadata()
        pooling_metadata_qaicpooler.build_pooling_cursor(
            num_scheduled_tokens_np,
            seq_lens_qaicpooler,
            device=hidden_states_qaicpooler.device,
        )

        finished_mask_qaicpooler = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(
                seq_lens_qaicpooler, pooling_metadata_qaicpooler.prompt_lens
            )
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=kv_connector_output,
        )

        if not any(finished_mask_qaicpooler):
            model_runner_output.pooler_output = [None] * num_reqs_qaicpooler
            return model_runner_output

        model_runner_output.pooler_output = [
            out if include else None
            for out, include in zip(
                hidden_states_qaicpooler, finished_mask_qaicpooler, strict=False
            )
        ]

        return model_runner_output

    def _prepare_qaic_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        """
        :return: SpecDecodeMetadata
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Save actual counts for filtering logits later
        if self.max_decode_tokens > 1:
            # Keep original scheduler token counts for all active requests.
            # Decode rows are later used for QAIC decode-logit masking, and this
            # full vector is also used to compute GPU-like max_seq_len.
            self.signal_num_scheduled_tokens[:num_reqs] = num_scheduled_tokens[  # type: ignore[index]
                :num_reqs
            ]
            # Pad decode requests to active_k+1 tokens (1 for K=0 fallback,
            # max_decode_tokens for the full SpD kernel).
            num_scheduled_tokens[: self.num_decodes] = self.active_k + 1
            total_num_scheduled_tokens = np.sum(num_scheduled_tokens)

        # Get request indices.
        # num_scheduled_tokens, [2, 5, 3]
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
        self.cu_num_tokens = cu_num_tokens
        total_tokens = cu_num_tokens[-1]
        self.positions_np[:total_tokens] = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[:total_tokens]
        )
        positions_np = self.positions_np[:total_tokens]
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        self.batch_indices = (
            self.input_batch.block_table[0].get_numpy_array()[:num_reqs, 0] - 1
        )

        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)
        # Compute max_seq_len if SpD
        if self.max_decode_tokens > 1:
            # value matches GPU SpecDecodeCommonAttnMetadata.max_seq_len
            self.spec_decode_max_seq_len = int(
                np.max(
                    self.input_batch.num_computed_tokens_cpu[:num_reqs]
                    + self.signal_num_scheduled_tokens[:num_reqs]  # type: ignore[index]
                )
            )
        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        self.discard_request_mask.np[:num_reqs] = (
            self.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np
        )
        if self.input_batch.prev_sampled_token_ids is not None:
            self._prepare_input_ids(
                scheduler_output, num_scheduled_tokens, cu_num_tokens
            )

        spec_decode_metadata = None
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if use_spec_decode:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
        return spec_decode_metadata

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        num_sampled_tokens = num_draft_tokens + 1
        # Use a temporary buffer for the batched arange output to avoid
        # corrupting self.arange_np (which _get_cumsum_and_arange reads from).
        arange_buf = np.empty_like(self.arange_np)
        cu_num_sampled_tokens = self._get_cumsum_and_arange(
            num_sampled_tokens, arange_buf, cumsum_dtype=np.int32
        )
        bonus_logits_indices = cu_num_sampled_tokens - 1
        cu_num_draft_tokens = self._get_cumsum_and_arange(
            num_draft_tokens, arange_buf, cumsum_dtype=np.int32
        )
        total_draft = int(cu_num_draft_tokens[-1]) if cu_num_draft_tokens.size else 0
        arange = arange_buf[:total_draft]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        target_logits_indices += arange

        # Compute draft_token_indices in the PADDED space.
        # These are used to extract draft_token_ids from the padded input_ids_cpu.
        # QAIC pads each decode request to max_decode_tokens
        num_reqs = num_draft_tokens.size
        sig_counts = self.signal_num_scheduled_tokens[:num_reqs]  # type: ignore[index]
        num_pads = self.max_decode_tokens - sig_counts
        cu_num_pads = np.cumsum(num_pads)
        draft_token_indices = np.repeat(
            (cu_num_sampled_tokens - num_sampled_tokens) + (cu_num_pads - num_pads),
            num_draft_tokens,
        )
        draft_token_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )
        # Compute the draft token ids from the padded input_ids_cpu layout.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids_cpu[draft_token_indices + 1]
        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=None,
        )
        return metadata

    @contextmanager
    def synchronize_input_prep(self):
        if self.use_async_scheduling:
            # If there are any previous processed batch or pending model state
            if self.input_batch.req_ids or self.execute_model_state is not None:
                # Only block when we actually expect a previous step
                self.model_runner_output_event.wait(
                    timeout=self.model.async_scheduling_exec_timeout  # type: ignore
                )
            # Reset the event before the new step begins so the next caller will block
            self.model_runner_output_event.clear()

        yield

    def create_logits_np(self, batch_size, vocab_size, num_decode_tokens: int = 1):
        if num_decode_tokens > 1:
            return np.empty(
                (batch_size, num_decode_tokens, vocab_size), dtype=np.float32
            )
        if self.model.logits_ndim == 3:  # type: ignore[has-type]
            return np.empty((batch_size, 1, vocab_size), dtype=np.float32)
        return np.empty((batch_size, vocab_size), dtype=np.float32)

    def complete_all_inf(
        self, pending_prefill_exec_queue: Queue | None, num_decodes: int | None
    ):
        if pending_prefill_exec_queue is not None:
            while not pending_prefill_exec_queue.empty():
                exec_obj_idx = pending_prefill_exec_queue.get(
                    timeout=self.model.async_scheduling_exec_timeout  # type: ignore
                )
                self.model.complete_inf(exec_obj_idx, True)  # type: ignore[has-type]
        # kv_consumer runs decode on last prompt token as well
        if (num_decodes is not None and num_decodes > 0) or self.is_kv_consumer:
            self.model.complete_inf(self.model.decode_execObj_idx, False)  # type: ignore
            if self.model.on_device_sampling_en:
                fresh_retained_states = self.model.session.refresh_retained_state_buffers(
                    [
                        "past_repetition_penalty_buffer_RetainedState",
                        "past_presence_penalty_buffer_RetainedState",
                    ],
                    self.model.decode_execObj_idx,
                )
                np.copyto(
                    self.model.past_repetition_penalty_buffer_retained_state,
                    fresh_retained_states[
                        "past_repetition_penalty_buffer_RetainedState"
                    ],
                )
                np.copyto(
                    self.model.past_presence_penalty_buffer_retained_state,
                    fresh_retained_states[
                        "past_presence_penalty_buffer_RetainedState"
                    ],
                )

    def _compute_hidden_states_and_logits(
        self,
        hidden_states_decode,
        hidden_states_prefill,
        num_decodes,
        spec_decode_metadata=None,
    ):
        if (
            hidden_states_decode is not None
            and self.max_decode_tokens > 1
            and self.active_k > 0
        ):
            # SpD: hidden_states_decode has shape (decode_bsz, active_k+1, vocab).
            # Reshape to 2D and apply garbage-logit mask to remove padding positions.
            active_mdt = self.active_k + 1
            dec_flat = hidden_states_decode[:num_decodes].reshape(
                num_decodes * active_mdt, -1
            )
            assert self.tile_arange_mdts is not None
            tile_arange_mdt = self.tile_arange_mdts[num_decodes - 1]
            # When there are active draft proposals, derive signal_num_decode_tokens
            # from spec_decode_metadata.num_draft_tokens (numerically equivalent to
            # signal_num_scheduled_tokens but avoids a separate array lookup).
            # On the first decode step — no proposals yet — fall back to
            # signal_num_scheduled_tokens.
            if spec_decode_metadata is not None:
                num_draft_arr = np.array(
                    spec_decode_metadata.num_draft_tokens[:num_decodes], dtype=np.int32
                )
                signal_num_decode_tokens = 1 + num_draft_arr
            else:
                signal_num_decode_tokens = self.signal_num_scheduled_tokens[  # type: ignore[index]
                    :num_decodes
                ]
            signal_repeated = np.repeat(signal_num_decode_tokens, active_mdt)
            mask = tile_arange_mdt < signal_repeated
            hidden_states_decode = dec_flat[mask]  # (total_signal_tokens, vocab)

        if hidden_states_prefill is not None and hidden_states_decode is not None:
            if self.max_decode_tokens > 1 and self.active_k > 0:
                # SpD with K>0: garbage masking applied, shape (signal_tokens, vocab)
                dec_tensor = torch.from_numpy(hidden_states_decode)
            else:
                # K=0 fallback or nospec: shape is (bsz, 1, vocab) → squeeze to 2D
                dec_tensor = torch.from_numpy(
                    hidden_states_decode[:num_decodes].squeeze(1)
                )
            hidden_states_prefill = torch.from_numpy(hidden_states_prefill.squeeze(1))
            hidden_states = torch.cat((dec_tensor, hidden_states_prefill), dim=0)
        else:
            if hidden_states_prefill is not None:
                hidden_states = torch.from_numpy(hidden_states_prefill.squeeze(1))
            elif self.max_decode_tokens > 1 and self.active_k > 0:
                # SpD with K>0: already garbage-masked to 2D (signal_tokens, vocab)
                hidden_states = torch.from_numpy(hidden_states_decode)
            else:
                hidden_states = torch.from_numpy(
                    hidden_states_decode[:num_decodes].squeeze(1)
                )
        logits = self.model.compute_logits(hidden_states)  # type: ignore[has-type]
        return hidden_states, logits

    def _mm_preprocess(
        self,
        scheduler_output: SchedulerOutput,
    ) -> list[dict]:
        mm_kwargs_list = []
        is_encoder_decoder = self.model_config.is_encoder_decoder
        if is_encoder_decoder:
            for req in scheduler_output.scheduled_new_reqs:
                if len(req.mm_features) != 1 or req.mm_features[0].data is None:
                    raise ValueError(
                        "Encoder-decoder models require exactly one multimodal input "
                        f"with non-None data, got {len(req.mm_features)} features."
                    )
                mm_data = req.mm_features[0].data.get_data()
                _res = self.model.parse_and_validate_multimodal_raw_input(  # type: ignore
                    **mm_data
                )
                mm_kwargs, _ = _res
                mm_kwargs_list.append(mm_kwargs)
        else:
            # TODO: if we can support ec connector, it will go here
            # with self.maybe_get_ec_connector_output(
            #    scheduler_output,
            #    encoder_cache=self.encoder_cache,
            # ) as ec_connector_output:
            self._execute_mm_encoder(scheduler_output)
            prefill_req_ids = self.input_batch.req_ids[self.num_decodes :]
            mm_embeds = self._gather_mm_embeddings(
                scheduler_output, req_ids=prefill_req_ids
            )
            for mm_embed in mm_embeds:
                _kw = self.model.prepare_embedding_mm_kwargs(mm_embed)  # type: ignore
                mm_kwargs_list.append(_kw)
        return mm_kwargs_list

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with (
            record_function_or_nullcontext("qaic_model_runner: preprocess"),
            self.synchronize_input_prep(),
        ):
            if self.execute_model_state is not None:
                raise RuntimeError(
                    "State error: sample_tokens() must be called "
                    "after execute_model() returns None."
                )

            self._update_states(scheduler_output)

            if self.model.is_vision_encoder:
                # This is referencing to gpu_model_runner
                # if has_ec_transfer() and get_ec_transfer().is_producer
                self._execute_mm_encoder(scheduler_output)
                # Gather the encoder outputs that _execute_mm_encoder placed in
                # the encoder cache; pass all req_ids since every request in this
                # batch is a vision-encoder pooling request.
                mm_embeds = self._gather_mm_embeddings(
                    scheduler_output,
                    req_ids=self.input_batch.req_ids,
                    for_pooling_output=True,
                )
                return self.make_encoder_model_runner_output(mm_embeds)

            if not num_scheduled_tokens:
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT

                return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)

            # Variable-K: determine active K BEFORE _prepare_qaic_inputs,
            # which uses self.active_k to pad decode requests.
            self.active_k = self._determine_active_k(scheduler_output)
            if len(self.decode_ks) > 1:
                # Multi-spec: propagate the per-step active_k to the model so
                # _run_decode dispatches to the correct per-K buffer.
                # Single-spec: model.active_k stays at its init default (max K);
                # no update needed.
                self.model.active_k = self.active_k

            # Prepare inputs
            spec_decode_metadata = self._prepare_qaic_inputs(
                scheduler_output, num_scheduled_tokens_np
            )

            # Split positions and inputs into decode and prefill
            # Variable-K: num_decode_tokens = num_decodes * (active_k + 1)
            self.num_decode_tokens = self.num_decodes * (self.active_k + 1)
            if self.max_decode_tokens > 1:
                # SpD pads decode slots; recompute total scheduled token count
                num_scheduled_tokens = int(num_scheduled_tokens_np.sum())

            # Note: Disagg consumer runs decode on the last prompt token
            # decode arrays
            input_ids = self.input_ids.cpu.numpy()
            decode_input_ids: np.ndarray = (
                input_ids[: self.num_decode_tokens]
                if not self.is_kv_consumer
                else input_ids[:num_scheduled_tokens]
            )
            if self.uses_mrope:
                # Concatenate mrope positions with positions
                # because for models that use mrope,
                # QEfficient requires position ids to be (4, batch_size, seq_len)
                # while mrope is (3, batch_size, seq_len)
                _decode_tokens_end = (
                    self.num_decode_tokens
                    if not self.is_kv_consumer
                    else num_scheduled_tokens
                )
                decode_positions: np.ndarray = np.concatenate(
                    [
                        self.positions_np[:_decode_tokens_end][np.newaxis],
                        self.mrope_positions_np[:, :_decode_tokens_end],
                    ],
                    axis=0,
                )
            else:
                decode_positions = (
                    self.positions_np[: self.num_decode_tokens]
                    if not self.is_kv_consumer
                    else self.positions_np[:num_scheduled_tokens]
                )
            decode_block_ids: np.ndarray = (
                self.batch_indices[: self.num_decodes]
                if not self.is_kv_consumer
                else self.batch_indices[:num_scheduled_tokens]
            )

            if self.max_decode_tokens > 1:
                # mark padded positions as -1 so QAIC hardware ignores them
                decode_positions[decode_input_ids == -1] = -1

            # prefill arrays
            prefill_input_ids: np.ndarray = input_ids[
                self.num_decode_tokens : num_scheduled_tokens
            ]
            if self.uses_mrope:
                prefill_positions: np.ndarray = np.concatenate(
                    [
                        self.positions_np[
                            self.num_decode_tokens : num_scheduled_tokens
                        ][np.newaxis],
                        self.mrope_positions_np[
                            :, self.num_decode_tokens : num_scheduled_tokens
                        ],
                    ],
                    axis=0,
                )
            else:
                prefill_positions = self.positions_np[
                    self.num_decode_tokens : num_scheduled_tokens
                ]
            prefill_block_ids: np.ndarray = self.batch_indices[self.num_decodes :]
            prefill_cum_sum = (
                self.cu_num_tokens[self.num_decodes :] - self.num_decode_tokens
            )

            prefill_is_partial = self.discard_request_mask.np[
                self.num_decodes : self.input_batch.num_reqs
            ]
            discard_request_mask_np = self.discard_request_mask.np[
                : self.input_batch.num_reqs
            ].copy()

            # mm_kwargs_list is only needed for prefill requests; skip preprocessing
            # entirely when there are no prefills or the model has no mm inputs.
            if self.supports_mm_inputs and prefill_input_ids.size > 0:
                mm_kwargs_list = self._mm_preprocess(scheduler_output)
                num_prefill_reqs = len(self.input_batch.req_ids) - self.num_decodes
                assert len(mm_kwargs_list) == num_prefill_reqs
            else:
                mm_kwargs_list = None

        with (
            set_forward_context(
                None,
                self.vllm_config,
                num_tokens=scheduler_output.total_num_scheduled_tokens,
            ),
            record_function_or_nullcontext("qaic_model_runner: forward"),
            self.maybe_get_kv_connector_output(
                scheduler_output,
                kv_cache_info=self.kv_cache_info,  # type: ignore[has-type]
                defer_save=self.is_async_kv_producer,
            ) as kv_connector_output,
        ):
            callback = get_forward_context().additional_kwargs.get(
                "cleanup_callback", None
            )
            # Run Decode & Prefill requests separately
            # Note: The np buffer should have a lifecycle longer than
            # waitForCompletion to avoid SEGFAULT
            hidden_states_decode = None
            hidden_states_prefill = None
            pending_prefill_exec_queue = None

            num_reqs = self.input_batch.num_reqs
            decode_lora_ids = None
            prefill_lora_ids = None
            if self.lora_config:
                req_lora_mapping = self.input_batch.request_lora_mapping[:num_reqs]
                decode_lora_ids = req_lora_mapping[: self.num_decodes].astype(np.int64)
                prefill_lora_ids = req_lora_mapping[self.num_decodes :].astype(np.int64)

            ods_sampling_params_prefill: dict[str, np.ndarray] | None = None
            ods_sampling_params_decode: dict[str, np.ndarray] | None = None
            if self.model.on_device_sampling_en:
                self.model.ods_step_prefill_next_tokens = None
                self.model.ods_step_num_decodes = 0
                self.model.ods_step_decode_mdt = 1

                sampling_metadata = self.input_batch.sampling_metadata
                if sampling_metadata is None:
                    raise ValueError(
                        "On-device sampling is enabled but sampling_metadata is missing "
                        "for the current batch."
                    )

                req_ids = self.input_batch.req_ids
                num_reqs = self.input_batch.num_reqs
                temperatures = np.empty((num_reqs,), dtype=np.float32)
                top_ks = np.empty((num_reqs,), dtype=np.int32)
                top_ps = np.empty((num_reqs,), dtype=np.float32)
                min_ps = np.empty((num_reqs,), dtype=np.float32)
                for slot_index, req_id in enumerate(req_ids):
                    sampling_params = self.requests[req_id].sampling_params
                    temperatures[slot_index] = np.float32(sampling_params.temperature)
                    top_ks[slot_index] = np.int32(sampling_params.top_k)
                    top_ps[slot_index] = np.float32(sampling_params.top_p)
                    min_ps[slot_index] = np.float32(sampling_params.min_p)

                top_ks = np.where(top_ks <= 0, self.model.ods_max_top_k_ids, top_ks)
                top_ks = np.minimum(top_ks, self.model.ods_max_top_k_ids).astype(
                    np.int32,
                    copy=False,
                )
                if sampling_metadata.no_penalties:
                    repetition_penalties = np.full((num_reqs,), 1.0, dtype=np.float32)
                    presence_penalties = np.full((num_reqs,), 0.0, dtype=np.float32)
                else:
                    repetition_penalties = (
                        sampling_metadata.repetition_penalties.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=True)
                    )
                    presence_penalties = (
                        sampling_metadata.presence_penalties.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32, copy=True)
                    )
                random_numbers = np.empty(
                    (num_reqs, self.model.ods_max_top_k_ids),
                    dtype=np.float32,
                )
                for slot_index in range(num_reqs):
                    generator = sampling_metadata.generators.get(slot_index)
                    if generator is None:
                        random_values = torch.rand(
                            self.model.ods_max_top_k_ids,
                            dtype=torch.float32,
                        )
                    else:
                        random_values = torch.rand(
                            self.model.ods_max_top_k_ids,
                            generator=generator,
                            dtype=torch.float32,
                        )
                    random_numbers[slot_index] = (
                        random_values.detach().cpu().numpy().astype(np.float32, copy=False)
                    )

                all_sampling_params: dict[str, np.ndarray] = {
                    "temperatures": temperatures,
                    "top_ks": top_ks,
                    "top_ps": top_ps,
                    "min_ps": min_ps,
                    "repetition_penalties": repetition_penalties,
                    "presence_penalties": presence_penalties,
                    "random_numbers": random_numbers,
                }
                ods_sampling_params_decode = {
                    key: values[: self.num_decodes]
                    for key, values in all_sampling_params.items()
                }
                ods_sampling_params_prefill = {
                    key: values[self.num_decodes : num_reqs]
                    for key, values in all_sampling_params.items()
                }

            if prefill_input_ids.size > 0:
                hidden_states_prefill = (
                    self.create_logits_np(len(prefill_cum_sum), self.model.vocab_size)
                    if not self.is_kv_consumer and not self.model.on_device_sampling_en
                    else None
                )
                # Per-prefill-request total prompt length, for CCL bucket selection.
                num_prompt_tokens_prefill = self.input_batch.num_prompt_tokens[
                    self.num_decodes : self.input_batch.num_reqs
                ]
                pending_prefill_exec_queue = self.model(
                    input_ids=prefill_input_ids,
                    positions=prefill_positions,
                    batch_indices=prefill_block_ids,
                    is_prompt=True,
                    prefill_cum_sum=prefill_cum_sum,
                    mm_kwargs_list=mm_kwargs_list,
                    prefill_is_partial=prefill_is_partial,
                    logits=hidden_states_prefill,
                    kv_caches=self.kv_caches,
                    callback=callback,
                    lora_ids=prefill_lora_ids,
                    num_prompt_tokens_prefill=num_prompt_tokens_prefill,
                    sampling_params=ods_sampling_params_prefill,
                )

            if decode_input_ids.size > 0:
                hidden_states_decode = (
                    None
                    if self.model.on_device_sampling_en
                    else self.create_logits_np(
                        self.model.decode_bsz, self.model.vocab_size, self.active_k + 1
                    )
                )
                self.model(
                    input_ids=decode_input_ids,
                    positions=decode_positions,
                    batch_indices=decode_block_ids,
                    is_prompt=False,
                    logits=hidden_states_decode,
                    callback=callback,
                    lora_ids=decode_lora_ids,
                    sampling_params=ods_sampling_params_decode,
                )

        hidden_states, logits = None, None
        num_decodes_executed = (
            self.num_decodes if not self.is_kv_consumer else len(self.cu_num_tokens)
        )
        if not self.use_async_scheduling:
            if self.model.on_device_sampling_en:
                hidden_states = torch.empty((0, 0), dtype=torch.float32)
                logits = torch.zeros((self.input_batch.num_reqs, 1), dtype=torch.float32)
            else:
                hidden_states, logits = self._compute_hidden_states_and_logits(
                    hidden_states_decode,
                    hidden_states_prefill,
                    num_decodes_executed,
                    spec_decode_metadata=spec_decode_metadata,
                )

        spec_decode_common_attn_metadata = None
        if self.speculative_config is not None:
            spec_decode_common_attn_metadata = QaicSpecDecodeCommonAttnMetadata(
                max_seq_len=self.spec_decode_max_seq_len
            )
        if self.is_pooling_model and not self.model_config.is_multimodal_model:
            if not self.use_async_scheduling:
                return self._pool(
                    pending_prefill_exec_queue,
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                    kv_connector_output,
                )
            else:
                return QaicAsyncPoolingModelRunnerOutput(
                    model_runner=self,
                    pending_prefill_exec_queue=pending_prefill_exec_queue,
                    num_scheduled_tokens=num_scheduled_tokens,
                    num_scheduled_tokens_np=num_scheduled_tokens_np,
                    kv_connector_output=kv_connector_output,
                )

        self.execute_model_state = QaicExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            None,  # sample_hidden_states
            None,  # aux_hidden_states
            None,  # ec_connector_output
            None,  # cudagraph_stats
            None,  # slot_mappings
            hidden_states_decode,
            hidden_states_prefill,
            pending_prefill_exec_queue,
            num_decodes_executed,
            discard_request_mask_np,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            if self.use_async_scheduling:
                self.model_runner_output_event.set()
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        if self.use_async_scheduling:
            state = self.execute_model_state

            _connector_metadata = None
            _input_batch_req_ids = None
            _input_batch_req_id_to_index = None
            if self.is_async_kv_producer:
                if has_kv_transfer_group():
                    kv_connector = get_kv_transfer_group()
                    _connector_metadata = deepcopy(
                        kv_connector._get_connector_metadata()
                    )
                    kv_connector.clear_connector_metadata()
                state = self.execute_model_state
                _input_batch_req_ids = self.input_batch.req_ids.copy()
                _input_batch_req_id_to_index = self.input_batch.req_id_to_index.copy()
                # Disagg prefill: unblock the next batch early since sampled
                # tokens are not needed.
                self.execute_model_state = None
                self.model_runner_output_event.set()

            return QaicAsyncGPUModelRunnerOutput(
                model_runner=self,
                state=state,
                grammar_output=grammar_output,
                kv_connector_output=kv_connector_output,
                kv_connector_metadata=_connector_metadata,
                input_batch_req_ids=_input_batch_req_ids,
                input_batch_req_id_to_index=_input_batch_req_id_to_index,
            )

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
            *_,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None and not self.model.on_device_sampling_en:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        if self.is_kv_producer:
            sampler_output = self._make_sampler_output(
                torch.zeros((len(self.batch_indices), 1), dtype=torch.int64)
            )
        else:
            sampling_metadata = None
            if self.model.on_device_sampling_en:
                sampling_metadata = self.input_batch.sampling_metadata
                if sampling_metadata is not None:
                    nondefault_slots = torch.nonzero(
                        sampling_metadata.frequency_penalties != 0.0,
                        as_tuple=False,
                    ).flatten()
                    req_ids = self.input_batch.req_ids
                    for slot_index in nondefault_slots.tolist():
                        req_id = req_ids[slot_index]
                        if req_id in self._ods_frequency_penalty_warned_req_ids:
                            continue
                        logger.warning(
                            "Request %s specified non-default frequency_penalty, "
                            "but frequency_penalty is ignored when on-device "
                            "sampling is enabled.",
                            req_id,
                        )
                        self._ods_frequency_penalty_warned_req_ids.add(req_id)
            if self.model.on_device_sampling_en:
                if spec_decode_metadata is not None:
                    raise ValueError(
                        "On-device sampling requires speculative decoding to be "
                        "disabled; spec_decode_metadata must be None."
                    )
                self.input_batch.update_async_output_token_ids()
                sampler_output = self.model.sample(logits, sampling_metadata)
            else:
                sampler_output = self._sample(logits, spec_decode_metadata)

        # AOT-only path: eager mode returns early via super().execute_model() above.
        self._draft_token_ids = None
        self._draft_token_req_ids = None
        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            self._draft_token_ids = self.propose_draft_token_ids(
                scheduler_output,
                sampled_token_ids,
                self.input_batch.sampling_metadata,
                hidden_states,
                sample_hidden_states,
                aux_hidden_states,
                spec_decode_metadata,
                spec_decode_common_attn_metadata,  # type: ignore[arg-type]
                slot_mappings,
            )
            self._copy_draft_token_ids_to_cpu(scheduler_output)

        # This block mirrors GPUModelRunner (gpu_model_runner.py:3640).
        # Difference: spec_decode_common_attn_metadata uses
        # QaicSpecDecodeCommonAttnMetadata (minimal stub with .max_seq_len only)
        # instead of CommonAttentionMetadata.
        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None and not self.is_kv_producer:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            # QAIC: draft_model always proposes after bookkeeping because
            # QAICInferenceSession.run() is fully synchronous and we need CPU
            # token IDs.
            propose_drafts_after_bookkeeping = input_fits_in_drafter

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
        )
        if self.model.on_device_sampling_en:
            num_nans_in_logits = None
        if self.model.on_device_sampling_en and scheduler_output.finished_req_ids:
            self._ods_frequency_penalty_warned_req_ids.difference_update(
                scheduler_output.finished_req_ids
            )

        if propose_drafts_after_bookkeeping:
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        return ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

    def _make_sampler_output(
        self,
        next_token_ids: torch.Tensor,
    ) -> SamplerOutput:
        return SamplerOutput(next_token_ids, None)

    def propose_draft_token_ids(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: QaicSpecDecodeCommonAttnMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]] | torch.Tensor:
        spec_config = self.speculative_config
        assert spec_config is not None
        assert self.drafter is not None
        if spec_config.method == "ngram":
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        elif spec_config.method == "suffix":
            # convert input_ids from int64 -> int32 to satisfy
            # ArcticInference type checks
            token_ids_cpu_orig = self.input_batch.token_ids_cpu
            self.input_batch.token_ids_cpu = token_ids_cpu_orig.astype(
                np.int32, copy=False
            )
            draft_token_ids = self.drafter.propose(
                self.input_batch, sampled_token_ids, slot_mappings=slot_mappings
            )
            self.input_batch.token_ids_cpu = token_ids_cpu_orig
        elif spec_config.uses_draft_model():
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                scheduler_output,
                self.input_batch,
                self.batch_indices,
            )
        else:
            raise ValueError(
                f"Unknown speculative decoding method: {spec_config.method}"
            )
        return draft_token_ids

    def load_model(self, *args, **kwargs) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        time_before_load = time.perf_counter()
        from vllm_qaic.model_loader.qaic import load_qaic_model

        with set_current_vllm_config(self.vllm_config):
            speculative_model_type = "default"
            if self.num_spec_tokens:
                speculative_model_type = "target"
            self.model: nn.Module = load_qaic_model(
                self.vllm_config, speculative_model_type
            )
            # FIXME load_lora_model parameters have changed in the mixin
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model,
                    self.vllm_config,
                    self.device,
                )
        self.kv_cache_info = (
            self.model.kv_cache_info if self.model.disagg_serving_en else None
        )
        kv_transfer_config = self.vllm_config.kv_transfer_config
        if (
            self.kv_cache_info
            and kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "QaicLMCacheConnectorV1"
        ):
            assert len(set(self.kv_cache_info)) == 1, (
                "QaicLMCacheConnectorV1 currently does not support"
                " models with hybrid KV cache"
            )
        if (
            self.speculative_config is not None
            and self.speculative_config.uses_draft_model()
            and self.drafter is not None
        ):
            self.drafter.load_model()

        time_after_load = time.perf_counter()
        logger.info(
            "Model loading took %.6f seconds",
            time_after_load - time_before_load,
        )

    def _qaic_dummy_run(self) -> None:
        if self.is_pooling_model:
            # TODO: check if pooler dummy run can be added
            logger.debug("Skipping dummy run for pooling model")
            return

        if self.model.disagg_serving_en:
            self.model.disagg_dummy_run()
            return
        if self.model.is_vision_encoder:
            return

        # Decode (SpD-aware: allocate max_decode_tokens per request)
        decode_bsz = self.model.decode_bsz
        decode_num_tokens = decode_bsz * self.max_decode_tokens
        decode_input_ids = np.array([0] * decode_num_tokens, dtype=np.int64)
        if self.uses_mrope:
            decode_positions = np.array([[0] * decode_bsz] * 4, dtype=np.int64)
        else:
            decode_positions = np.array([0] * decode_num_tokens, dtype=np.int64)
        decode_block_ids = np.arange(decode_bsz, dtype=np.int64)
        decode_lora_ids = None
        if self.lora_config:
            decode_lora_ids = np.arange(decode_bsz, dtype=np.int64)

        decode_mm_kwargs_list = None
        if self.model.is_multimodal_model and self.model.default_mm_kwargs:
            decode_mm_kwargs_list = [self.model.default_mm_kwargs] * decode_bsz

        if self.model.on_device_sampling_en:
            from QEfficient.utils import constants as qefficient_constants
            from QEfficient.utils.constants import Constants

            example_temperature = getattr(
                Constants,
                "ONNX_EXPORT_EXAMPLE_TEMPERATURES",
                qefficient_constants.ONNX_EXPORT_EXAMPLE_TEMPERATURES,
            )
            example_top_p = getattr(
                Constants,
                "ONNX_EXPORT_EXAMPLE_TOP_PS",
                qefficient_constants.ONNX_EXPORT_EXAMPLE_TOP_PS,
            )
            example_min_p = getattr(
                Constants,
                "ONNX_EXPORT_EXAMPLE_MIN_PS",
                qefficient_constants.ONNX_EXPORT_EXAMPLE_MIN_PS,
            )
            example_repetition_penalty = getattr(
                Constants,
                "ONNX_EXPORT_EXAMPLE_REPETITION_PENALTIES",
                qefficient_constants.ONNX_EXPORT_EXAMPLE_REPETITION_PENALTIES,
            )
            example_presence_penalty = getattr(
                Constants,
                "ONNX_EXPORT_EXAMPLE_PRESENCE_PENALTIES",
                qefficient_constants.ONNX_EXPORT_EXAMPLE_PRESENCE_PENALTIES,
            )

            decode_sampling_params: dict[str, np.ndarray] = {
                "temperatures": np.full(
                    (decode_bsz, 1),
                    example_temperature,
                    dtype=np.float32,
                ),
                "top_ks": np.full(
                    (decode_bsz, 1),
                    self.model.ods_max_top_k_ids,
                    dtype=np.int32,
                ),
                "top_ps": np.full(
                    (decode_bsz, 1),
                    example_top_p,
                    dtype=np.float32,
                ),
                "min_ps": np.full(
                    (decode_bsz, 1),
                    example_min_p,
                    dtype=np.float32,
                ),
                "repetition_penalties": np.full(
                    (decode_bsz, 1),
                    example_repetition_penalty,
                    dtype=np.float32,
                ),
                "presence_penalties": np.full(
                    (decode_bsz, 1),
                    example_presence_penalty,
                    dtype=np.float32,
                ),
                "random_numbers": np.random.default_rng(0)
                .random((decode_bsz, self.model.ods_max_top_k_ids))
                .astype(np.float32),
            }
            # ODS QPCs consume next_tokens/sampling controls, not logits.
            # Skip unused logits allocation on warm-up to avoid wasted memory.
            self.model(
                input_ids=decode_input_ids,
                positions=decode_positions,
                batch_indices=decode_block_ids,
                lora_ids=decode_lora_ids,
                is_prompt=False,
                mm_kwargs_list=decode_mm_kwargs_list,
                sampling_params=decode_sampling_params,
            )
        else:
            decode_logits = self.create_logits_np(
                self.model.decode_bsz, self.model.vocab_size, self.max_decode_tokens
            )
            self.model(
                input_ids=decode_input_ids,
                positions=decode_positions,
                batch_indices=decode_block_ids,
                lora_ids=decode_lora_ids,
                is_prompt=False,
                logits=decode_logits,
                mm_kwargs_list=decode_mm_kwargs_list,
            )

        # Prefill
        prefill_bsz = self.model.prefill_bsz
        prefill_seq_len = self.model.prefill_seq_len
        prefill_input_ids = self.input_batch.token_ids_cpu[
            :prefill_bsz, :prefill_seq_len
        ].flatten()
        if self.uses_mrope:
            prefill_positions = np.tile(
                self.arange_np[:prefill_seq_len], (4, prefill_bsz, 1)
            )
        else:
            prefill_positions = self.arange_np[:prefill_seq_len].repeat(prefill_bsz)
        prefill_block_ids = np.arange(prefill_bsz)
        prefill_cum_sum = np.array(
            [prefill_seq_len] * prefill_bsz, dtype=np.int64
        ).cumsum()
        prefill_lora_ids = None
        if self.lora_config:
            prefill_lora_ids = np.arange(prefill_bsz, dtype=np.int64)
        mm_kwargs_list = None
        if self.model.is_multimodal_model and self.model.default_mm_kwargs:
            mm_kwargs_list = [self.model.default_mm_kwargs] * prefill_bsz

        if self.model.on_device_sampling_en:
            prefill_sampling_params: dict[str, np.ndarray] = {
                "temperatures": np.full(
                    (prefill_bsz, 1),
                    example_temperature,
                    dtype=np.float32,
                ),
                "top_ks": np.full(
                    (prefill_bsz, 1),
                    self.model.ods_max_top_k_ids,
                    dtype=np.int32,
                ),
                "top_ps": np.full(
                    (prefill_bsz, 1),
                    example_top_p,
                    dtype=np.float32,
                ),
                "min_ps": np.full(
                    (prefill_bsz, 1),
                    example_min_p,
                    dtype=np.float32,
                ),
                "repetition_penalties": np.full(
                    (prefill_bsz, 1),
                    example_repetition_penalty,
                    dtype=np.float32,
                ),
                "presence_penalties": np.full(
                    (prefill_bsz, 1),
                    example_presence_penalty,
                    dtype=np.float32,
                ),
                "random_numbers": np.random.default_rng(0)
                .random((prefill_bsz, self.model.ods_max_top_k_ids))
                .astype(np.float32),
            }
            # ODS QPCs consume next_tokens/sampling controls, not logits.
            # Skip unused logits allocation on warm-up to avoid wasted memory.
            pending_prefill_exec_queue = self.model(
                input_ids=prefill_input_ids,
                positions=prefill_positions,
                batch_indices=prefill_block_ids,
                lora_ids=prefill_lora_ids,
                is_prompt=True,
                prefill_cum_sum=prefill_cum_sum,
                mm_kwargs_list=mm_kwargs_list,
                sampling_params=prefill_sampling_params,
            )
        else:
            prefill_logits = self.create_logits_np(prefill_bsz, self.model.vocab_size)
            pending_prefill_exec_queue = self.model(
                input_ids=prefill_input_ids,
                positions=prefill_positions,
                batch_indices=prefill_block_ids,
                lora_ids=prefill_lora_ids,
                is_prompt=True,
                prefill_cum_sum=prefill_cum_sum,
                mm_kwargs_list=mm_kwargs_list,
                logits=prefill_logits,
            )

        if self.use_async_scheduling:
            self.complete_all_inf(pending_prefill_exec_queue, decode_bsz)

    def _gather_mm_embeddings(
        self,
        scheduler_output: SchedulerOutput,
        req_ids: list[str] | None = None,
        for_pooling_output: bool = False,
    ) -> list:
        """
        Modified from GPUModelRunner's _gather_mm_embeddings
        Added for_pooling_output argument
        Removed code related to multimodal pruning
        (Efficient Video Sampling (EVS) for redundant video tokens pruning)
        to support that in the future once we support video,
        need to pull in relevant code from vllm's implementation
        for Qwen2.5VL and Qwen3VL.
        """
        mm_embeds: list[torch.Tensor | None] = []

        if req_ids is None:
            req_ids = self.input_batch.req_ids

        for req_id in req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if pos_info.is_embed is not None:
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                mm_embeds_req.append(mm_embeds_item)

            if len(mm_embeds_req) == 0:
                mm_embeds.append(None)
            elif len(mm_embeds_req) == 1:
                item = mm_embeds_req[0]
                mm_embeds.append(item.unsqueeze(0) if item.ndim == 2 else item)
            else:
                # Multi-image support is currently limited to Qwen2.5-VL, Qwen3-VL,
                # and InternVL. Compatibility with video inputs and
                # other models is untested and may not work as expected.
                if for_pooling_output and not self.model.is_qwenvl:
                    mm_embeds.append(torch.stack(mm_embeds_req, dim=0))
                else:
                    mm_embeds.append(torch.cat(mm_embeds_req, dim=-2))
        return mm_embeds

    def make_encoder_model_runner_output(
        self,
        mm_embeds: list[torch.Tensor],
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )
        mm_embeds = mm_embeds[:num_reqs]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
        )

        model_runner_output.pooler_output = mm_embeds

        return model_runner_output

    def get_model(self) -> nn.Module:
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        supported_tasks = list[GenerationTask]()
        if self.model.config.model_type == "whisper":
            supported_tasks.append("transcription")
        else:
            supported_tasks.append("generate")
        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.model.pooler.get_supported_tasks())

        return tuple(tasks)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        self.kv_cache_config = kv_cache_config
        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(self.kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        block_size = self.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        n_layers = self.model_config.get_num_layers(self.parallel_config)
        for i in range(n_layers):
            layer_name = f"layer_{i}"
            kv_cache_spec[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_dtype,
            )
        return kv_cache_spec

    def add_lora(self, lora_request: LoRARequest) -> bool:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        return True

    @staticmethod
    def _save_kv_and_finalize_connector_output(
        kv_connector: KVConnectorBase,
        output: KVConnectorOutput,
        finished_req_ids,
        wait_for_save: bool = True,
        clear_metadata: bool = False,
        **kwargs,
    ) -> None:
        kv_connector.save_kv_layer(
            layer_name=None,
            kv_layer=None,
            attn_metadata=None,
            **kwargs,
        )
        if wait_for_save:
            kv_connector.wait_for_save(**kwargs)
        output.finished_sending, output.finished_recving = kv_connector.get_finished(
            finished_req_ids
        )
        output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = kv_connector.get_kv_connector_stats()
        output.kv_cache_events = kv_connector.get_kv_connector_kv_cache_events()
        if clear_metadata:
            kv_connector.clear_connector_metadata()

    @staticmethod
    def maybe_get_kv_connector_output(
        scheduler_output: SchedulerOutput, **kwargs
    ) -> AbstractContextManager[KVConnectorOutput | None]:
        return (
            QaicModelRunnerAoT._get_kv_connector_output(scheduler_output, **kwargs)
            if has_kv_transfer_group()
            else nullcontext()
        )

    # This context manager must be used within an active forward context.
    # It encapsulates the entire KV connector lifecycle within execute_model
    @staticmethod
    @contextmanager
    def _get_kv_connector_output(
        scheduler_output: SchedulerOutput,
        wait_for_save: bool = True,
        defer_save: bool = False,
        **kwargs,
    ) -> Generator[KVConnectorOutput, None, None]:
        output = KVConnectorOutput()

        # Update KVConnector with the KVConnector metadata forward().
        kv_connector = get_kv_transfer_group()
        assert isinstance(kv_connector, KVConnectorBase)
        assert scheduler_output.kv_connector_metadata is not None
        kv_connector.bind_connector_metadata(scheduler_output.kv_connector_metadata)

        # Background KV cache transfers happen here.
        # These transfers are designed to be async and the requests
        # involved may be disjoint from the running requests.
        # Do this here to save a collective_rpc.
        kv_connector.start_load_kv(get_forward_context(), **kwargs)
        try:
            yield output
        finally:
            if not defer_save:
                QaicModelRunnerAoT._save_kv_and_finalize_connector_output(
                    kv_connector,
                    output,
                    scheduler_output.finished_req_ids,
                    wait_for_save=wait_for_save,
                    clear_metadata=True,
                    **kwargs,
                )


@contextmanager
def _torch_cuda_wrapper():
    class _EventPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


def reorder_batch_to_split_decodes_and_prefills(
    input_batch: InputBatch,
    scheduler_output: SchedulerOutput,
) -> tuple[bool, int, int]:
    """
    Modified from the original method in utils to customize it for qaic

    Reorders the batch to split into prefill and decode requests; places all
    decode requests at the front of the batch.
    Returns:
        True if the batch was modified, False otherwise.
    """
    decodes = []
    prefills = []
    num_decode_tokens = 0
    num_prefill_tokens = 0

    for i, req_id in enumerate(input_batch.req_ids):
        num_tokens = scheduler_output.num_scheduled_tokens[req_id]
        req_index = input_batch.req_id_to_index.get(req_id)
        if (
            input_batch.num_computed_tokens_cpu[req_index]
            < input_batch.num_prompt_tokens[req_index]
        ):
            prefills.append(i)
            num_prefill_tokens += num_tokens
        else:
            decodes.append(i)
            num_decode_tokens += num_tokens
    num_decodes = len(decodes)
    num_prefills = len(prefills)
    modified_batch = False

    for i in range(1, min(num_decodes, num_prefills) + 1):
        decode_idx = decodes[num_decodes - i]
        if decode_idx < num_decodes:
            break

        input_batch.swap_states(prefills[i - 1], decode_idx)
        modified_batch = True

    return modified_batch, num_decodes, num_decode_tokens


@contextmanager
def _torch_qaic_wrapper():
    cuda_event = torch.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.Event = qaic.Event
        torch.cuda.Stream = qaic.Stream
        yield
    finally:
        torch.Event = cuda_event
        torch.cuda.Stream = cuda_stream


@contextmanager
def optional_qaic_profiling(profiling_dir, profiling_wrapper, **kwargs):
    if profiling_dir is not None:
        world_rank = dist.get_rank() if dist.is_initialized() else 0
        lrt_trace_dir = f"lrt_trace_iter{kwargs.pop('profiling_iter')}_pid{world_rank}"
        kwargs["dump_dir"] = os.path.join(profiling_dir, lrt_trace_dir)
        with profiling_wrapper(**kwargs) as model:
            yield model
    else:
        yield kwargs["model"]
