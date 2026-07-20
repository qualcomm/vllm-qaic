# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Utilities for selecting and loading qaic models."""

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from queue import Queue
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftConfig

from vllm.config import ModelConfig, VllmConfig
from vllm.entrypoints.openai.models.protocol import LoRAModulePath
from vllm_qaic.logger import init_logger
from vllm.model_executor.layers.pooler.common import ClassifierFn
from vllm.model_executor.layers.pooler.seqwise import (
    SequencePoolingFn,
    SequencePoolingMethod,
    pooler_for_classify,
    pooler_for_embed,
)
from vllm.model_executor.layers.pooler.special import DispatchPooler
from vllm.model_executor.layers.pooler.tokwise import (
    AllPool,
    pooler_for_token_classify,
    pooler_for_token_embed,
)
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.platforms import current_platform
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.utils.math_utils import cdiv
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_qaic.utils.qaic_utils import _clean_config, derive_ods_state

logger = init_logger(__name__)

lock = threading.Lock()


# NOTE: This startup guard depends on QEfficient's binding naming convention
# from QEfficient/utils/sampler_utils.py:get_sampling_inputs_and_outputs().
# Re-validate these binding names when bumping QEfficient versions.
_ODS_CORE_BINDING_NAMES = frozenset({"temperatures", "top_ks", "random_numbers"})
_ODS_DEBUG_PROBS_BINDING_NAME = "probs"


def _check_ods_binding_mismatch(
    engine_expects_ods: bool,
    binding_index_map: Mapping[str, int],
    engine_expects_debug_probs: bool = False,
) -> None:
    qpc_ods_binding_names = _ODS_CORE_BINDING_NAMES.intersection(binding_index_map)
    if 0 < len(qpc_ods_binding_names) < len(_ODS_CORE_BINDING_NAMES):
        present_bindings = sorted(qpc_ods_binding_names)
        missing_bindings = sorted(_ODS_CORE_BINDING_NAMES - qpc_ods_binding_names)
        raise ValueError(
            "On-device sampling binding integrity error at startup: loaded QPC "
            "contains a partial on-device-sampling binding set "
            f"(present={present_bindings}, missing={missing_bindings}). This "
            "indicates a corrupted or incompletely-compiled QPC. Recompile a "
            "valid QPC artifact before deployment."
        )

    qpc_has_ods_bindings = _ODS_CORE_BINDING_NAMES.issubset(binding_index_map)

    if engine_expects_ods and not qpc_has_ods_bindings:
        raise ValueError(
            "On-device sampling mismatch at startup: deployment is configured "
            "with on-device sampling enabled, but the loaded QPC was not compiled "
            "with on-device sampling bindings. Recompile the QPC with on-device "
            "sampling enabled, or disable on-device sampling in deployment "
            "configuration."
        )

    if qpc_has_ods_bindings and not engine_expects_ods:
        raise ValueError(
            "On-device sampling mismatch at startup: loaded QPC was compiled with "
            "on-device sampling bindings, but deployment is not configured to use "
            "on-device sampling. Enable on-device sampling in deployment "
            "configuration, or use a QPC compiled without on-device sampling."
        )

    if (
        engine_expects_debug_probs
        and _ODS_DEBUG_PROBS_BINDING_NAME not in binding_index_map
    ):
        raise ValueError(
            "On-device sampling debug/evaluation mismatch at startup: deployment "
            "is configured with aic_return_pdfs enabled, but the loaded QPC does "
            "not "
            "expose a `probs` output binding. Recompile the QPC with "
            "aic_return_pdfs=True (alongside aic_include_sampler=1), or disable "
            "aic_return_pdfs in deployment configuration."
        )

    if (
        _ODS_DEBUG_PROBS_BINDING_NAME in binding_index_map
        and not engine_expects_debug_probs
    ):
        raise ValueError(
            "On-device sampling debug/evaluation mismatch at startup: loaded QPC "
            "was compiled with return_pdfs=True (exposes a `probs` output "
            "binding), but deployment is not configured to use aic_return_pdfs. "
            "Enable aic_return_pdfs in override_qaic_config, or use a QPC compiled "
            "without return_pdfs."
        )


class QaicCompilationComplete(Exception):
    def __init__(
        self,
        message=(
            "Compilation is done, exiting from backend via "
            "raising a dummy exception, please ignore this exception"
        ),
    ):
        super().__init__(message)


class QaicCausalLM(nn.Module, SupportsLoRA):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        vllm_config: VllmConfig,
        pooling: SequencePoolingMethod | SequencePoolingFn | None = None,
        classifier: ClassifierFn | None = None,
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        config = model_config.hf_config

        pooler_config = vllm_config.model_config.pooler_config
        self._pooler = None
        self.is_pooling_model = False
        self.task = None
        if vllm_config.model_config.runner_type == "pooling":
            self.is_pooling_model = True
            self._pooler = DispatchPooler(
                {
                    "token_embed": pooler_for_token_embed(pooler_config),
                    "embed": pooler_for_embed(pooler_config),
                    "token_classify": pooler_for_token_classify(
                        pooler_config,
                        pooling=AllPool(),
                        classifier=classifier,
                    ),
                    "classify": pooler_for_classify(
                        pooler_config,
                        pooling=pooling,
                        act_fn=None,  # softmax not applied; raw logits returned
                    ),
                    # TODO: "score" removed: not a valid PoolingTask in upstream vllm 0.23.0
                    # (DispatchPooler validates keys; "score" was added in the Qualcomm fork)
                    # "score": pooler_for_classify(
                    #     pooler_config,
                    #     pooling=pooling,
                    #     classifier=classifier,
                    #     act_fn=None,  # sigmoid not applied; raw logits returned
                    # ),
                }
            )
            self.hidden_dimension = model_config.get_hidden_size()
            self.encode_num_logits_buffer: dict | None = None
            self.is_qaic_pooler = False
            self.normalize = False
            self.softmax = False
            override_qaic_config = (vllm_config.additional_config or {}).get(
                "override_qaic_config"
            ) or {}
            if override_qaic_config.get("pooling_device", None) == "qaic":
                self.is_qaic_pooler = True
                self.normalize = bool(override_qaic_config.get("normalize", False))
                self.softmax = bool(override_qaic_config.get("softmax", False))
            self.task: str | None = override_qaic_config.get("task", None)

        # TODO: Add new variables for turbo

        self.config = config
        self.vocab_size = config.get_text_config().vocab_size
        # `long_prefill_token_threshold` will define prefill chunk length
        if self.config.model_type == "whisper":
            # Encoder-decoder models have chunked prefill disabled by vllm,
            # but QAIC still requires a prefill sequence length.
            # For whisper, the prefill sequence length is fixed to 1.
            self.prefill_seq_len = 1
        else:
            self.prefill_seq_len = (
                vllm_config.scheduler_config.long_prefill_token_threshold
            )

        self.ctx_len = model_config.max_model_len
        self.decode_bsz = vllm_config.scheduler_config.max_num_seqs
        self.full_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.prefill_bsz = 1
        self.lora_mode = bool(vllm_config.lora_config)
        self.last_decode = False
        self.num_spec_tokens = 0
        override_qaic_config = (vllm_config.additional_config or {}).get(
            "override_qaic_config"
        ) or {}
        (
            self.on_device_sampling_en,
            self.ods_max_top_k_ids,
            self.debug_return_probs_en,
        ) = derive_ods_state(
            override_qaic_config,
            self.vocab_size,
        )
        if self.debug_return_probs_en and not self.on_device_sampling_en:
            raise ValueError(
                "The on-device sampling debug/evaluation sub-mode (return_pdfs) "
                "requires on-device sampling itself to be enabled; set "
                "aic_include_sampler=1 in override_qaic_config, or disable "
                "aic_return_pdfs."
            )
        self.max_decode_tokens = 1
        if vllm_config.speculative_config:
            self.num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens
            self.max_decode_tokens += self.num_spec_tokens

        self.num_logits_to_keep: int | None = None
        self.decode_logits: dict[str, np.ndarray] | None = None
        self.is_spec_decode_target_model = False

        # Variable-K decode specializations: compile with [0, K] for ngram/suffix
        # and dispatch to the smallest kernel that covers actual proposals each step.
        _method = (
            vllm_config.speculative_config.method
            if vllm_config.speculative_config
            else None
        )
        self.decode_ks: list[int] = (
            [0, self.num_spec_tokens]
            if _method in ("ngram", "suffix") and self.num_spec_tokens > 0
            else [self.num_spec_tokens]
        )
        # active_k is updated per step by QaicModelRunner; defaults to max K.
        self.active_k: int = self.decode_ks[-1]

        self.sampler = Sampler(logprobs_mode=model_config.logprobs_mode)
        self.pad = np.full(self.prefill_seq_len, fill_value=-1, dtype=np.int64)

        # Pre-allocate one input dict per K so session.run() receives the correct
        # static input shape for hardware dispatch.
        self.prefill_ods_inputs: dict[str, np.ndarray] | None = None
        self.prefill_next_tokens: np.ndarray | None = None
        self.prefill_last_accepted_output_tokens: np.ndarray | None = None
        self.past_repetition_penalty_buffer: np.ndarray | None = None
        self.past_presence_penalty_buffer: np.ndarray | None = None
        self.past_repetition_penalty_buffer_retained_state: np.ndarray | None = None
        self.past_presence_penalty_buffer_retained_state: np.ndarray | None = None
        self.decode_next_tokens_by_k: dict[int, np.ndarray] | None = None
        self.ods_step_prefill_next_tokens: np.ndarray | None = None
        self.ods_step_num_decodes: int = 0
        self.ods_step_decode_mdt: int = 1
        self.decode_probs_by_k: dict[int, np.ndarray] | None = None
        self.prefill_probs: np.ndarray | None = None
        # Debug/eval-only ODS probability capture (full vocab transfer each step);
        # not part of request/response sampling output, and unsuitable for
        # production throughput/latency paths.
        self.ods_debug_last_decode_probs: np.ndarray | None = None
        self.ods_debug_last_prefill_probs: np.ndarray | None = None
        self.decode_batch_inputs_by_k: dict[int, dict] = {}
        for _k in self.decode_ks:
            _mdt = _k + 1
            _d: dict = {
                "input_ids": np.full((self.decode_bsz, _mdt), -1, dtype=np.int64),
                "position_ids": np.full((self.decode_bsz, _mdt), -1, dtype=np.int64),
                "batch_index": np.full((self.decode_bsz, 1), -1, dtype=np.int64),
            }
            if self.on_device_sampling_en:
                _d.update(
                    {
                        "temperatures": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                        "top_ks": np.zeros((self.decode_bsz, 1), dtype=np.int32),
                        "top_ps": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                        "min_ps": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                        "repetition_penalties": np.zeros(
                            (self.decode_bsz, 1), dtype=np.float32
                        ),
                        "presence_penalties": np.zeros(
                            (self.decode_bsz, 1), dtype=np.float32
                        ),
                        "random_numbers": np.zeros(
                            (self.decode_bsz, self.ods_max_top_k_ids), dtype=np.float32
                        ),
                    }
                )
            if self.lora_mode:
                _d["lora_ids"] = np.full((self.decode_bsz, 1), -1, dtype=np.int64)
            self.decode_batch_inputs_by_k[_k] = _d

        if self.on_device_sampling_en:
            self.prefill_ods_inputs = {
                "temperatures": np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                "top_ks": np.zeros((self.prefill_bsz, 1), dtype=np.int32),
                "top_ps": np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                "min_ps": np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                "repetition_penalties": np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                "presence_penalties": np.zeros((self.prefill_bsz, 1), dtype=np.float32),
                "random_numbers": np.zeros(
                    (self.prefill_bsz, self.ods_max_top_k_ids), dtype=np.float32
                ),
            }

        # Backward-compat alias pointing at the max-K input dict.
        self.decode_batch_inputs = self.decode_batch_inputs_by_k[self.decode_ks[-1]]
        self.list_of_comp_ctx_lengths: dict[int, np.ndarray] | None = None
        # Async scheduling
        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling
        # Pooling
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"
        # Multimodal
        self.uses_mrope = model_config.uses_mrope
        self.is_vision_encoder = False
        self.is_multimodal_model = vllm_config.model_config.is_multimodal_model

    def forward(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        is_prompt: bool,
        lora_ids: np.ndarray | None = None,
        sampling_params: dict[str, np.ndarray] | None = None,
        kv_caches: list[list[np.ndarray]] | None = None,
        callback: Callable | None = None,
        mm_kwargs_list: list[dict] | None = None,
        prefill_is_partial: list[bool] | None = None,
        prefill_cum_sum: np.ndarray | None = None,
        logits: np.ndarray | None = None,
        num_prompt_tokens_prefill: np.ndarray | None = None,
    ) -> Queue | None:
        if self.is_pooling_model and not self.is_multimodal_model:
            output = self._run_encode(prefill_cum_sum, input_ids, positions)
            return output
        elif is_prompt and (self.disagg_producer_en or self.disagg_serving_en):
            assert kv_caches is not None
            if self.disagg_producer_en:
                assert prefill_is_partial is not None
                pending_prefill_exec_queue: Queue[Any] = Queue()
                self._run_pipeline_prefill(
                    input_ids,
                    positions,
                    batch_indices,
                    prefill_cum_sum,
                    pending_prefill_exec_queue,
                    kv_caches,
                    prefill_is_partial,
                    lora_ids,
                    callback,
                    mm_kwargs_list,
                    logits,
                    num_prompt_tokens_prefill,
                )
                return pending_prefill_exec_queue
            else:
                self._kv_handoff(batch_indices, kv_caches)
        else:
            with lock:  # TODO: Re-evaluate if needed for MultiProcExecutor
                if is_prompt:
                    pending_prefill_exec_queue = Queue()
                    self._run_prefill(
                        input_ids,
                        positions,
                        batch_indices,
                        prefill_cum_sum,
                        pending_prefill_exec_queue,
                        logits,
                        lora_ids,
                        mm_kwargs_list,
                        sampling_params,
                    )
                    return pending_prefill_exec_queue
                else:
                    self._run_decode(
                        input_ids,
                        positions,
                        batch_indices,
                        logits,
                        lora_ids=lora_ids,
                        sampling_params=sampling_params,
                        callback=callback,
                    )
        return None

    @property
    def pooler(self):
        if self._pooler is None:
            raise ValueError("Pooler is not initialized")
        return self._pooler

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata | None = None,
    ) -> torch.Tensor:
        # QAIC runtime returns logits directly
        return hidden_states

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if self.on_device_sampling_en:
            if self.decode_next_tokens_by_k is None:
                raise ValueError(
                    "On-device sampling is enabled but decode `next_tokens` buffers "
                    "are uninitialized."
                )

            current_k = self.active_k
            decode_next_tokens = self.decode_next_tokens_by_k.get(current_k)
            if decode_next_tokens is None:
                raise ValueError(
                    "On-device sampling decode output buffer missing for decode "
                    f"specialization K={current_k}."
                )

            num_decode_reqs = self.ods_step_num_decodes
            decode_mdt = self.ods_step_decode_mdt if num_decode_reqs > 0 else 1
            if num_decode_reqs > 0:
                if decode_mdt <= 0:
                    raise ValueError(
                        "Invalid decode token count per request for ODS step: "
                        f"{decode_mdt}."
                    )
                decode_token_ids = decode_next_tokens[:num_decode_reqs, :decode_mdt, 0]
            else:
                decode_token_ids = np.empty((0, 1), dtype=np.int64)

            prefill_token_ids = self.ods_step_prefill_next_tokens
            if prefill_token_ids is None:
                prefill_token_ids = np.empty(
                    (0, decode_token_ids.shape[1]), dtype=decode_token_ids.dtype
                )

            if (
                decode_token_ids.size > 0
                and prefill_token_ids.size > 0
                and decode_token_ids.shape[1] != prefill_token_ids.shape[1]
            ):
                raise ValueError(
                    "Cannot combine ODS decode/prefill sampled tokens with different "
                    "token counts per request: "
                    f"decode={decode_token_ids.shape[1]}, "
                    f"prefill={prefill_token_ids.shape[1]}."
                )

            sampled_token_ids_np = np.concatenate(
                (decode_token_ids, prefill_token_ids), axis=0
            )
            # temperature can be None on vLLM's all_greedy/no_top_k/no_top_p
            # batch-level fast path, while repetition_penalties is always set.
            expected_num_reqs = int(sampling_metadata.repetition_penalties.shape[0])
            if sampled_token_ids_np.shape[0] != expected_num_reqs:
                raise ValueError(
                    "ODS sampled token count mismatch: expected "
                    f"{expected_num_reqs} requests from sampling metadata, got "
                    f"{sampled_token_ids_np.shape[0]} from accelerator outputs."
                )

            sampled_token_ids = torch.from_numpy(
                sampled_token_ids_np.astype(np.int64, copy=False)
            )
            return SamplerOutput(sampled_token_ids, None)

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_comp_ctx_lengths(self):
        comp_ctx_lengths_prefill, comp_ctx_lengths_decode = [], []
        if "comp_ctx_lengths" not in self.session.binding_index_map:
            return None, None
        input_idx = self.session.binding_index_map["input_ids"]
        ccl_idx = self.session.binding_index_map["comp_ctx_lengths"]
        for i in range(len(self.session.allowed_shapes)):
            if self.session.allowed_shapes[i][input_idx][1][1] == self.prefill_seq_len:
                comp_ctx_lengths_prefill.append(
                    self.session.allowed_shapes[i][ccl_idx][1][0]
                )
            elif (
                self.session.allowed_shapes[i][input_idx][1][1] == 1
                or self.num_logits_to_keep
            ):
                comp_ctx_lengths_decode.append(
                    self.session.allowed_shapes[i][ccl_idx][1][0]
                )
            else:
                raise ValueError("QPC not compiled for required seq_len")

        comp_ctx_lengths_prefill.sort()
        comp_ctx_lengths_decode.sort()
        if comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
            self.list_of_comp_ctx_lengths = {
                comp_ctx_len: np.empty(comp_ctx_len, dtype=np.int64)
                for comp_ctx_len in comp_ctx_lengths_prefill + comp_ctx_lengths_decode
            }
        return comp_ctx_lengths_prefill, comp_ctx_lengths_decode

    def _decode_ks_from_session(self) -> list[int]:
        """Derive which K values are compiled in this QPC from session.allowed_shapes.

        Reads the ``input_ids`` binding shapes and collects every distinct
        ``seq_len`` value that does NOT match ``prefill_seq_len``.  Each such
        value ``s`` corresponds to a decode specialization for ``K = s - 1``.

        Falls back to ``self.decode_ks`` (set at init time from the method name)
        if allowed_shapes is unavailable or empty.  This makes the feature
        transparent: old QPCs compiled without a K=0 spec continue to work
        unchanged because ``decode_ks`` stays at ``[max_k]``.
        """
        try:
            input_idx = self.session.binding_index_map["input_ids"]
            ks: list[int] = []
            for i in range(len(self.session.allowed_shapes)):
                s = self.session.allowed_shapes[i][input_idx][1][1]
                if s != self.prefill_seq_len:
                    k = s - 1
                    if k not in ks:
                        ks.append(k)
            if ks:
                return sorted(ks)
        except Exception:
            pass
        return list(self.decode_ks)

    def load_model(
        self,
        qpc_path: str,
        device_id: list,
        num_logits_to_keep: int | None = None,
        stages: int | None = 1,
        kv_transfer_role: str | None = None,
    ) -> None:
        self.qpc_path: str = qpc_path
        self.device_id: list = device_id
        self.num_logits_to_keep = num_logits_to_keep
        self.stages: int = stages if stages is not None else 1
        self.disagg_serving_en = kv_transfer_role is not None
        self.disagg_producer_en = kv_transfer_role == "kv_producer"

        logger.info("Loading QPC...")
        logger.info(
            "This may take some time, please don't press CTRL-C during this phase..."
        )
        s = time.perf_counter()
        cluster_id = None
        if self.disagg_serving_en:
            cluster_id = "prefill" if self.disagg_producer_en else "decode"
        else:
            self.stages = 1
        from .qaic_session_np import QAICInferenceSession

        self.session = QAICInferenceSession(
            qpc_path,
            full_batch_size=self.full_batch_size,
            device_ids=device_id,
            stages=self.stages,
            cluster_id=cluster_id,
            use_async_scheduling=self.use_async_scheduling,
        )
        # ODS-compiled QPCs (including the aic_return_pdfs debug sub-mode) never
        # expose a `logits` binding, so querying it here is unnecessary and
        # produces a spurious qaicrt binding-not-found warning. self.logits_ndim
        # is only consumed by non-ODS disaggregated-serving warmup code below.
        self.logits_ndim = (
            3 if self.on_device_sampling_en else self.session.get_logits_ndim()
        )
        _check_ods_binding_mismatch(
            engine_expects_ods=self.on_device_sampling_en,
            binding_index_map=self.session.binding_index_map,
            engine_expects_debug_probs=self.debug_return_probs_en,
        )

        e = time.perf_counter() - s
        logger.info("Successfully loaded QPC in %s secs", e)

        if self.is_multimodal_model:
            self._load_multimodal()
            if self.is_vision_encoder:
                # The vision encoder runs standalone, outside the prefill/decode
                # paradigm, so remaining configs do not apply.
                return

        self.comp_ctx_lengths_prefill, self.comp_ctx_lengths_decode = (
            self.get_comp_ctx_lengths()
        )
        self.prefill_num_logits_buffer = None
        if not self.on_device_sampling_en:
            # ODS-compiled QPCs perform sampling on-device and can omit the
            # `logits` binding entirely; registering this buffer at load time is
            # unnecessary and causes spurious qaicrt binding-not-found warnings.
            self.prefill_logits = dict(
                logits=np.random.randn(self.prefill_bsz, 1, self.vocab_size).astype(
                    np.float32
                )
            )
            self.session.set_buffers(self.prefill_logits)
            self.batch_prefill_logits = np.empty(
                (self.decode_bsz, self.vocab_size), dtype=np.float32
            )
        else:
            self.prefill_logits = None
            self.batch_prefill_logits = None
        self.decode_num_logits_buffer = None
        if self.num_logits_to_keep is not None:
            self.is_spec_decode_target_model = True
            # Derive which K values are actually compiled in this QPC.
            # Falls back to the init-time decode_ks if allowed_shapes is unavailable.
            self.decode_ks = self._decode_ks_from_session()
            self.active_k = self.decode_ks[-1]

            # Ensure decode_batch_inputs_by_k covers every K in the (possibly
            # updated) decode_ks — _run_decode indexes it by current_k.
            for _k in self.decode_ks:
                if _k not in self.decode_batch_inputs_by_k:
                    _mdt = _k + 1
                    _d: dict = {
                        "input_ids": np.full(
                            (self.decode_bsz, _mdt), -1, dtype=np.int64
                        ),
                        "position_ids": np.full(
                            (self.decode_bsz, _mdt), -1, dtype=np.int64
                        ),
                        "batch_index": np.full(
                            (self.decode_bsz, 1), -1, dtype=np.int64
                        ),
                    }
                    if self.lora_mode:
                        _d["lora_ids"] = np.full(
                            (self.decode_bsz, 1), -1, dtype=np.int64
                        )
                    if self.on_device_sampling_en:
                        _d.update(
                            {
                                "temperatures": np.zeros(
                                    (self.decode_bsz, 1), dtype=np.float32
                                ),
                                "top_ks": np.zeros((self.decode_bsz, 1), dtype=np.int32),
                                "top_ps": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                                "min_ps": np.zeros((self.decode_bsz, 1), dtype=np.float32),
                                "repetition_penalties": np.zeros(
                                    (self.decode_bsz, 1), dtype=np.float32
                                ),
                                "presence_penalties": np.zeros(
                                    (self.decode_bsz, 1), dtype=np.float32
                                ),
                                "random_numbers": np.zeros(
                                    (self.decode_bsz, self.ods_max_top_k_ids),
                                    dtype=np.float32,
                                ),
                            }
                        )
                    self.decode_batch_inputs_by_k[_k] = _d

            # Pre-allocate per-K logit output buffers.
            self.decode_logits_by_k: dict[int, dict] = {}
            self.decode_num_logits_buffer_by_k: dict[int, dict] = {}
            for _k in self.decode_ks:
                _mdt = _k + 1
                self.decode_logits_by_k[_k] = dict(
                    logits=np.random.randn(
                        self.decode_bsz, _mdt, self.vocab_size
                    ).astype(np.float32)
                )
                self.decode_num_logits_buffer_by_k[_k] = dict(
                    num_logits_to_keep=np.zeros((_mdt, 1), np.int64)
                )

            # Backward-compat aliases pointing at max-K buffers.
            self.decode_logits = self.decode_logits_by_k[self.decode_ks[-1]]
            self.decode_num_logits_buffer = self.decode_num_logits_buffer_by_k[
                self.decode_ks[-1]
            ]
            self.prefill_num_logits_buffer = dict(
                num_logits_to_keep=np.zeros((1, 1), np.int64)
            )
            self.session.set_buffers(self.prefill_num_logits_buffer)
        else:
            if not self.on_device_sampling_en:
                self.decode_logits = dict(
                    logits=np.random.randn(self.decode_bsz, 1, self.vocab_size).astype(
                        np.float32
                    )
                )
            else:
                self.decode_logits = None

        if self.on_device_sampling_en:
            probs_info = None
            penalty_repetition_info = self.get_io_shape_and_dtype(
                "past_repetition_penalty_buffer", is_input=True
            )
            if penalty_repetition_info is None:
                raise ValueError(
                    "On-device sampling requires `past_repetition_penalty_buffer` "
                    "in QPC inputs, but this binding was not found at startup."
                )
            penalty_presence_info = self.get_io_shape_and_dtype(
                "past_presence_penalty_buffer", is_input=True
            )
            if penalty_presence_info is None:
                raise ValueError(
                    "On-device sampling requires `past_presence_penalty_buffer` "
                    "in QPC inputs, but this binding was not found at startup."
                )
            penalty_repetition_retained_info = self.get_io_shape_and_dtype(
                "past_repetition_penalty_buffer_RetainedState", is_input=False
            )
            if penalty_repetition_retained_info is None:
                raise ValueError(
                    "On-device sampling requires "
                    "`past_repetition_penalty_buffer_RetainedState` in QPC outputs, "
                    "but this binding was not found at startup."
                )
            penalty_presence_retained_info = self.get_io_shape_and_dtype(
                "past_presence_penalty_buffer_RetainedState", is_input=False
            )
            if penalty_presence_retained_info is None:
                raise ValueError(
                    "On-device sampling requires "
                    "`past_presence_penalty_buffer_RetainedState` in QPC outputs, "
                    "but this binding was not found at startup."
                )

            penalty_shape = tuple(penalty_repetition_info[0])
            penalty_dtype = penalty_repetition_info[1]
            if len(penalty_shape) != 2:
                raise ValueError(
                    "On-device sampling `past_repetition_penalty_buffer` shape "
                    f"must be rank-2, got {penalty_shape}."
                )
            if penalty_shape[0] != self.full_batch_size:
                raise ValueError(
                    "On-device sampling penalty-history buffer batch dimension "
                    "mismatch: expected full_batch_size="
                    f"{self.full_batch_size}, got {penalty_shape[0]}."
                )
            if tuple(penalty_presence_info[0]) != penalty_shape:
                raise ValueError(
                    "On-device sampling penalty-history input shape mismatch: "
                    "`past_repetition_penalty_buffer` and "
                    "`past_presence_penalty_buffer` must match."
                )
            if penalty_presence_info[1] != penalty_dtype:
                raise ValueError(
                    "On-device sampling penalty-history input dtype mismatch: "
                    "`past_repetition_penalty_buffer` and "
                    "`past_presence_penalty_buffer` must match."
                )
            if tuple(penalty_repetition_retained_info[0]) != penalty_shape:
                raise ValueError(
                    "On-device sampling penalty-history retained-state shape "
                    "mismatch: `past_repetition_penalty_buffer_RetainedState` does "
                    "not match `past_repetition_penalty_buffer`."
                )
            if tuple(penalty_presence_retained_info[0]) != penalty_shape:
                raise ValueError(
                    "On-device sampling penalty-history retained-state shape "
                    "mismatch: `past_presence_penalty_buffer_RetainedState` does "
                    "not match `past_presence_penalty_buffer`."
                )
            if penalty_repetition_retained_info[1] != penalty_dtype:
                raise ValueError(
                    "On-device sampling penalty-history retained-state dtype mismatch: "
                    "`past_repetition_penalty_buffer_RetainedState` does not match "
                    "`past_repetition_penalty_buffer`."
                )
            if penalty_presence_retained_info[1] != penalty_dtype:
                raise ValueError(
                    "On-device sampling penalty-history retained-state dtype mismatch: "
                    "`past_presence_penalty_buffer_RetainedState` does not match "
                    "`past_presence_penalty_buffer`."
                )

            next_tokens_info = self.get_io_shape_and_dtype(
                "next_tokens", is_input=False
            )
            if next_tokens_info is None:
                raise ValueError(
                    "On-device sampling requires `next_tokens` in QPC outputs, "
                    "but this binding was not found at startup."
                )
            if self.debug_return_probs_en:
                probs_info = self.get_io_shape_and_dtype("probs", is_input=False)
                if probs_info is None:
                    raise ValueError(
                        "On-device sampling debug/evaluation sub-mode requires "
                        "`probs` in QPC outputs, but this binding was not found at "
                        "startup. The loaded QPC was likely not compiled with "
                        "return_pdfs=True."
                    )

            random_numbers_info = self.get_io_shape_and_dtype(
                "random_numbers", is_input=True
            )
            if random_numbers_info is not None:
                random_numbers_shape = tuple(random_numbers_info[0])
                if len(random_numbers_shape) != 2:
                    raise ValueError(
                        "On-device sampling `random_numbers` binding shape mismatch: "
                        "expected rank-2 with shape `(bs, ods_max_top_k_ids)`, "
                        f"but loaded QPC compiled shape={random_numbers_shape}."
                    )
                random_numbers_width = random_numbers_shape[-1]
                if random_numbers_width != self.ods_max_top_k_ids:
                    raise ValueError(
                        "On-device sampling `random_numbers` binding width mismatch: "
                        f"configured ods_max_top_k_ids={self.ods_max_top_k_ids}, "
                        f"but loaded QPC compiled width={random_numbers_width}. "
                        "Reconcile deployment `max_top_k_ids` with the loaded QPC "
                        "compilation settings."
                    )

                random_numbers_actual_dtype = np.dtype(random_numbers_info[1])
                random_numbers_expected_dtype = np.dtype(np.float32)
                if random_numbers_actual_dtype != random_numbers_expected_dtype:
                    raise ValueError(
                        "On-device sampling `random_numbers` binding dtype mismatch: "
                        "backend allocates "
                        f"dtype={random_numbers_expected_dtype.name}, but loaded "
                        "QPC compiled "
                        f"dtype={random_numbers_actual_dtype.name}. Reconcile "
                        "deployment sampling-input dtype assumptions with the loaded "
                        "QPC compilation settings."
                    )

                random_numbers_idx = self.session.binding_index_map.get("random_numbers")
                if random_numbers_idx is None:
                    raise ValueError(
                        "On-device sampling binding index lookup failed for "
                        "`random_numbers` at startup."
                    )

                input_idx = self.session.binding_index_map["input_ids"]
                for i in range(len(self.session.allowed_shapes)):
                    seq_len = self.session.allowed_shapes[i][input_idx][1][1]
                    actual_batch_size = self.session.allowed_shapes[i][random_numbers_idx][1][0]
                    if seq_len == self.prefill_seq_len:
                        if actual_batch_size != self.prefill_bsz:
                            raise ValueError(
                                "On-device sampling prefill binding batch-size mismatch: "
                                "`random_numbers` specialization row="
                                f"{i}, expected prefill_bsz={self.prefill_bsz}, "
                                f"got {actual_batch_size}."
                            )
                    elif actual_batch_size != self.decode_bsz:
                        raise ValueError(
                            "On-device sampling decode binding batch-size mismatch: "
                            "`random_numbers` specialization row="
                            f"{i}, expected decode_bsz={self.decode_bsz}, "
                            f"got {actual_batch_size}."
                        )

            ods_input_binding_expected_dtypes = {
                "temperatures": np.float32,
                "top_ks": np.int32,
                "top_ps": np.float32,
                "min_ps": np.float32,
                "repetition_penalties": np.float32,
                "presence_penalties": np.float32,
            }
            for binding_name, expected_dtype in (
                ods_input_binding_expected_dtypes.items()
            ):
                binding_info = self.get_io_shape_and_dtype(binding_name, is_input=True)
                if binding_info is None:
                    raise ValueError(
                        f"On-device sampling requires `{binding_name}` in QPC inputs, "
                        "but this binding was not found at startup."
                    )

                binding_shape = tuple(binding_info[0])
                if len(binding_shape) != 2 or binding_shape[-1] != 1:
                    raise ValueError(
                        f"On-device sampling `{binding_name}` binding shape mismatch: "
                        "expected rank-2 with trailing shape `(bs, 1)`, "
                        f"but loaded QPC compiled shape={binding_shape}."
                    )

                binding_idx = self.session.binding_index_map.get(binding_name)
                if binding_idx is None:
                    raise ValueError(
                        "On-device sampling binding index lookup failed for "
                        f"`{binding_name}` at startup."
                    )

                # Validate per-specialization batch-size dims from
                # session.allowed_shapes using the same row-classification idiom
                # as get_comp_ctx_lengths/_decode_ks_from_session.
                input_idx = self.session.binding_index_map["input_ids"]
                for i in range(len(self.session.allowed_shapes)):
                    seq_len = self.session.allowed_shapes[i][input_idx][1][1]
                    actual_batch_size = self.session.allowed_shapes[i][binding_idx][1][0]
                    if seq_len == self.prefill_seq_len:
                        if actual_batch_size != self.prefill_bsz:
                            raise ValueError(
                                "On-device sampling prefill binding batch-size mismatch: "
                                f"`{binding_name}` specialization row={i}, "
                                f"expected prefill_bsz={self.prefill_bsz}, "
                                f"got {actual_batch_size}."
                            )
                    elif actual_batch_size != self.decode_bsz:
                        raise ValueError(
                            "On-device sampling decode binding batch-size mismatch: "
                            f"`{binding_name}` specialization row={i}, "
                            f"expected decode_bsz={self.decode_bsz}, "
                            f"got {actual_batch_size}."
                        )

                actual_dtype = np.dtype(binding_info[1])
                expected_np_dtype = np.dtype(expected_dtype)
                if actual_dtype != expected_np_dtype:
                    raise ValueError(
                        f"On-device sampling `{binding_name}` binding dtype mismatch: "
                        f"backend allocates dtype={expected_np_dtype.name}, but "
                        f"loaded QPC compiled dtype={actual_dtype.name}. Reconcile "
                        "deployment sampling-input dtype assumptions with the loaded "
                        "QPC compilation settings."
                    )

            last_accepted_output_tokens_info = self.get_io_shape_and_dtype(
                "last_accepted_output_tokens", is_input=True
            )
            if last_accepted_output_tokens_info is None:
                raise ValueError(
                    "On-device sampling requires `last_accepted_output_tokens` in "
                    "QPC inputs, but this binding was not found at startup."
                )

            last_accepted_output_tokens_shape = tuple(last_accepted_output_tokens_info[0])
            if (
                len(last_accepted_output_tokens_shape) != 2
                or last_accepted_output_tokens_shape[-1] <= 0
            ):
                raise ValueError(
                    "On-device sampling `last_accepted_output_tokens` binding shape "
                    "mismatch: expected rank-2 with a positive trailing "
                    "dimension, "
                    "but loaded QPC compiled "
                    f"shape={last_accepted_output_tokens_shape}."
                )

            last_accepted_output_tokens_actual_dtype = np.dtype(
                last_accepted_output_tokens_info[1]
            )
            last_accepted_output_tokens_expected_dtype = np.dtype(np.int64)
            if (
                last_accepted_output_tokens_actual_dtype
                != last_accepted_output_tokens_expected_dtype
            ):
                raise ValueError(
                    "On-device sampling `last_accepted_output_tokens` binding dtype "
                    "mismatch: backend allocates "
                    f"dtype={last_accepted_output_tokens_expected_dtype.name}, but "
                    "loaded QPC compiled "
                    f"dtype={last_accepted_output_tokens_actual_dtype.name}. "
                    "Reconcile deployment sampling-input dtype assumptions with the "
                    "loaded QPC compilation settings."
                )

            last_accepted_output_tokens_idx = self.session.binding_index_map.get(
                "last_accepted_output_tokens"
            )
            if last_accepted_output_tokens_idx is None:
                raise ValueError(
                    "On-device sampling binding index lookup failed for "
                    "`last_accepted_output_tokens` at startup."
                )

            input_idx = self.session.binding_index_map["input_ids"]
            for i in range(len(self.session.allowed_shapes)):
                seq_len = self.session.allowed_shapes[i][input_idx][1][1]
                actual_batch_size = self.session.allowed_shapes[i][
                    last_accepted_output_tokens_idx
                ][1][0]
                if seq_len == self.prefill_seq_len:
                    if actual_batch_size != self.prefill_bsz:
                        raise ValueError(
                            "On-device sampling prefill binding batch-size mismatch: "
                            "`last_accepted_output_tokens` specialization row="
                            f"{i}, expected prefill_bsz={self.prefill_bsz}, "
                            f"got {actual_batch_size}."
                        )
                elif actual_batch_size != self.decode_bsz:
                    raise ValueError(
                        "On-device sampling decode binding batch-size mismatch: "
                        "`last_accepted_output_tokens` specialization row="
                        f"{i}, expected decode_bsz={self.decode_bsz}, "
                        f"got {actual_batch_size}."
                    )

            penalty_history_buffers: dict[str, np.ndarray] = {}
            self.session.create_numpy_penalty_buffers(
                penalty_history_buffers,
                direction="in",
                shape=penalty_shape,
                dtype=penalty_dtype,
            )
            self.session.create_numpy_penalty_buffers(
                penalty_history_buffers,
                direction="out",
                shape=penalty_shape,
                dtype=penalty_dtype,
            )

            # Keep penalty-history input and *_RetainedState output as distinct
            # host arrays. Each step explicitly copies prior retained-state output
            # back into the next step's input buffer via np.copyto in prefill/decode,
            # avoiding reliance on unverified SDK-level aliasing behavior.

            self.past_repetition_penalty_buffer = penalty_history_buffers[
                "past_repetition_penalty_buffer"
            ]
            self.past_presence_penalty_buffer = penalty_history_buffers[
                "past_presence_penalty_buffer"
            ]
            self.past_repetition_penalty_buffer_retained_state = (
                penalty_history_buffers[
                    "past_repetition_penalty_buffer_RetainedState"
                ]
            )
            self.past_presence_penalty_buffer_retained_state = penalty_history_buffers[
                "past_presence_penalty_buffer_RetainedState"
            ]

            for _k in self.decode_ks:
                self.decode_batch_inputs_by_k[_k].update(
                    {
                        "past_repetition_penalty_buffer": self.past_repetition_penalty_buffer,
                        "past_presence_penalty_buffer": self.past_presence_penalty_buffer,
                        "past_repetition_penalty_buffer_RetainedState": self.past_repetition_penalty_buffer_retained_state,
                        "past_presence_penalty_buffer_RetainedState": self.past_presence_penalty_buffer_retained_state,
                    }
                )

            if self.prefill_ods_inputs is None:
                raise ValueError(
                    "On-device sampling is enabled but prefill sampling-control "
                    "buffers are uninitialized."
                )
            self.prefill_ods_inputs.update(
                {
                    "past_repetition_penalty_buffer": self.past_repetition_penalty_buffer,
                    "past_presence_penalty_buffer": self.past_presence_penalty_buffer,
                    "past_repetition_penalty_buffer_RetainedState": self.past_repetition_penalty_buffer_retained_state,
                    "past_presence_penalty_buffer_RetainedState": self.past_presence_penalty_buffer_retained_state,
                }
            )

            last_accepted_output_tokens_width = last_accepted_output_tokens_info[0][-1]
            last_accepted_output_tokens_dtype = last_accepted_output_tokens_info[1]
            self.prefill_last_accepted_output_tokens = np.zeros(
                (self.prefill_bsz, last_accepted_output_tokens_width),
                dtype=last_accepted_output_tokens_dtype,
            )

            next_tokens_dtype = next_tokens_info[1]
            self.decode_next_tokens_by_k = {}
            for _k in self.decode_ks:
                _mdt = _k + 1
                expected_next_tokens_shape = (self.decode_bsz, _mdt, 1)
                self.session.create_output_buffers(
                    self.decode_batch_inputs_by_k[_k],
                    expected_next_tokens_shape,
                    next_tokens_dtype,
                    buffer_name="next_tokens",
                )
                if "next_tokens" not in self.decode_batch_inputs_by_k[_k]:
                    raise ValueError(
                        "Failed to register `next_tokens` decode output buffer for "
                        f"decode specialization K={_k}."
                    )
                registered_next_tokens = self.decode_batch_inputs_by_k[_k]["next_tokens"]
                if tuple(registered_next_tokens.shape) != expected_next_tokens_shape:
                    raise ValueError(
                        "On-device sampling `next_tokens` decode output buffer shape "
                        "mismatch for decode specialization "
                        f"K={_k}: expected {expected_next_tokens_shape}, got "
                        f"{tuple(registered_next_tokens.shape)}."
                    )
                self.decode_next_tokens_by_k[_k] = registered_next_tokens

            prefill_next_tokens_buffer: dict[str, np.ndarray] = {}
            self.session.create_output_buffers(
                prefill_next_tokens_buffer,
                (self.prefill_bsz, 1, 1),
                next_tokens_dtype,
                buffer_name="next_tokens",
            )
            if "next_tokens" not in prefill_next_tokens_buffer:
                raise ValueError("Failed to register `next_tokens` prefill output buffer.")
            self.prefill_next_tokens = prefill_next_tokens_buffer["next_tokens"]

            if self.debug_return_probs_en:
                if probs_info is None:
                    raise ValueError(
                        "On-device sampling debug/evaluation sub-mode expected "
                        "`probs` output metadata at startup, but none was available."
                    )
                probs_dtype = probs_info[1]
                self.decode_probs_by_k = {}
                for _k in self.decode_ks:
                    _mdt = _k + 1
                    expected_probs_shape = (self.decode_bsz, _mdt, self.vocab_size)
                    self.session.create_output_buffers(
                        self.decode_batch_inputs_by_k[_k],
                        expected_probs_shape,
                        probs_dtype,
                        buffer_name="probs",
                    )
                    if "probs" not in self.decode_batch_inputs_by_k[_k]:
                        raise ValueError(
                            "Failed to register `probs` decode output buffer for "
                            f"decode specialization K={_k}."
                        )
                    registered_probs = self.decode_batch_inputs_by_k[_k]["probs"]
                    if tuple(registered_probs.shape) != expected_probs_shape:
                        raise ValueError(
                            "On-device sampling `probs` decode output buffer shape "
                            "mismatch for decode specialization "
                            f"K={_k}: expected {expected_probs_shape}, got "
                            f"{tuple(registered_probs.shape)}."
                        )
                    self.decode_probs_by_k[_k] = registered_probs

                prefill_probs_buffer: dict[str, np.ndarray] = {}
                self.session.create_output_buffers(
                    prefill_probs_buffer,
                    (self.prefill_bsz, 1, self.vocab_size),
                    probs_dtype,
                    buffer_name="probs",
                )
                if "probs" not in prefill_probs_buffer:
                    raise ValueError("Failed to register `probs` prefill output buffer.")
                self.prefill_probs = prefill_probs_buffer["probs"]
        # CCL state for prefill: dict keyed by prefill exec-object slot id, value
        # is the bucket currently in flight on that slot (0 = idle). Used by
        # _run_pipeline_prefill to pick one CCL per batch =
        # max(needed-by-this-batch, max-in-flight).
        if self.comp_ctx_lengths_prefill:
            prefill_start = self.session.decode_num_execObj
            self.active_ccl: dict[int, int] = {
                i: 0
                for i in range(
                    prefill_start,
                    prefill_start + self.session.prefill_num_execObj,
                )
            }
        if "batch_index" in self.session.input_names:
            self.ignore_batch_index = False
        else:
            self.ignore_batch_index = True
            self.decode_batch_inputs.pop("batch_index", None)
        if self.disagg_producer_en:
            self.decode_bsz = 0

    def get_allowed_seqlens(self):
        allowed_seqlens = []
        for allowed_shape in self.session.allowed_shapes:
            allowed_seqlens.append(allowed_shape[1][1][1])
        return allowed_seqlens

    @property
    def kv_cache_info(self) -> list[tuple]:  # [(kv_shape, kv_type, kv_size)]
        return self.session.kv_cache_info

    def get_io_shape_and_dtype(
        self, buffer_name: str, is_input: bool = True
    ) -> tuple[list, np.dtype, int] | None:
        from .qaic_session_np import aic_to_np_dtype_mapping

        if is_input and buffer_name not in self.session.input_names:
            return None
        if not is_input and buffer_name not in self.session.output_names:
            return None
        binding = self.session.bindings[self.session.binding_index_map[buffer_name]]
        return (
            list(binding.dims),
            aic_to_np_dtype_mapping[binding.type],
            binding.size,
        )

    def _kv_handoff(
        self,
        batch_indices: list[int],
        kv_caches: list[list[np.ndarray]],
    ):
        for _, bidx in enumerate(batch_indices):
            if kv_caches[bidx] and len(kv_caches[bidx]) > 0:
                assert len(self.kv_cache_info) == len(kv_caches[bidx]), (
                    f"KV buffer count mismatch: "
                    f"expected {len(self.kv_cache_info)} buffers, "
                    f"got {len(kv_caches[bidx])}"
                )
                for buff_idx in range(len(kv_caches[bidx])):
                    buf = kv_caches[bidx][buff_idx]
                    target_heads = self.kv_cache_info[buff_idx][0][1]
                    # broadcast MLA buffers across num heads (zero-copy)
                    if buf.size > 0 and target_heads != buf.shape[1]:
                        assert buf.shape[1] == 1, (
                            f"MLA KV head expansion expects num_heads=1 in received "
                            f"buffer, got {buf.shape[1]} (shape={buf.shape})"
                        )
                        kv_caches[bidx][buff_idx] = np.broadcast_to(
                            buf, (buf.shape[0], target_heads) + buf.shape[2:]
                        )

                # Update kv cache setDataWith
                _ = self.session.set_data_for_kv_handoff(
                    kv_caches[bidx],
                    [("batch_index", bidx), ("ctx_start", 0)],
                    self.decode_execObj_idx,
                    self.session.decode_buff_map,
                )
        return

    def _run_pipeline_prefill(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        prefill_cum_sum: np.ndarray,
        pending_exec_queue: Queue,
        kv_caches: list[list[np.ndarray]],
        prefill_is_partial: list[bool],
        lora_ids: list[int] | None = None,
        callback: Callable | None = None,
        mm_kwargs_list: list[dict] | None = None,
        logits: np.ndarray | None = None,
        num_prompt_tokens_prefill: np.ndarray | None = None,
    ):
        pending_exec_count = 0  # in-flight executions in current batch
        # set qpc prefill state
        if self.last_decode:
            self.last_decode = False

        # One CCL bucket for the batch, keyed off each request's TOTAL prompt
        # length (final position) so every chunk of a request keeps one bucket
        # across steps — positions.max() would only see the current chunk.
        # Bumped to the max of what's currently in flight on prefill exec-object
        # slots so all simultaneously-in-flight chunks share one CCL.
        comp_ctx_val = None
        chosen_ccl = 0
        if self.comp_ctx_lengths_prefill is not None:
            assert num_prompt_tokens_prefill is not None
            assert self.list_of_comp_ctx_lengths is not None
            batch_max_position = int(num_prompt_tokens_prefill.max()) - 1
            chosen_ccl = self.comp_ctx_lengths_prefill[-1]
            for ccl in self.comp_ctx_lengths_prefill:
                if batch_max_position < ccl:
                    chosen_ccl = ccl
                    break
            chosen_ccl = max(chosen_ccl, max(self.active_ccl.values()))
            comp_ctx_val = self.list_of_comp_ctx_lengths[chosen_ccl]

        idx_start = 0
        for index, idx_end in enumerate(prefill_cum_sum):
            # extract indices of specific request
            iids = input_ids[idx_start:idx_end]
            pids = positions[..., idx_start:idx_end]
            idx_start = idx_end
            # concatenate to multiple of `self.prefill_seq_len`
            # to avoid reading/writing KV$ on `num_pads` tokens
            n_prompt_tokens = iids.shape[-1]
            if (remainder := n_prompt_tokens % self.prefill_seq_len) > 0:
                num_pads = self.prefill_seq_len - remainder
                if self.uses_mrope:
                    pad = self.pad[:, :num_pads]
                    iids = np.concatenate([iids, pad[0]], dtype=np.int64)
                    pids = np.concatenate([pids, pad], dtype=np.int64, axis=1)
                else:
                    pad = self.pad[:num_pads]
                    iids = np.concatenate([iids, pad], dtype=np.int64)
                    pids = np.concatenate([pids, pad], dtype=np.int64)

            # create chunk inputs
            chunk_inputs = dict()
            batch_index = batch_indices[index]
            chunk_inputs["batch_index"] = batch_indices[index : index + 1].reshape(1, 1)
            if lora_ids is not None:
                chunk_inputs["lora_ids"] = lora_ids
            if mm_kwargs_list and (mm_kwargs := mm_kwargs_list[index]):
                chunk_inputs.update(mm_kwargs)

            # chunk the request
            n_chunks: int = iids.shape[-1] // self.prefill_seq_len
            assert n_chunks > 0
            for chunk in range(n_chunks):
                last_chunk = False
                lower_idx = int(chunk * self.prefill_seq_len)
                upper_idx = int((chunk + 1) * self.prefill_seq_len)
                if chunk + 1 == n_chunks and not prefill_is_partial[index]:
                    last_chunk = True
                chunk_inputs["input_ids"] = iids[lower_idx:upper_idx].reshape(
                    1, self.prefill_seq_len
                )
                # Reconstruct mm_token_type_ids per chunk from input_ids.
                # mm_token_type_ids is a binary mask: 1 at image-token positions,
                # 0 elsewhere. It is initialised as zeros((1,1)) from
                # default_mm_kwargs before the chunk loop. Each chunk must
                # receive a (1, prefill_seq_len) mask derived from that chunk's
                # input_ids so the QPC knows which tokens to replace with
                # vision_embeds. Without this every chunk gets zeros((1,1)) and
                # the QPC treats all tokens as text, ignoring vision_embeds.
                if "mm_token_type_ids" in chunk_inputs:
                    image_token_id = getattr(self.config, "image_token_id", None)
                    if image_token_id is not None:
                        chunk_inputs["mm_token_type_ids"] = (
                            chunk_inputs["input_ids"] == image_token_id
                        ).astype(np.int64)
                if self.uses_mrope:
                    chunk_inputs["position_ids"] = pids[:, lower_idx:upper_idx].reshape(
                        4, 1, self.prefill_seq_len
                    )
                else:
                    chunk_inputs["position_ids"] = pids[lower_idx:upper_idx].reshape(
                        1, self.prefill_seq_len
                    )

                if self.comp_ctx_lengths_prefill is not None:
                    # Single batch-wide bucket chosen before the loop; held constant.
                    if comp_ctx_val is not None:
                        chunk_inputs["comp_ctx_lengths"] = comp_ctx_val
                    # TODO: Workaround for CCL—LRT requires a buffer matching logits shape
                    if logits is not None:
                        chunk_inputs["logits"] = logits[index : index + 1]

                if pending_exec_count == self.session.prefill_num_execObj:
                    if callback:
                        callback()
                    logger.debug(
                        "All execObjs allocated; waiting for pending execObj completion."
                    )
                    eid = pending_exec_queue.get(timeout=120)
                    self.complete_inf(eid, True, pipeline_prefill_en=True)
                    pending_exec_count -= 1

                # Submit Chunk to LRT Queue
                exec_obj_idx = self.session.np_run_pipeline(
                    inputs=chunk_inputs,
                    last_chunk=last_chunk,
                    kv_cache_buffers=kv_caches[batch_index] if last_chunk else None,
                )
                if self.comp_ctx_lengths_prefill is not None:
                    # Record the bucket this slot is now running so future
                    # batches can fold it into max(active_ccl.values()).
                    self.active_ccl[exec_obj_idx] = chosen_ccl
                time.sleep(0.01)
                pending_exec_queue.put(exec_obj_idx)
                pending_exec_count += 1

        # wait for all chunks to finish
        if not self.use_async_scheduling:
            while not pending_exec_queue.empty():
                exec_obj_idx = pending_exec_queue.get(timeout=120)
                if callback:
                    callback()
                self.complete_inf(exec_obj_idx, True, pipeline_prefill_en=True)
        elif callback:
            # async path: free shm buffers, overlapping cleanup with device execution
            callback(self.stages)

        return

    def _run_prefill(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        prefill_cum_sum: np.ndarray,
        pending_exec_queue: Queue,
        logits: np.ndarray | None,
        lora_ids: np.ndarray | None = None,
        mm_kwargs_list: list[dict] | None = None,
        sampling_params: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        # perform prefill (only prefill_bsz=1 is supported)
        pending_exec_count = 0  # in-flight executions in current batch
        num_prefill_reqs = int(len(prefill_cum_sum))
        ods_prefill_temperatures = None
        ods_prefill_top_ks = None
        ods_prefill_top_ps = None
        ods_prefill_min_ps = None
        ods_prefill_repetition_penalties = None
        ods_prefill_presence_penalties = None
        ods_prefill_random_numbers = None
        if self.on_device_sampling_en:
            if self.prefill_next_tokens is None:
                raise ValueError(
                    "On-device sampling is enabled but prefill `next_tokens` "
                    "output buffer is uninitialized."
                )
            self.ods_step_prefill_next_tokens = np.empty(
                (num_prefill_reqs, 1),
                dtype=self.prefill_next_tokens.dtype,
            )
            if sampling_params is not None:
                ods_prefill_temperatures = np.asarray(
                    sampling_params["temperatures"], dtype=np.float32
                )
                ods_prefill_top_ks = np.asarray(sampling_params["top_ks"], dtype=np.int32)
                ods_prefill_top_ps = np.asarray(sampling_params["top_ps"], dtype=np.float32)
                ods_prefill_min_ps = np.asarray(sampling_params["min_ps"], dtype=np.float32)
                ods_prefill_repetition_penalties = np.asarray(
                    sampling_params["repetition_penalties"], dtype=np.float32
                )
                ods_prefill_presence_penalties = np.asarray(
                    sampling_params["presence_penalties"], dtype=np.float32
                )
                ods_prefill_random_numbers = np.asarray(
                    sampling_params["random_numbers"], dtype=np.float32
                )
                if (
                    ods_prefill_temperatures.shape[0] != num_prefill_reqs
                    or ods_prefill_top_ks.shape[0] != num_prefill_reqs
                    or ods_prefill_top_ps.shape[0] != num_prefill_reqs
                    or ods_prefill_min_ps.shape[0] != num_prefill_reqs
                    or ods_prefill_repetition_penalties.shape[0] != num_prefill_reqs
                    or ods_prefill_presence_penalties.shape[0] != num_prefill_reqs
                    or ods_prefill_random_numbers.shape[0] != num_prefill_reqs
                ):
                    raise ValueError(
                        "Prefill ODS sampling parameter length mismatch: expected "
                        f"{num_prefill_reqs} rows for each control array."
                    )
                if ods_prefill_random_numbers.shape[1] != self.ods_max_top_k_ids:
                    raise ValueError(
                        "Prefill ODS random_numbers width mismatch: expected "
                        f"{self.ods_max_top_k_ids}, got "
                        f"{ods_prefill_random_numbers.shape[1]}."
                    )
        idx_start = 0
        for i, idx_end in enumerate(prefill_cum_sum):
            # extract indices of specific request
            iids = input_ids[idx_start:idx_end]
            pids = positions[..., idx_start:idx_end]
            idx_start = idx_end
            # concatenate to multiple of `self.prefill_seq_len`
            # to avoid reading/writing KV$ on `num_pads` tokens
            n_prompt_tokens = iids.shape[-1]
            if (remainder := n_prompt_tokens % self.prefill_seq_len) > 0:
                num_pads = self.prefill_seq_len - remainder
                if self.uses_mrope:
                    pad = self.pad[:, :num_pads]
                    iids = np.concatenate([iids, pad[0]], dtype=np.int64)
                    pids = np.concatenate([pids, pad], dtype=np.int64, axis=1)
                else:
                    pad = self.pad[:num_pads]
                    iids = np.concatenate([iids, pad], dtype=np.int64)
                    pids = np.concatenate([pids, pad], dtype=np.int64)

            if self.config.model_type == "whisper":
                # Keep encoder-decoder prefill input shape compatible with the
                # current QEff support (seq_len=1) when serving Whisper.
                # vLLM does not allow prefill chunking for encoder-decoder model,
                # and currently QEff only supports seq_len=1.
                iids = np.full(
                    (1, self.prefill_seq_len), self.config.decoder_start_token_id
                )
                pids = pids[..., : self.prefill_seq_len]

            # create chunk inputs
            chunk_inputs = dict()
            if not self.ignore_batch_index:
                batch_index = batch_indices[i : i + 1].reshape(1, 1)
                chunk_inputs["batch_index"] = batch_index
            if lora_ids is not None:
                lora_index = lora_ids[i : i + 1].reshape(1, 1)
                chunk_inputs["lora_ids"] = lora_index
            if mm_kwargs_list and (mm_kwargs := mm_kwargs_list[i]):
                chunk_inputs.update(mm_kwargs)
            if self.on_device_sampling_en:
                if self.prefill_ods_inputs is None:
                    raise ValueError(
                        "On-device sampling is enabled but prefill sampling-control "
                        "buffers are uninitialized."
                    )
                if self.prefill_next_tokens is None:
                    raise ValueError(
                        "On-device sampling is enabled but prefill `next_tokens` "
                        "output buffer is uninitialized."
                    )
                if self.prefill_last_accepted_output_tokens is None:
                    raise ValueError(
                        "On-device sampling is enabled but prefill "
                        "`last_accepted_output_tokens` input buffer is "
                        "uninitialized."
                    )
                chunk_inputs.update(self.prefill_ods_inputs)
                chunk_inputs["next_tokens"] = self.prefill_next_tokens
                if self.debug_return_probs_en:
                    if self.prefill_probs is None:
                        raise ValueError(
                            "On-device sampling debug/evaluation sub-mode is enabled "
                            "but prefill `probs` output buffer is uninitialized."
                        )
                    chunk_inputs["probs"] = self.prefill_probs
                # During prefill the sampler selects prefill_path via is_prefill,
                # so decode_path's last_accepted_output_tokens input is ignored.
                # Still provide a correctly-shaped placeholder tensor.
                chunk_inputs["last_accepted_output_tokens"] = (
                    self.prefill_last_accepted_output_tokens
                )
                if ods_prefill_temperatures is not None:
                    self.prefill_ods_inputs["temperatures"][0, 0] = (
                        ods_prefill_temperatures[i]
                    )
                    self.prefill_ods_inputs["top_ks"][0, 0] = ods_prefill_top_ks[i]
                    self.prefill_ods_inputs["top_ps"][0, 0] = ods_prefill_top_ps[i]
                    self.prefill_ods_inputs["min_ps"][0, 0] = ods_prefill_min_ps[i]
                    self.prefill_ods_inputs["repetition_penalties"][0, 0] = (
                        ods_prefill_repetition_penalties[i]
                    )
                    self.prefill_ods_inputs["presence_penalties"][0, 0] = (
                        ods_prefill_presence_penalties[i]
                    )
                    self.prefill_ods_inputs["random_numbers"][0, :] = (
                        ods_prefill_random_numbers[i]
                    )
            # chunk the request
            n_chunks: int = iids.shape[-1] // self.prefill_seq_len

            if logits is not None and not self.on_device_sampling_en:
                chunk_inputs["logits"] = logits[i : i + 1]

            prefill_ccl_id = 0
            for chunk in range(n_chunks):
                lower_idx = int(chunk * self.prefill_seq_len)
                upper_idx = int((chunk + 1) * self.prefill_seq_len)
                chunk_inputs["input_ids"] = iids[lower_idx:upper_idx].reshape(
                    1, self.prefill_seq_len
                )
                # Reconstruct mm_token_type_ids from input_ids: positions where
                # input_ids == image_token_id get value 1, all others get 0.
                # This tells the decoder which tokens are image embeddings.
                if "mm_token_type_ids" in chunk_inputs:
                    image_token_id = getattr(self.config, "image_token_id", None)
                    if image_token_id is not None:
                        chunk_inputs["mm_token_type_ids"] = (
                            chunk_inputs["input_ids"] == image_token_id
                        ).astype(np.int64)
                if self.uses_mrope:
                    chunk_inputs["position_ids"] = pids[:, lower_idx:upper_idx].reshape(
                        4, 1, self.prefill_seq_len
                    )
                else:
                    chunk_inputs["position_ids"] = pids[lower_idx:upper_idx].reshape(
                        1, self.prefill_seq_len
                    )
                if self.comp_ctx_lengths_prefill is not None:
                    assert self.list_of_comp_ctx_lengths is not None
                    prefill_ccl = self.comp_ctx_lengths_prefill[0]
                    for j in range(prefill_ccl_id, len(self.comp_ctx_lengths_prefill)):
                        if (
                            chunk_inputs["position_ids"].max()
                            < self.comp_ctx_lengths_prefill[j]
                        ):
                            prefill_ccl_id, prefill_ccl = (
                                j,
                                self.comp_ctx_lengths_prefill[j],
                            )
                            break
                    chunk_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[
                        prefill_ccl
                    ]

                if "num_logits_to_keep" in self.session.input_names:
                    # SpD target QPC: prefill keeps 1 logit (last token position).
                    chunk_inputs["num_logits_to_keep"] = np.array([[1]], dtype=np.int64)

                if self.on_device_sampling_en:
                    # Explicit host round-trip: feed previous step's retained-state
                    # output back as this step's penalty-history input.
                    np.copyto(
                        self.past_repetition_penalty_buffer,
                        self.past_repetition_penalty_buffer_retained_state,
                    )
                    np.copyto(
                        self.past_presence_penalty_buffer,
                        self.past_presence_penalty_buffer_retained_state,
                    )

                if pending_exec_count == self.session.prefill_num_execObj:
                    logger.debug(
                        "All execObjs allocated; waiting for pending execObj completion."
                    )
                    eid = pending_exec_queue.get()
                    self.session.complete_inf(eid, True)
                    pending_exec_count -= 1

                exec_obj_idx = self.session.np_run(chunk_inputs, is_prefill=True)
                logger.debug("Ran prefill on %s", exec_obj_idx)
                if self.on_device_sampling_en:
                    self.session.complete_inf(exec_obj_idx, True)
                    fresh_retained_states = self.session.refresh_retained_state_buffers(
                        [
                            "past_repetition_penalty_buffer_RetainedState",
                            "past_presence_penalty_buffer_RetainedState",
                        ],
                        exec_obj_idx,
                    )
                    np.copyto(
                        self.past_repetition_penalty_buffer_retained_state,
                        fresh_retained_states[
                            "past_repetition_penalty_buffer_RetainedState"
                        ],
                    )
                    np.copyto(
                        self.past_presence_penalty_buffer_retained_state,
                        fresh_retained_states[
                            "past_presence_penalty_buffer_RetainedState"
                        ],
                    )
                    if self.debug_return_probs_en:
                        if self.prefill_probs is None:
                            raise ValueError(
                                "On-device sampling debug/evaluation sub-mode is "
                                "enabled but prefill `probs` output buffer is "
                                "uninitialized at step completion."
                            )
                        self.ods_debug_last_prefill_probs = self.prefill_probs.copy()
                    if self.ods_step_prefill_next_tokens is None:
                        raise ValueError(
                            "On-device sampling prefill step output buffer was not "
                            "allocated."
                        )
                    if chunk + 1 == n_chunks:
                        self.ods_step_prefill_next_tokens[i, 0] = (
                            self.prefill_next_tokens[0, 0, 0]
                        )
                elif not self.use_async_scheduling:
                    self.session.complete_inf(exec_obj_idx, True)
                else:
                    time.sleep(0.01)
                    pending_exec_count += 1
                    pending_exec_queue.put(exec_obj_idx)

            if (
                self.config.model_type == "whisper"
                and mm_kwargs_list
                and (mm_kwargs := mm_kwargs_list[i])
            ):
                for k, v in mm_kwargs.items():
                    self.decode_batch_inputs[k][i] = v[0]

        return

    def _run_decode(
        self,
        input_ids: np.ndarray,
        positions: np.ndarray,
        batch_indices: np.ndarray,
        logits: np.ndarray | None,
        lora_ids: np.ndarray | None = None,
        sampling_params: dict[str, np.ndarray] | None = None,
        callback: Callable | None = None,
    ) -> None:
        # Use the per-step K set by QaicModelRunner.  Falls back to max K so
        # the method is callable without a runner (e.g., unit tests).
        current_k = self.active_k
        mdt = current_k + 1  # tokens per decode request for this step
        num_tokens = input_ids.shape[0]
        num_decodes = num_tokens // mdt  # number of decode requests
        if self.on_device_sampling_en:
            self.ods_step_num_decodes = num_decodes
            self.ods_step_decode_mdt = mdt
        batch_inputs = self.decode_batch_inputs_by_k[current_k]
        if self.on_device_sampling_en and "next_tokens" not in batch_inputs:
            raise ValueError(
                "On-device sampling is enabled but decode `next_tokens` output "
                f"buffer is missing for decode specialization K={current_k}."
            )
        if self.debug_return_probs_en and "probs" not in batch_inputs:
            raise ValueError(
                "On-device sampling debug/evaluation sub-mode is enabled but "
                "decode `probs` output buffer is missing for decode "
                f"specialization K={current_k}."
            )

        ods_decode_temperatures = None
        ods_decode_top_ks = None
        ods_decode_top_ps = None
        ods_decode_min_ps = None
        ods_decode_repetition_penalties = None
        ods_decode_presence_penalties = None
        ods_decode_random_numbers = None
        if self.on_device_sampling_en and sampling_params is not None:
            ods_decode_temperatures = np.asarray(
                sampling_params["temperatures"], dtype=np.float32
            )
            ods_decode_top_ks = np.asarray(sampling_params["top_ks"], dtype=np.int32)
            ods_decode_top_ps = np.asarray(sampling_params["top_ps"], dtype=np.float32)
            ods_decode_min_ps = np.asarray(sampling_params["min_ps"], dtype=np.float32)
            ods_decode_repetition_penalties = np.asarray(
                sampling_params["repetition_penalties"], dtype=np.float32
            )
            ods_decode_presence_penalties = np.asarray(
                sampling_params["presence_penalties"], dtype=np.float32
            )
            ods_decode_random_numbers = np.asarray(
                sampling_params["random_numbers"], dtype=np.float32
            )
            if (
                ods_decode_temperatures.shape[0] != num_decodes
                or ods_decode_top_ks.shape[0] != num_decodes
                or ods_decode_top_ps.shape[0] != num_decodes
                or ods_decode_min_ps.shape[0] != num_decodes
                or ods_decode_repetition_penalties.shape[0] != num_decodes
                or ods_decode_presence_penalties.shape[0] != num_decodes
                or ods_decode_random_numbers.shape[0] != num_decodes
            ):
                raise ValueError(
                    "Decode ODS sampling parameter length mismatch: expected "
                    f"{num_decodes} rows for each control array."
                )
            # Scalar ODS decode controls may arrive as either 1D
            # (num_decodes,) or one-column 2D (num_decodes, 1). Assigning the
            # un-squeezed one-column form into per-slot [:, 0] slices (shape
            # (num_decodes,)) raises NumPy "could not broadcast" ValueError
            # when num_decodes > 1 (a hard failure, not silent corruption).
            # num_decodes == 1 happens to work via incidental broadcasting, but
            # relying on that is fragile, so we always squeeze one-column
            # inputs. random_numbers is intentionally exempt because it is
            # truly 2D (num_decodes, ods_max_top_k_ids).
            if (
                ods_decode_temperatures.ndim == 2
                and ods_decode_temperatures.shape[1] == 1
            ):
                ods_decode_temperatures = ods_decode_temperatures[:, 0]
            if ods_decode_top_ks.ndim == 2 and ods_decode_top_ks.shape[1] == 1:
                ods_decode_top_ks = ods_decode_top_ks[:, 0]
            if ods_decode_top_ps.ndim == 2 and ods_decode_top_ps.shape[1] == 1:
                ods_decode_top_ps = ods_decode_top_ps[:, 0]
            if ods_decode_min_ps.ndim == 2 and ods_decode_min_ps.shape[1] == 1:
                ods_decode_min_ps = ods_decode_min_ps[:, 0]
            if (
                ods_decode_repetition_penalties.ndim == 2
                and ods_decode_repetition_penalties.shape[1] == 1
            ):
                ods_decode_repetition_penalties = ods_decode_repetition_penalties[:, 0]
            if (
                ods_decode_presence_penalties.ndim == 2
                and ods_decode_presence_penalties.shape[1] == 1
            ):
                ods_decode_presence_penalties = ods_decode_presence_penalties[:, 0]
            if ods_decode_random_numbers.ndim != 2:
                raise ValueError(
                    "Decode ODS random_numbers must be a 2D array with shape "
                    f"({num_decodes}, {self.ods_max_top_k_ids})."
                )
            if ods_decode_random_numbers.shape[1] != self.ods_max_top_k_ids:
                raise ValueError(
                    "Decode ODS random_numbers width mismatch: expected "
                    f"{self.ods_max_top_k_ids}, got "
                    f"{ods_decode_random_numbers.shape[1]}."
                )

        batch_inputs["input_ids"][:num_decodes] = input_ids.reshape(num_decodes, mdt)
        if self.uses_mrope:
            # positions shape: (4, num_decodes * mdt) -> (4, num_decodes, mdt)
            batch_inputs["position_ids"][:, :num_decodes] = positions.reshape(
                4, num_decodes, mdt
            )
        else:
            batch_inputs["position_ids"][:num_decodes] = positions.reshape(
                num_decodes, mdt
            )
        if logits is not None and not self.on_device_sampling_en:
            batch_inputs["logits"] = logits
        if num_decodes < self.decode_bsz:
            batch_inputs["input_ids"][num_decodes:] = -1
            batch_inputs["position_ids"][..., num_decodes:, :] = -1

        if not self.ignore_batch_index:
            batch_inputs["batch_index"][:num_decodes, 0] = batch_indices
            if num_decodes < self.decode_bsz:
                # Pad idle rows with real, unused physical slots: negative indices
                # can wrap to the last slot in device-side KV-cache indexing.
                active_batch_slots = {int(slot_index) for slot_index in batch_indices}
                unused_batch_slots = [
                    slot_index
                    for slot_index in range(self.decode_bsz)
                    if slot_index not in active_batch_slots
                ]
                num_idle_rows = self.decode_bsz - num_decodes
                batch_inputs["batch_index"][num_decodes:, 0] = unused_batch_slots[
                    :num_idle_rows
                ]

        if lora_ids is not None:
            batch_inputs["lora_ids"][:num_decodes] = lora_ids.reshape(num_decodes, 1)
            if num_decodes < self.decode_bsz:
                batch_inputs["lora_ids"][num_decodes:] = -1

        # For spec-decode target: include num_logits_to_keep in batch_inputs
        # so the hardware knows how many token positions to compute logits for.
        if self.is_spec_decode_target_model:
            batch_inputs.update(self.decode_num_logits_buffer_by_k[current_k])

        if self.on_device_sampling_en:
            if ods_decode_temperatures is not None:
                batch_inputs["temperatures"][:num_decodes, 0] = ods_decode_temperatures
                batch_inputs["top_ks"][:num_decodes, 0] = ods_decode_top_ks
                batch_inputs["top_ps"][:num_decodes, 0] = ods_decode_top_ps
                batch_inputs["min_ps"][:num_decodes, 0] = ods_decode_min_ps
                batch_inputs["repetition_penalties"][:num_decodes, 0] = (
                    ods_decode_repetition_penalties
                )
                batch_inputs["presence_penalties"][:num_decodes, 0] = (
                    ods_decode_presence_penalties
                )
                batch_inputs["random_numbers"][:num_decodes, :] = ods_decode_random_numbers
                if num_decodes < self.decode_bsz:
                    batch_inputs["temperatures"][num_decodes:, 0] = 0.0
                    batch_inputs["top_ks"][num_decodes:, 0] = 0
                    batch_inputs["top_ps"][num_decodes:, 0] = 0.0
                    batch_inputs["min_ps"][num_decodes:, 0] = 0.0
                    batch_inputs["repetition_penalties"][num_decodes:, 0] = 0.0
                    batch_inputs["presence_penalties"][num_decodes:, 0] = 0.0
                    batch_inputs["random_numbers"][num_decodes:, :] = 0.0
            # For non-speculative ODS decode, next-step input_ids already contain
            # the accepted token IDs from the previous step.
            batch_inputs["last_accepted_output_tokens"] = batch_inputs["input_ids"]
            # Explicit host round-trip: previous decode step's retained-state
            # output becomes this step's penalty-history input.
            np.copyto(
                self.past_repetition_penalty_buffer,
                self.past_repetition_penalty_buffer_retained_state,
            )
            np.copyto(
                self.past_presence_penalty_buffer,
                self.past_presence_penalty_buffer_retained_state,
            )

        if self.comp_ctx_lengths_decode is not None:
            assert self.list_of_comp_ctx_lengths is not None
            max_position_id = positions.max().item()
            batch_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[
                max(self.list_of_comp_ctx_lengths.keys())
            ]
            for comp_ctx_len in self.comp_ctx_lengths_decode:
                if max_position_id < comp_ctx_len:
                    batch_inputs["comp_ctx_lengths"] = self.list_of_comp_ctx_lengths[
                        comp_ctx_len
                    ]
                    break

        exec_obj_idx = self.session.np_run(batch_inputs, is_prefill=False)
        logger.debug("Ran decode on %s", exec_obj_idx)

        if callback:
            callback()

        if not self.use_async_scheduling:
            self.session.complete_inf(exec_obj_idx, is_prefill=False)
            if self.on_device_sampling_en:
                # NOTE: This readback handles only the synchronous completion path here.
                # Async-scheduling completion runs later in model_runner.py's
                # complete_all_inf(), which refreshes retained-state buffers in that path.
                fresh_retained_states = self.session.refresh_retained_state_buffers(
                    [
                        "past_repetition_penalty_buffer_RetainedState",
                        "past_presence_penalty_buffer_RetainedState",
                    ],
                    exec_obj_idx,
                )
                np.copyto(
                    self.past_repetition_penalty_buffer_retained_state,
                    fresh_retained_states[
                        "past_repetition_penalty_buffer_RetainedState"
                    ],
                )
                np.copyto(
                    self.past_presence_penalty_buffer_retained_state,
                    fresh_retained_states[
                        "past_presence_penalty_buffer_RetainedState"
                    ],
                )
            if self.debug_return_probs_en:
                if self.decode_probs_by_k is None:
                    raise ValueError(
                        "On-device sampling debug/evaluation sub-mode is enabled but "
                        "decode `probs` output buffers are uninitialized."
                    )
                decode_probs = self.decode_probs_by_k.get(current_k)
                if decode_probs is None:
                    raise ValueError(
                        "On-device sampling debug/evaluation decode output buffer "
                        f"missing for decode specialization K={current_k}."
                    )
                self.ods_debug_last_decode_probs = decode_probs.copy()

        return

    def _run_encode(self, prefill_cum_sum, input_ids, positions):
        prompt_lens = np.diff(np.concatenate(([0], prefill_cum_sum))).tolist()
        allowed_seqlens = self.get_allowed_seqlens()
        encode_seq_len = (
            min(sq for sq in allowed_seqlens if int(sq) > max(prompt_lens))
            if allowed_seqlens
            else self.prefill_seq_len
        )
        output_key = "logits" if self.task in ("score", "classify") else "output"

        # Build output buffer.
        if self.is_qaic_pooler:
            encode_num_logits_buffer: dict = {
                "output": np.empty((self.decode_bsz, self.hidden_dimension), np.float32)
            }
        elif output_key == "logits":
            encode_num_logits_buffer = {
                "logits": np.empty(
                    (self.decode_bsz, getattr(self.config, "num_labels", 1)), np.float32
                )
            }
        else:
            encode_num_logits_buffer = {
                "output": np.empty(
                    (self.decode_bsz, encode_seq_len, self.hidden_dimension), np.float32
                ),
                **{
                    name: np.empty((self.decode_bsz, self.hidden_dimension), np.float32)
                    for name in self.session.output_names
                    if name != "output"
                },
            }

        self._encode_merged_inputs = {
            "input_ids": np.full(
                (self.decode_bsz, encode_seq_len), 2, dtype=input_ids[0].dtype
            ),
            "attention_mask": np.zeros(
                (self.decode_bsz, encode_seq_len), dtype=input_ids[0].dtype
            ),
        }
        idx_start = 0
        for i, idx_end in enumerate(prefill_cum_sum):
            pl = prompt_lens[i]
            self._encode_merged_inputs["input_ids"][i, :pl] = input_ids[
                idx_start:idx_end
            ]
            self._encode_merged_inputs["attention_mask"][i, :pl] = 1
            idx_start = idx_end

        self.run_encode(
            self._encode_merged_inputs, output_key, encode_num_logits_buffer
        )

        if not self.use_async_scheduling:
            return self._process_encode_output(prefill_cum_sum, prompt_lens, output_key)
        else:
            self._encode_prefill_cum_sum = prefill_cum_sum
            self._encode_prompt_lens = prompt_lens
            self._encode_output_key = output_key
            pending_encode_queue: Queue = Queue()
            pending_encode_queue.put(self.encode_execObj_idx)
            return pending_encode_queue

    def _process_encode_output(
        self,
        prefill_cum_sum: np.ndarray,
        prompt_lens: list[int],
        output_key: str,
    ) -> torch.Tensor:
        """Convert the filled encode output buffer into a torch tensor.

        Called either immediately (sync path) or after complete_inf (async path).
        """
        output = self.encode_num_logits_buffer
        output_array = output[output_key][: len(prefill_cum_sum)]
        output_tensor = torch.tensor(output_array)

        if not self.is_qaic_pooler and not output_key == "logits":
            hidden_states = torch.cat(
                [
                    output_tensor[i, : prompt_lens[i]]
                    for i in range(output_tensor.shape[0])
                ],
                dim=0,
            )
            return hidden_states
        else:
            if self.normalize:
                output_tensor = F.normalize(output_tensor, p=2, dim=1)
            if self.softmax:
                output_tensor = F.softmax(output_tensor, dim=1)
        return output_tensor

    def complete_inf(
        self, index: int, is_prefill: bool, pipeline_prefill_en: bool = False
    ):
        """Complete the inference execution for the given execution object index.

        Args:
            index: Index of the execObj.
            is_prefill: True if completing a prefill inference, False for decode.
        """
        self.session.complete_inf(index, is_prefill)
        if pipeline_prefill_en and self.comp_ctx_lengths_prefill is not None:
            # Slot is no longer in flight; drop its CCL so max(active_ccl)
            # reflects only what's still running.
            self.active_ccl[index] = 0

    @property
    def decode_execObj_idx(self) -> int | None:
        return self.session.decode_execObj_idx

    @property
    def async_scheduling_exec_timeout(self) -> int | None:
        return self.session.async_scheduling_exec_timeout

    def run_encode(
        self,
        qpc_inputs: dict,
        output_key: str | None = None,
        encode_num_logits_buffer: dict | None = None,
    ) -> dict:
        """Run encode (embedding) inference on the QPC.

        The np session maintains two separate exec-object pools: one for prefill
        and one for decode.  Encode is a single-pass forward pass (no KV-cache
        update), so it reuses the prefill pool:
          - np_run(is_prefill=True)  → dequeues an exec object from
            prefill_available_exec_objs
          - complete_inf(is_prefill=True) → returns that exec object back to
            prefill_available_exec_objs after the inference completes

        When async_scheduling is enabled the caller is responsible for calling
        complete_inf later (via complete_all_inf in the model runner), mirroring
        the pattern used by _run_decode.

        Args:
            qpc_inputs: input tensors (e.g. input_ids, attention_mask)
            output_key: key for the output buffer ("output" or "logits")
            encode_num_logits_buffer: output buffer dict; re-registered when shape changes
        Returns:
            dict: output buffer dict containing the hidden-state / pooled output
        """
        if (
            self.encode_num_logits_buffer is None
            or encode_num_logits_buffer[output_key].shape
            != self.encode_num_logits_buffer[output_key].shape
        ):
            self.session.set_buffers(encode_num_logits_buffer)
            self.encode_num_logits_buffer = encode_num_logits_buffer

        encode_exec_obj_idx = self.session.np_run(
            {**qpc_inputs, **self.encode_num_logits_buffer}
        )

        if not self.use_async_scheduling:
            self.session.complete_inf(encode_exec_obj_idx, is_prefill=True)
        else:
            self.encode_execObj_idx = encode_exec_obj_idx

        return self.encode_num_logits_buffer

    def disagg_dummy_run(self):
        """assert prefill and decode work by running dummy inputs

        also creates attention_mask and decode input buffers
        that will be used throughout the lifeycle of worker
        """

        # Prepare dummy run inputs
        if self.session.cluster_id == "prefill":
            if self.uses_mrope:
                # QEfficient requires position ids to be (4, batch_size, seq_len)
                _pids = np.tile(
                    np.full((self.prefill_seq_len), -1, dtype=np.int64).reshape(
                        1, 1, self.prefill_seq_len
                    ),
                    (4, self.prefill_bsz, 1),
                )
            else:
                _pids = np.tile(
                    np.full((self.prefill_seq_len), -1, dtype=np.int64).reshape(
                        1, self.prefill_seq_len
                    ),
                    (self.prefill_bsz, 1),
                )
            prefill_inputs = {
                "input_ids": np.zeros(
                    (self.prefill_bsz, self.prefill_seq_len), dtype=np.int64
                ),
                "position_ids": _pids,
                "batch_index": np.arange(self.prefill_bsz).reshape(-1, 1),
            }
            # TODO: mllama3.2 is currently not supported in v0.15.0
            # if input_info := self.get_io_shape_and_dtype("cross_attention_mask"):
            #     self.is_cross_attention = True
            #     (dims, dtype, _) = input_info
            #     self.prefill_cross_attention_mask = np.zeros(
            #         (dims[0], dims[1], dims[2], dims[3]), dtype=dtype
            #     )
            # else:
            #     self.is_cross_attention = False

            if self.lora_mode:
                prefill_inputs["lora_ids"] = np.arange(self.prefill_bsz).reshape(-1, 1)

            if self.comp_ctx_lengths_prefill is not None:
                prefill_inputs["comp_ctx_lengths"] = np.zeros(
                    self.comp_ctx_lengths_prefill[-1], dtype=np.int64
                )
            self.prefill_batch_inputs = prefill_inputs.copy()

        # Prepare decode inputs
        if self.session.cluster_id == "decode":
            # decode inputs
            if self.uses_mrope:
                # QEfficient requires position ids to be (4, batch_size, seq_len)
                decode_single_inputs = {
                    "input_ids": np.array([[0]]),
                    "position_ids": np.zeros((4, 1, 1), dtype=np.int64),
                }
                decode_batch_inputs = {
                    "input_ids": np.zeros((self.decode_bsz, 1), dtype=np.int64),
                    "position_ids": np.full(
                        (4, self.decode_bsz, 1), -1, dtype=np.int64
                    ),
                }
            else:
                decode_single_inputs = {
                    "input_ids": np.array([[0]]),
                    "position_ids": np.array([[0]]),
                }
                decode_batch_inputs = {
                    "input_ids": np.zeros((self.decode_bsz, 1), dtype=np.int64),
                    "position_ids": np.full((self.decode_bsz, 1), -1, dtype=np.int64),
                }
            if self.is_spec_decode_target_model:
                # decode on this model has multiple tokens per batch (aka precode)
                decode_single_inputs = dict(
                    input_ids=np.zeros((1, self.num_logits_to_keep), dtype=np.int64),
                    position_ids=np.full(
                        (1, self.num_logits_to_keep), -1, dtype=np.int64
                    ),
                )
                decode_batch_inputs = dict(
                    input_ids=np.zeros(
                        (self.decode_bsz, self.num_logits_to_keep), dtype=np.int64
                    ),
                    position_ids=np.full(
                        (self.decode_bsz, self.num_logits_to_keep), -1, dtype=np.int64
                    ),
                )
            if "batch_index" in self.session.input_names:
                decode_single_inputs["batch_index"] = np.array([[0]])
                decode_batch_inputs["batch_index"] = np.arange(
                    self.decode_bsz, dtype=np.int64
                ).reshape(-1, 1)
                self.ignore_batch_index = False
            else:
                self.ignore_batch_index = True

            if self.lora_mode:
                decode_single_inputs["lora_ids"] = np.array([[0]])
                decode_batch_inputs["lora_ids"] = np.arange(
                    self.decode_bsz, dtype=np.int64
                ).reshape(-1, 1)

            # TODO: mllama3.2 is currently not supported in v0.15.0
            # if input_info := self.get_io_shape_and_dtype("cross_attention_mask"):
            #     self.is_cross_attention = True
            #     (dims, dtype, _) = input_info
            #     self.prefill_cross_attention_mask = np.zeros(
            #         (dims[0], dims[1], dims[2], dims[3]), dtype=dtype
            #     )
            #     decode_single_inputs["cross_attention_mask"] = np.ones(
            #         (1, dims[2], dims[3]), dtype=dtype
            #     )
            #     decode_batch_inputs["cross_attention_mask"] = np.ones(
            #         (dims[0], 1, dims[2], dims[3]), dtype=dtype
            #     )
            # else:
            #     self.is_cross_attention = False

            self.decode_single_inputs = decode_single_inputs
            self.decode_batch_inputs = decode_batch_inputs
            # Re-inject any mm kwargs (e.g. image_idx) that _load_multimodal()
            # added to decode_batch_inputs before this dict was replaced.
            if getattr(self, "default_mm_kwargs", None):
                self.decode_batch_inputs.update(self.default_mm_kwargs)
        # TODO: Clean up
        # self._input_map_chg_needed = "decoder_input_ids" in self.session.input_names
        # # This is a hack for mapping names to qpc input,
        # # will be changed in future releases
        # embeds_name = [
        #     input_name
        #     for input_name in self.session.input_names
        #     if "embeds" in input_name or "image_features" in input_name
        # ]
        # if embeds_name:
        #     self.embeds_name = embeds_name[0]
        # run dummy inputs
        if "logits" in self.session.output_names:
            logger.debug("starting dummy run...")
            if self.session.cluster_id == "prefill":
                logger.info("Running dummy prefill run for %s stages", self.stages)
                bidx = 0
                while bidx < self.session.prefill_num_execObj:
                    if self.uses_mrope:
                        _dummy_pids = np.tile(
                            np.full((self.prefill_seq_len), -1, dtype=np.int64).reshape(
                                1, 1, self.prefill_seq_len
                            ),
                            (4, self.prefill_bsz, 1),
                        )
                    else:
                        _dummy_pids = np.tile(
                            np.full((self.prefill_seq_len), -1, dtype=np.int64).reshape(
                                1, self.prefill_seq_len
                            ),
                            (self.prefill_bsz, 1),
                        )
                    prefill_inputs = {
                        "input_ids": np.zeros(
                            (self.prefill_bsz, self.prefill_seq_len), dtype=np.int64
                        ),
                        "position_ids": _dummy_pids,
                        "batch_index": np.arange(self.prefill_bsz).reshape(-1, 1),
                        "logits": np.empty(
                            (self.prefill_bsz, 1, self.vocab_size)
                            if self.logits_ndim == 3
                            else (self.prefill_bsz, self.vocab_size),
                            dtype=np.float32,
                        ),
                    }
                    if self.lora_mode:
                        prefill_inputs["lora_ids"] = np.arange(
                            self.prefill_bsz
                        ).reshape(-1, 1)
                    if self.is_multimodal_model and self.default_mm_kwargs:
                        prefill_inputs.update(self.default_mm_kwargs)
                    KvCache_buff = []
                    for kv_shape, kv_type, _ in self.kv_cache_info:
                        _kv_shape = (1,) + kv_shape[1:]
                        KvCache_buff.append(np.empty(shape=_kv_shape, dtype=kv_type))
                    if self.comp_ctx_lengths_prefill is not None:
                        prefill_inputs["comp_ctx_lengths"] = np.zeros(
                            self.comp_ctx_lengths_prefill[-1], dtype=np.int64
                        )
                    exec_obj_idx = self.session.np_run_pipeline(
                        inputs=prefill_inputs,
                        slicing_parameters=None,
                        last_chunk=True,
                        kv_cache_buffers=KvCache_buff,
                    )
                    self.session.complete_inf(exec_obj_idx, is_prefill=True)
                    bidx += 1
                logger.info("Finished dummy prefill run")
            if self.session.cluster_id == "decode":
                logger.info("Running dummy decode run with bsz %s", self.decode_bsz)
                bidx = 0
                input_kv_buffers: dict[str, Any] = {}
                KvCache_buff = []
                for kv_shape, kv_type, _ in self.kv_cache_info:
                    _kv_shape = (self.decode_bsz,) + kv_shape[1:]
                    KvCache_buff.append(np.empty(shape=kv_shape, dtype=kv_type))

                decode_logits_shape = (
                    (self.decode_bsz, 1, self.vocab_size)
                    if self.logits_ndim == 3
                    else (self.decode_bsz, self.vocab_size)
                )
                self.session.create_output_buffers(
                    input_kv_buffers, decode_logits_shape, np.float32
                )
                # Pre-allocate logits buffer into decode_batch_inputs once so that
                # _run_decode can reuse it instead of allocating on every decode step.
                self.session.create_output_buffers(
                    self.decode_batch_inputs,
                    decode_logits_shape,
                    np.float32,
                )
                # CCL: seed comp_ctx_lengths buffer for the dummy run; _run_decode
                # overwrites it per step with the selected bucket buffer.
                if self.comp_ctx_lengths_decode is not None:
                    self.decode_batch_inputs["comp_ctx_lengths"] = np.zeros(
                        self.comp_ctx_lengths_decode[-1], dtype=np.int64
                    )
                _ = self.session.set_data_for_kv_handoff(
                    input_kv_buffers,
                    [("batch_index", bidx), ("ctx_start", 0)],
                    self.decode_execObj_idx,
                    self.session.decode_buff_map,
                )
                bidx += 1
                exec_obj_idx = self.session.np_run(
                    self.decode_batch_inputs, is_prefill=False
                )
                self.session.complete_inf(exec_obj_idx, is_prefill=False)
                # self.decode_batch_inputs = decode_batch_inputs_temp
                logger.info("Finished dummy decode run with bsz %s", self.decode_bsz)
            logger.debug("finished dummy run")


def load_qaic_model(
    vllm_config: VllmConfig, speculative_model_type: str | None = None
) -> nn.Module:
    # Draft model must compile with max_decode_tokens=1. Clear speculative_config
    # so QaicCausalLM doesn't inherit num_spec_tokens from the target config.
    if speculative_model_type == "draft":
        from copy import copy

        vllm_config = copy(vllm_config)
        vllm_config.speculative_config = None

    # Create a model instance
    if vllm_config.model_config.is_multimodal_model:
        from vllm_qaic.model_loader.qaic_multimodal import QaicMultiModal

        cls = QaicMultiModal
    else:
        cls = QaicCausalLM
    model = cls(vllm_config)

    if speculative_model_type is None:
        speculative_model_type = "default"
        model.sampler.include_gpu_probs_tensor = False
    else:
        speculative_model_type = speculative_model_type.lower()
        model.sampler.include_gpu_probs_tensor = True

    if speculative_model_type not in QAIC_DEVICE_CONFIG:
        raise ValueError(
            f"Unable to find default profile for model type {speculative_model_type}!!\n"
        )

    qaic_compile_config = _get_qaic_compile_config(vllm_config, speculative_model_type)
    qpc_path = qaic_compile_config.qpc_path

    # set lora max adapters
    if vllm_config.lora_config:
        qaic_max_adapters = int(os.environ.get("VLLM_QAIC_LORA_MAX_ID_SUPPORTED", 128))

    if (
        vllm_config.model_config.runner_type == "pooling"
        and not vllm_config.model_config.is_multimodal_model
    ):
        override_qaic_config = (vllm_config.additional_config or {}).get(
            "override_qaic_config"
        ) or {}
        assert override_qaic_config.get("pooling_device"), (
            "pooling_device must be provided in override_qaic_config for pooling task"
        )
        if override_qaic_config.get("pooling_device", None) == "qaic":
            if override_qaic_config.get("task") not in ("score", "classify"):
                assert override_qaic_config.get("pooling_method"), (
                    "pooling_method must be provided in override_qaic_config for qaic pooling task"
                )

    # if provided qpc is valid
    if qpc_path and not check_qpc_exists(qpc_path):
        raise ValueError(
            f"Environment variable VLLM_QAIC_QPC_PATH is set!\n"
            f"QAIC qpc path {qpc_path} doesn't exist or didn't have compiled binary!\n"
            "Unset VLLM_QAIC_QPC_PATH, if you don't want to provide compiled qpc.\n"
        )

    # set adaptername_to_id from previous dump file if qpc_path exist
    adaptername_to_id = {}
    if vllm_config.lora_config and (qpc_path and check_qpc_exists(qpc_path)):
        # check if json file exist
        if os.path.exists(f"{qpc_path}/adaptername_to_id.json"):
            with open(f"{qpc_path}/adaptername_to_id.json") as file:
                adaptername_to_id = json.load(file)
        else:
            raise FileNotFoundError(
                f"The file at {qpc_path}/adaptername_to_id.json was not found. "
                "Please provide a correct VLLM_QAIC_QPC_PATH."
            )

        # check if json file content is correct
        _lora_modules: list[LoRAModulePath] | None = (
            vllm_config.additional_config or {}
        ).get("lora_modules")
        if _lora_modules is None or not verify_adaptername_to_id_consistency(
            adaptername_to_id, _lora_modules
        ):
            raise ValueError(
                f"Inconsistent file content in {qpc_path}/adaptername_to_id.json"
                " and input lora modules."
            )

    # Generate qpc using QEfficient transformer
    if not qpc_path:
        # TODO: what will break by me doing this?
        quant_cfg = None
        quant_method = None
        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

        if (
            vllm_config.model_config.quantization is not None
            and vllm_config.model_config.quantization in ["awq", "gptq"]
            and quant_method != vllm_config.model_config.quantization
        ):
            raise ValueError(
                "Currently qaic backend only supports pre-quantized AWQ | GPTQ models"
                " via vllm!"
            )

        try:
            qeff_model = get_hf_model(
                vllm_config.model_config,
                qaic_compile_config.qaic_config,
                vllm_config.lora_config,
                qaic_compile_config.kv_offload,
                vllm_config.additional_config,
            )

            from QEfficient import (
                QEFFAutoModel,
                QEFFAutoModelForCTC,
                QEFFAutoModelForSequenceClassification,
                QEFFAutoModelForSpeechSeq2Seq,
            )
            from QEfficient.transformers.models.modeling_auto import (
                _QEFFAutoModelForImageTextToTextSingleQPC,
            )

            if isinstance(
                qeff_model,
                (
                    QEFFAutoModel,
                    QEFFAutoModelForCTC,
                    QEFFAutoModelForSequenceClassification,
                    QEFFAutoModelForSpeechSeq2Seq,
                    _QEFFAutoModelForImageTextToTextSingleQPC,
                ),
            ):
                # These model types do not expose `prefill_only` as an explicit
                # `compile()` parameter. QEFFAutoModelForCausalLM and the dual-QPC
                # image-text-to-text model (_QEffAutoModelForImageTextToTextDualQPC)
                # do support it, so they are intentionally excluded here.
                # NOTE: QEFFAutoModelForImageTextToText is a factory (__new__) that
                # returns one of the private single/dual QPC classes, so an
                # `isinstance` check against it never matches a real instance.
                # For the single-QPC path, `prefill_only` would otherwise be
                # forwarded via **compiler_options to the QAIC compiler as an
                # unknown flag — pop it to avoid unexpected behavior.
                qaic_compile_config.cfg.pop("prefill_only", None)

            if vllm_config.lora_config:
                logger.info(
                    "Transforming and compiling lora model using QEfficient library"
                )

                # search adapter in cache
                _lora_modules = (vllm_config.additional_config or {}).get(
                    "lora_modules"
                )
                if not _lora_modules:
                    # search adapter in cache
                    filtered_cached_lora_module_paths = search_adapters_in_cache(
                        vllm_config.model_config.model
                    )

                    # error out if cache is empty
                    if len(filtered_cached_lora_module_paths) == 0:
                        raise ValueError(
                            "No adapter in cache, please either download some "
                            "into HF_HOME or provide lora_modules list."
                        )
                    # set lora_modules
                    _lora_modules = filtered_cached_lora_module_paths
                    vllm_config.additional_config["lora_modules"] = _lora_modules

                # error out if reach adapter limit
                assert len(_lora_modules) <= qaic_max_adapters, (
                    f"Number of cached adapters exceed limitation of "
                    f"{qaic_max_adapters}. Please either delete adapters from "
                    "HF_HOME or specify adapters in lora_modules."
                )

                # load adapters to model
                for lora_module_path in _lora_modules:
                    model_dir = lora_module_path.path.split("/")[-3]
                    adapter_model_id = (
                        f"{model_dir.split('--')[1]}/{model_dir.split('--')[2]}"
                    )
                    qeff_model.load_adapter(
                        adapter_model_id,
                        lora_module_path.name,
                    )  # adapters with inconsistent target_modules or ranks
                    # will not be added here (TODO: add another check here)

                # get adaptername_to_id
                adaptername_to_id = qeff_model.active_adapter_to_id

            else:
                logger.info(
                    "Transforming and compiling model[%s] using QEfficient library",
                    speculative_model_type,
                )
            logger.info("QEFF Compile called with %s", qaic_compile_config.cfg)
            qpc_path = qeff_model.compile(**qaic_compile_config.cfg)
            if isinstance(qpc_path, dict):
                if (
                    "skip_lang" in qaic_compile_config.cfg
                    and qaic_compile_config.cfg["skip_lang"]
                ):
                    qpc_path = qpc_path.get("vision_qpc_path")
                elif (
                    "prefill_only" in qaic_compile_config.cfg
                    and qaic_compile_config.cfg["prefill_only"]
                ):
                    qpc_path = qpc_path.get("lang_prefill_qpc_path")
                elif (
                    "prefill_seq_len" in qaic_compile_config.cfg
                    and qaic_compile_config.cfg["prefill_seq_len"] == 1
                ):
                    qpc_path = qpc_path.get("lang_decode_qpc_path")
                else:
                    qpc_path = qpc_path.get("lang_qpc_path")
                if qpc_path is None:
                    raise ValueError(
                        "Failed to extract QPC path from compilation result dictionary"
                    )
        except Exception as e:
            logger.error("Failed to transform and compile the model! %s", e)
            raise e

    # dump adaptername_to_id to folder for the first compilation
    if vllm_config.lora_config and not os.path.exists(
        f"{qpc_path}/adaptername_to_id.json"
    ):
        with open(f"{qpc_path}/adaptername_to_id.json", "w") as file:
            json.dump(adaptername_to_id, file)
            logger.info(
                "Dump adaptername_to_id mapping to %s/adaptername_to_id.json",
                qpc_path,
            )

    if speculative_model_type != "default":
        logger.info(
            "Spec model type %s_%s",
            speculative_model_type,
            qaic_compile_config.num_logits_to_keep,
        )

    logger.info("Using qpc:-%s", qpc_path)

    if qaic_compile_config.compile_only:
        # Hack for Model-IP execution flow
        # TODO: remove this in future
        raise QaicCompilationComplete()

    # Load the weights from the cached or downloaded files.
    # model_config.qpc in None
    model.load_model(
        qpc_path=qpc_path,
        device_id=qaic_compile_config.device_group,
        num_logits_to_keep=qaic_compile_config.num_logits_to_keep,
        stages=qaic_compile_config.stages,
        kv_transfer_role=(
            vllm_config.kv_transfer_config.kv_role
            if vllm_config.kv_transfer_config
            else None
        ),
    )

    return model.eval()


# -----------------------------------------------------------------------
# old `qaic.py` imports
# -----------------------------------------------------------------------

VLLM_CACHE_DTYPE_TO_QAIC_CACHE_DTYPE = {
    "auto": "fp16",
    "fp8": "mxint8",
    "mxint8": "mxint8",
}

single_qpc_config = {
    "qpc_path": os.environ.get("VLLM_QAIC_QPC_PATH", None),
    "mos": os.environ.get("VLLM_QAIC_MOS", None),
    "aic_enable_depth_first": os.environ.get("VLLM_QAIC_DFS_EN", None),
    "device_group": os.environ.get(current_platform.device_control_env_var, None),
    "num_cores": os.environ.get("VLLM_QAIC_NUM_CORES", None),
    "compiler_args": os.environ.get("VLLM_QAIC_COMPILER_ARGS", None),
}

QAIC_DEVICE_CONFIG = {
    "target": {
        "qpc_path": os.environ.get("VLLM_QAIC_SPEC_TARGET_QPC_PATH", None),
        "mos": os.environ.get("VLLM_QAIC_SPEC_TARGET_MOS", None),
        "aic_enable_depth_first": os.environ.get("VLLM_QAIC_SPEC_TARGET_DFS_EN", None),
        "device_group": os.environ.get("VLLM_QAIC_SPEC_TARGET_QID", None),
        "num_cores": os.environ.get("VLLM_QAIC_SPEC_TARGET_NUM_CORES", None),
        "compiler_args": os.environ.get("VLLM_QAIC_SPEC_TARGET_COMPILER_ARGS", None),
    },
    "draft": {
        "qpc_path": os.environ.get("VLLM_QAIC_SPEC_DRAFT_QPC_PATH", None),
        "mos": os.environ.get("VLLM_QAIC_SPEC_DRAFT_MOS", None),
        "aic_enable_depth_first": os.environ.get("VLLM_QAIC_SPEC_DRAFT_DFS_EN", None),
        "mxint8_kv_cache": os.environ.get("VLLM_QAIC_SPEC_DRAFT_KV_COMPRESSION", None),
        "device_group": os.environ.get("VLLM_QAIC_SPEC_DRAFT_QID", None),
        "num_cores": os.environ.get("VLLM_QAIC_SPEC_DRAFT_NUM_CORES", None),
        "compiler_args": os.environ.get("VLLM_QAIC_SPEC_DRAFT_COMPILER_ARGS", None),
    },
    "default": single_qpc_config,
    "turbo": single_qpc_config,
}


@dataclass
class QaicCompileConfig:
    compile_only: bool
    qpc_path: str
    device_group: list[int]
    cfg: dict[str, Any]
    num_logits_to_keep: int | None
    kv_offload: bool
    include_sampler: bool | None
    include_guided_decoding: bool | None
    return_pdfs: bool | None
    max_top_k_ids: int | None
    qaic_config: dict[str, Any] | None = None
    stages: int | None = 1


# -----------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------


def check_qpc_exists(qpc_path: str) -> bool:
    bin_path = os.path.join(qpc_path, "programqpc.bin")
    if not os.path.exists(qpc_path):
        return False
    if not os.path.isdir(qpc_path) and "programqpc.bin" in qpc_path:
        return True
    return os.path.exists(bin_path)


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False


def get_hf_model(
    model_config: ModelConfig,
    qaic_config: dict | None = None,
    lora_config=None,
    kv_offload=False,
    additional_config: dict | None = None,
):
    logger.info("Downloading model from Hugging face server")
    from QEfficient import (
        QEFFAutoModel,
        QEFFAutoModelForCausalLM,
        QEFFAutoModelForImageTextToText,
        QEFFAutoModelForSpeechSeq2Seq,
        QEFFAutoModelForSequenceClassification,
    )
    from QEfficient.peft.lora import QEffAutoLoraModelForCausalLM

    QEff_class_mapping = {
        "default": QEFFAutoModelForCausalLM,
        "lora": QEffAutoLoraModelForCausalLM,
        "imagetext": QEFFAutoModelForImageTextToText,
        "speech": QEFFAutoModelForSpeechSeq2Seq,
        "encode": QEFFAutoModel,
        "seq_classify": QEFFAutoModelForSequenceClassification,
    }
    hf_config = model_config.hf_config
    if hf_config.model_type in _CONFIG_REGISTRY or not is_json_serializable(hf_config):
        # If vllm uses a custom model config class,
        # convert it back to the transformers config class
        from transformers import AutoConfig

        pretrained_hf_config = AutoConfig.from_pretrained(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
        )
        hf_config = AutoConfig.from_pretrained(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code,
            **hf_config.to_dict(),
        )
        # If tie_word_embeddings is not set correctly,
        # single QPC's output would be wrong
        hf_config.tie_word_embeddings = pretrained_hf_config.tie_word_embeddings
    override_qaic_config = (
        additional_config.get("override_qaic_config") if additional_config else None
    )
    args = {
        "continuous_batching": (override_qaic_config or {}).get(
            "continuous_batching", True
        ),
        "qaic_config": qaic_config,
        "trust_remote_code": model_config.trust_remote_code,
        "revision": model_config.revision,
        "code_revision": model_config.code_revision,
        "attn_implementation": "eager",
        "config": hf_config,
        "kv_offload": kv_offload,
    }

    if override_qaic_config and override_qaic_config.get("pretrained_extra_args", None):
        args.update(override_qaic_config["pretrained_extra_args"])
        override_qaic_config.pop("pretrained_extra_args")

    model_type = "lora" if lora_config else "default"
    if model_config.is_multimodal_model:
        if model_config.hf_config.model_type == "whisper":
            model_type = "speech"
            del args["kv_offload"]
        elif model_config.hf_config.model_type != "internvl_chat":
            # InternVL uses QEFFAutoModelForCausalLM (the default class)
            model_type = "imagetext"
        # Speculative decoding not supported with multimodality
        # On-device sampling is supported with multimodality
        # Continuous batching is supported for dual QPC VLMs
        # Continuous batching is not supported for audio models
        if not kv_offload or model_type == "speech":
            del args["continuous_batching"]
        if model_type == "speech":
            del args["qaic_config"]
    if model_config.runner_type == "pooling" and not model_config.is_multimodal_model:
        _task = (override_qaic_config or {}).get("task", None)
        if _task in ("score", "classify"):
            model_type = "seq_classify"
        else:
            model_type = "encode"
        del args["continuous_batching"]
        del args["qaic_config"]
        del args["kv_offload"]
        if override_qaic_config and (
            "pooling_device" in override_qaic_config
            and override_qaic_config["pooling_device"] == "qaic"
            and _task not in ("score", "classify")
        ):
            args["pooling"] = override_qaic_config["pooling_method"]
    max_retries = 8
    retry_count = 0
    import requests

    logger.info(
        "QEFF Pretrained called with %s",
        {k: v for k, v in args.items() if k != "config"},
    )
    while retry_count < max_retries:
        try:
            model_hf = QEff_class_mapping[model_type].from_pretrained(
                model_config.model, **args
            )
            break
        except requests.ReadTimeout as e:
            logger.info("HF hub read timeout: %s", e)
            retry_count += 1
        except requests.exceptions.HTTPError as e:
            retry_count = max_retries
            if e.response.status_code == 401:
                logger.error(
                    "You need to set HF_TOKEN environment"
                    " variable to download private"
                    " checkpoints."
                )
            else:
                raise e
        except Exception as e:
            logger.warning("Unable to access HF hub due to an exception: %s", e)
            retry_count += 1
    if retry_count >= max_retries:
        raise ValueError(
            f"Unable to download model {model_config.model} from Hugging face!"
        )
    return model_hf


def search_adapters_in_cache(base_model_name: str) -> list[LoRAModulePath]:
    cached_lora_module_paths = []
    hf_home = os.environ.get("HF_HOME", None)
    hf_dir = f"{hf_home}/hub"
    if not any(os.scandir(hf_home)) or not any(os.scandir(hf_dir)):
        return []
    hf_root, hf_dirs, hf_files = next(os.walk(hf_dir))
    for model_dir in [
        m for m in hf_dirs if m.startswith("models")
    ]:  # walk through all models or adapter subdirs
        root, dirs, files = next(os.walk(f"{hf_root}/{model_dir}/snapshots"))
        if len(dirs) and os.path.isfile(f"{root}/{dirs[0]}/adapter_config.json"):
            cached_lora_module_paths.append(
                LoRAModulePath(
                    name=f"{model_dir.split('--')[2]}", path=f"{root}/{dirs[0]}"
                )
            )
    # filter out adapters that are with different base models
    filtered_cached_lora_module_paths = []
    for lora_module_path in cached_lora_module_paths:
        if (
            PeftConfig.from_pretrained(lora_module_path.path).base_model_name_or_path
            == base_model_name
        ):
            filtered_cached_lora_module_paths.append(lora_module_path)
    return filtered_cached_lora_module_paths


def verify_adaptername_to_id_consistency(
    json_file_input: dict, lora_modules_input: list[LoRAModulePath]
) -> bool:
    for i in range(len(lora_modules_input)):
        if (lora_modules_input[i].name not in json_file_input) or (
            json_file_input[lora_modules_input[i].name] != i + 1
        ):
            return False
    return True


def _get_qaic_compile_config(
    vllm_config: VllmConfig,
    speculative_model_type: str = "default",
) -> QaicCompileConfig:
    mxfp6_en, mxint8_en = False, False
    # mxfp6
    if (
        isinstance(vllm_config.model_config.quantization, str)
        and vllm_config.model_config.quantization == "mxfp6"
    ):
        mxfp6_en = True
    # mxint8
    if (
        "mxint8"
        in VLLM_CACHE_DTYPE_TO_QAIC_CACHE_DTYPE[vllm_config.cache_config.cache_dtype]
    ):
        mxint8_en = True
    # Number of kv cache blocks should be same as num_gpu_blocks, if CPL==blk_size
    kv_cache_batch_size = vllm_config.cache_config.num_cpu_blocks
    prefill_only = None
    # prefill_only options
    if vllm_config.kv_transfer_config:
        kv_role = vllm_config.kv_transfer_config.kv_role
        if kv_role in ["kv_producer", "kv_consumer"]:
            prefill_only = kv_role == "kv_producer"

    _device_group = (vllm_config.additional_config or {}).get("device_group")

    # Prepare default config
    cfg: dict[str, Any] = {
        "qpc_path": None,
        "prefill_seq_len": vllm_config.scheduler_config.long_prefill_token_threshold,
        "ctx_len": vllm_config.model_config.max_model_len,
        "batch_size": 1,
        "full_batch_size": vllm_config.scheduler_config.max_num_seqs,
        "kv_cache_batch_size": kv_cache_batch_size,
        "device_group": _device_group,
        "num_devices": len(_device_group) if _device_group is not None else 1,
        "num_cores": None,
        "mxfp6_matmul": mxfp6_en,
        "mxint8_kv_cache": mxint8_en,
        "num_speculative_tokens": None,
        "aic_enable_depth_first": True,
        "mos": -1,
        "prefill_only": prefill_only,
        "compile_only": False,
    }
    # update default settings with user overrides from additional_config
    override_qaic_config: dict[str, Any] | None = (
        vllm_config.additional_config or {}
    ).get("override_qaic_config")
    cfg.update(_clean_config(override_qaic_config, vllm_config))
    # update through environment variable
    cfg.update(_clean_config(QAIC_DEVICE_CONFIG[speculative_model_type]))
    # set aic num core as per the hw if not provided
    if cfg["num_cores"] is None:
        _hw_num_cores = 16
        try:
            from qaicrt import Util as qaic_util
        except ImportError:
            import platform
            import sys

            sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
            from qaicrt import Util as qaic_util
        from qaicrt import QStatus

        if cfg["device_group"] is not None:
            for id in cfg["device_group"]:
                _nsp_info = qaic_util().getResourceInfo(id)
                if _nsp_info[0] != QStatus.QS_SUCCESS:
                    raise ValueError(f"device_id {id} is not available !!")
                _hw_num_cores = min(_hw_num_cores, _nsp_info[1].nspTotal)
        cfg["num_cores"] = _hw_num_cores
        # Applicable for draft-target spd scheme
        if (
            vllm_config.speculative_config
            and "draft" in vllm_config.speculative_config.method
        ):
            other_cfg: dict[str, Any] = {"device_group": cfg["device_group"]}
            draft_override: dict[str, Any] | None = (
                vllm_config.additional_config or {}
            ).get("draft_override_qaic_config")
            if speculative_model_type == "target" and draft_override is not None:
                other_cfg.update(_clean_config(draft_override, vllm_config))
                other_cfg.update(
                    _clean_config(
                        {
                            "device_group": os.environ.get(
                                "VLLM_QAIC_SPEC_DRAFT_QID", None
                            )
                        }
                    )
                )
            else:
                # update default settings
                _override: dict[str, Any] | None = (
                    vllm_config.additional_config or {}
                ).get("override_qaic_config")
                other_cfg.update(_clean_config(_override, vllm_config))
                other_cfg.update(
                    _clean_config(
                        {
                            "device_group": os.environ.get(
                                "VLLM_QAIC_SPEC_TARGET_QID", None
                            )
                        }
                    )
                )
            if other_cfg["device_group"] == cfg["device_group"]:
                _targetCoreCount = cdiv(_hw_num_cores, 2)
                if speculative_model_type == "target":
                    cfg["num_cores"] = _targetCoreCount
                else:
                    cfg["num_cores"] = _hw_num_cores - _targetCoreCount
    if not vllm_config.cache_config.enable_prefix_caching:
        del cfg["kv_cache_batch_size"]
    if cfg["mos"] == -1:
        del cfg["mos"]
    num_logits_to_keep = None
    if speculative_model_type in ("target", "turbo"):
        spec_cfg = vllm_config.speculative_config
        K = spec_cfg.num_speculative_tokens if spec_cfg else None
        # For ngram/suffix, compile two decode specializations: K=0 (fallback,
        # no proposals) and K=max (full SpD).  The K=0 kernel is used on steps
        # where the proposer finds no matches, avoiding the wasted 5-token
        # forward pass.  For draft_model the single K is sufficient.
        if spec_cfg and spec_cfg.method in ("ngram", "suffix") and K:
            cfg["num_speculative_tokens"] = [0, K]
        else:
            cfg["num_speculative_tokens"] = K
        num_logits_to_keep = K + 1 if K is not None else None
    else:
        del cfg["num_speculative_tokens"]
    qpc_idx = None
    kv_offload = False
    if vllm_config.model_config.is_multimodal_model:
        hf_config = vllm_config.model_config.hf_config
        if vis_cfg := getattr(hf_config, "vision_config", None):
            kv_offload = cfg.pop("kv_offload", True)
            if "height" not in cfg:
                cfg["img_size"] = getattr(vis_cfg, "image_size", 448)
            mm_kwargs = (
                vllm_config.model_config.multimodal_config.mm_processor_kwargs or {}
            )
            if max_dynamic_patch := getattr(hf_config, "max_dynamic_patch", None):
                # For InternVL
                use_thumbnail = getattr(hf_config, "use_thumbnail", False)
                cfg["num_patches"] = mm_kwargs.get(
                    "max_dynamic_patch", max_dynamic_patch
                ) + int(mm_kwargs.get("use_thumbnail", use_thumbnail))
            if max_patches := mm_kwargs.get("max_patches"):
                # For Llama4
                cfg["max_num_tiles"] = max_patches + 1
        else:
            # Audio models (e.g. Whisper):
            # QEff requires fixed prefill_seq_len=1, no batching.
            if "encoder_ctx_len" not in cfg:
                cfg["encoder_ctx_len"] = getattr(
                    hf_config, "max_source_positions", None
                )
            cfg["prefill_seq_len"] = 1

        if kv_offload:
            # Dual QPC approach: select which QPC to load based on which path is skipped.
            skip_lang = cfg.get("skip_lang", False)
            skip_vision = cfg.get("skip_vision", False)
            if not skip_lang and not skip_vision:
                # Default: skip lang for pooling (embedding), skip vision otherwise.
                skip_lang = vllm_config.model_config.runner_type == "pooling"
                skip_vision = not skip_lang
                cfg["skip_lang" if skip_lang else "skip_vision"] = True
            qpc_idx = 0 if skip_lang else 1
            if not skip_lang:
                cfg["vision_size"] = cfg["prefill_seq_len"]
        else:
            assert cfg["full_batch_size"] == 1, (
                "Multimodal models do not support batching yet. "
                "Please set `max_num_seqs` (decode batch size) to 1."
            )
            del cfg["full_batch_size"]
    qaic_config: dict[str, Any] | None = None
    if speculative_model_type in ("target", "turbo"):
        qaic_config = dict(speculative_model_type=speculative_model_type)
    # On Device Sampling
    if cfg.get("aic_include_sampler") is not None:
        if qaic_config is None:
            qaic_config = dict()
        qaic_config["include_sampler"] = cfg["aic_include_sampler"]
        if cfg.get("aic_return_pdfs") is not None:
            qaic_config["return_pdfs"] = cfg["aic_return_pdfs"]
            del cfg["aic_return_pdfs"]
        if cfg.get("max_top_k_ids") is not None:
            qaic_config["max_top_k_ids"] = min(
                int(cfg["max_top_k_ids"]),
                vllm_config.model_config.get_vocab_size(),
            )
            del cfg["max_top_k_ids"]
        if cfg.get("aic_include_guided_decoding") is not None:
            qaic_config["include_guided_decoding"] = cfg["aic_include_guided_decoding"]
            del cfg["aic_include_guided_decoding"]
        del cfg["aic_include_sampler"]
    include_sampler: bool | None = None
    return_pdfs: bool | None = None
    max_top_k_ids: int | None = None
    include_guided_decoding: bool | None = None
    if qaic_config is not None:
        include_sampler = qaic_config.get("include_sampler")
        if include_sampler is not None:
            return_pdfs = qaic_config.get("return_pdfs")
            max_top_k_ids = qaic_config.get("max_top_k_ids", 512)
            include_guided_decoding = qaic_config.get("include_guided_decoding")
    # Check CCL is enabled
    if (
        "ccl_enabled" in cfg
        or len(cfg.get("comp_ctx_lengths_prefill", [])) > 0
        or len(cfg.get("comp_ctx_lengths_decode", [])) > 0
    ):
        if qaic_config is None:
            qaic_config = dict()
        qaic_config["ccl_enabled"] = True
        if not cfg.pop("ccl_enabled", False):
            cfg["comp_ctx_lengths_prefill"] = (
                [vllm_config.model_config.max_model_len]
                if (len(cfg.get("comp_ctx_lengths_prefill", [])) == 0)
                else cfg.get("comp_ctx_lengths_prefill")
            )
            cfg["comp_ctx_lengths_decode"] = (
                [vllm_config.model_config.max_model_len]
                if (len(cfg.get("comp_ctx_lengths_decode", [])) == 0)
                else cfg.get("comp_ctx_lengths_decode")
            )
    qpc_path = cfg.pop("qpc_path")
    if qpc_path and kv_offload and len(qpc_path.split(":")) > 1:
        assert qpc_idx is not None
        qpc_path = qpc_path.split(":")[qpc_idx]
    stages = int(cfg.pop("stages", 1))
    # "mdp_num_partitions" is needed for new MDP parititoner in QEFF
    if stages != 1 and "mdp_load_partition_config" not in cfg:
        cfg["mdp_num_partitions"] = stages
    if (
        vllm_config.model_config.runner_type == "pooling"
        and not vllm_config.model_config.is_multimodal_model
    ):
        if "prefill_seq_len" in cfg:
            cfg["seq_len"] = cfg["prefill_seq_len"]
            del cfg["prefill_seq_len"]
        if "ctx_len" in cfg:
            del cfg["ctx_len"]
        if "full_batch_size" in cfg:
            cfg["batch_size"] = cfg["full_batch_size"]
            del cfg["full_batch_size"]
    if "pooling_device" in cfg:
        del cfg["pooling_device"]
    if "pooling_method" in cfg:
        del cfg["pooling_method"]
    if "normalize" in cfg:
        del cfg["normalize"]
    if "softmax" in cfg:
        del cfg["softmax"]
    if "task" in cfg:
        del cfg["task"]
    # Add kv_cache_prefix for disagg only
    if vllm_config.kv_transfer_config:
        from .qaic_session_np import VLLM_KV_CACHE_PREFIX

        cfg["kv_cache_prefix"] = VLLM_KV_CACHE_PREFIX
    print(cfg)
    device_group = cfg.pop("device_group")
    if "io_encrypt" in cfg:
        # Currently Model IP flow is broken into two separate runs
        # one to compile the model and another to run using qpc route
        cfg["compile_only"] = True
    compile_only = cfg.pop("compile_only", False)
    # prefill_only option
    if vllm_config.kv_transfer_config and "prefill_only" in cfg:
        assert (
            (kv_role == "kv_producer" and cfg["prefill_only"])
            or (kv_role == "kv_consumer" and not cfg["prefill_only"])
            or (kv_role == "kv_both" and cfg["prefill_only"] is None)
        ), (
            "prefill_only False is only supported for kv_consumer"
            "prefill_only True is only supported for kv_producer"
            "prefill_only None is only supported for kv_both"
        )
    return QaicCompileConfig(
        compile_only=compile_only,
        qpc_path=qpc_path,
        device_group=device_group,
        cfg=cfg,
        num_logits_to_keep=num_logits_to_keep,
        kv_offload=kv_offload,
        include_sampler=include_sampler,
        include_guided_decoding=include_guided_decoding,
        return_pdfs=return_pdfs,
        max_top_k_ids=max_top_k_ids,
        qaic_config=qaic_config,
        stages=stages,
    )
