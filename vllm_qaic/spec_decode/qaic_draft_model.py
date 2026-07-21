# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/spec_decode/draft_model.py

from typing import TYPE_CHECKING

import numpy as np

from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_qaic.model_loader.qaic import load_qaic_model

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


class QaicDraftModelProposer:
    """
    QAIC-specific draft model proposer for speculative decoding.

    Loads the draft model as a separate QAIC QPC (speculative_model_type="draft")
    and runs autoregressive draft token generation using CPU numpy arrays.

    Unlike the GPU DraftModelProposer (which inherits SpecDecodeBaseProposer and
    uses GPU tensors, CUDA graphs, and attention metadata builders), this class
    is standalone and operates entirely on CPU/numpy, matching the QAIC execution
    model.

    Draft proposal timing: after bookkeeping (same as ngram/suffix), because
    QAICInferenceSession.run() is fully synchronous and we need CPU token IDs.
    """

    def __init__(self, draft_vllm_config: VllmConfig) -> None:
        # The draft proposer runs prefill and autoregressive decode synchronously.
        # Async scheduling is not supported because the pending_prefill_exec_queue
        # returned by forward(is_prompt=True) would be silently discarded.
        assert not draft_vllm_config.scheduler_config.async_scheduling, (
            "QaicDraftModelProposer requires synchronous scheduling "
            "(async_scheduling=False). Async mode is not supported."
        )
        self._draft_vllm_config = draft_vllm_config
        self.num_spec_tokens = (
            draft_vllm_config.speculative_config.num_speculative_tokens
        )
        self.decode_bsz = draft_vllm_config.scheduler_config.max_num_seqs
        _override_qaic_config = {}
        if "draft_override_qaic_config" in draft_vllm_config.additional_config:
            _override_qaic_config = draft_vllm_config.additional_config.get(
                "draft_override_qaic_config", None
            )
        elif "override_qaic_config" in draft_vllm_config.additional_config:
            _override_qaic_config = draft_vllm_config.additional_config.get(
                "override_qaic_config", None
            )
        self.prefill_seq_len = _override_qaic_config.get("prefill_seq_len", 128)

    def load_model(self) -> None:
        logger.info(
            "Loading draft model %s for QAIC speculative decoding...",
            self._draft_vllm_config.model_config.model,
        )
        self.model = load_qaic_model(self._draft_vllm_config, "draft")
        vocab_size = self._draft_vllm_config.model_config.get_vocab_size()
        self._decode_logits = np.empty(
            (self.decode_bsz, 1, vocab_size)
            if self.model.logits_ndim == 3
            else (self.decode_bsz, vocab_size),
            dtype=np.float32,
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch",
        batch_indices: np.ndarray,
    ) -> list[list[int]]:
        """Propose num_spec_tokens draft tokens for each active decode request.

        Called after bookkeeping (CPU sampled token IDs are available).

        Args:
            sampled_token_ids: CPU-side sampled tokens, list[list[int]] of shape
                [num_reqs]. Each inner list contains the accepted token(s) for
                that request from the previous step.
            scheduler_output: Current scheduler output.
            input_batch: The target runner's input batch (for token IDs and
                computed token counts).
            batch_indices: QAIC batch indices (KV cache slots), shape [num_reqs].

        Returns:
            list[list[int]] of shape [num_reqs, num_spec_tokens].
        """
        req_ids = input_batch.req_ids
        num_reqs = len(req_ids)

        # Skip proposing for requests that haven't yet produced their first
        # output token.  num_tokens_no_spec is updated inside _bookkeeping_sync
        # immediately after the target model samples a token, so this becomes
        # True on the very step that the last prefill chunk completes — one step
        # earlier than the num_computed_tokens_cpu-based check (which only
        # reflects KV state from the start of the step, lagging by one for the
        # last prefill chunk).
        in_decode_phase = (
            input_batch.num_tokens_no_spec[:num_reqs]
            > input_batch.num_prompt_tokens[:num_reqs]
        )
        if not in_decode_phase.any():
            return [[] for _ in range(num_reqs)]

        # Step 1: Batch-prefill all new requests in a single model call.
        # A request needs draft-model prefill on its very first decode step.
        # After bookkeeping of the last prefill chunk, num_tokens_no_spec[i]
        # equals num_prompt_tokens[i] + 1 (exactly one token accepted beyond
        # the prompt) and in_decode_phase[i] is True — so this fires on that
        # step, the earliest possible moment.  On all subsequent steps the
        # count is higher, so the check is stateless and fires only once.  It
        # also handles preempted-and-restarted requests naturally: they return
        # to num_prompt_tokens + 1 on their first decode step after re-prefill.
        # The prefill covers all tokens up to (but not including) the accepted
        # token, which will be fed as the first decode input in Step 2.
        new_req_indices = [
            i
            for i in range(num_reqs)
            if (
                input_batch.num_tokens_no_spec[i]
                == input_batch.num_prompt_tokens[i] + 1
                and in_decode_phase[i]
            )
        ]
        if new_req_indices:
            all_tokens: list[np.ndarray] = []
            all_positions: list[np.ndarray] = []
            new_batch_idxs: list[int] = []
            cum_counts: list[int] = []
            for i in new_req_indices:
                num_tokens = int(input_batch.num_tokens_no_spec[i])
                # token_ids_cpu has shape [max_num_reqs, max_model_len+num_spec_tokens]
                toks = input_batch.token_ids_cpu[i, : num_tokens - 1].astype(np.int64)
                if (n := len(toks)) == 0:
                    continue
                all_tokens.append(toks)
                all_positions.append(np.arange(n, dtype=np.int64))
                new_batch_idxs.append(int(batch_indices[i]))
                cum_counts.append(n)
            if all_tokens:
                self.model(
                    input_ids=np.concatenate(all_tokens),
                    positions=np.concatenate(all_positions),
                    batch_indices=np.array(new_batch_idxs, dtype=np.int64),
                    is_prompt=True,
                    prefill_cum_sum=np.cumsum(cum_counts, dtype=np.int64),
                    logits=self._decode_logits,
                )

        # Step 2: Autoregressive draft generation for num_spec_tokens steps.
        # current_token_ids[i] = the token to feed as input for the next draft step.
        # For step 0, this is the last accepted token from sampled_token_ids.
        current_token_ids = np.full(num_reqs, -1, dtype=np.int64)
        for i, toks in enumerate(sampled_token_ids):
            if toks:
                current_token_ids[i] = toks[-1]

        # current_positions[i] = position of the accepted token being fed as the
        # first decode input. After bookkeeping, num_tokens_no_spec[i] includes the
        # accepted token, so its position is num_tokens_no_spec[i] - 1.
        current_positions = (
            input_batch.num_tokens_no_spec[:num_reqs].copy().astype(np.int64) - 1
        )

        # KV cache catch-up: when all num_spec_tokens were accepted in the previous
        # verification step, the target model wrote KV for the last draft token at
        # position (current_positions[i] - 1) as a byproduct of its batched forward
        # pass. The draft model's propose loop only writes KV at positions
        # [Q, Q+1, ..., Q+num_spec-1] — it never decodes the last draft token
        # d_{num_spec} itself, leaving KV[Q+num_spec-1+1 = current_positions[i]-1]
        # as stale/uninitialized. This causes wrong attention when the bonus token is
        # fed at current_positions[i] and the attention reads the garbage KV entry.
        #
        # Fix: when all spec tokens were accepted (detected by
        # len(sampled_token_ids[i]) == num_spec_tokens + 1), run an extra decode of
        # the last accepted token (sampled_token_ids[i][-2]) at
        # current_positions[i] - 1 to populate KV before the propose loop.
        catchup_indices = [
            i
            for i, toks in enumerate(sampled_token_ids)
            if len(toks) == self.num_spec_tokens + 1
        ]
        if catchup_indices:
            catchup_input_ids = np.array(
                [sampled_token_ids[i][-2] for i in catchup_indices], dtype=np.int64
            )
            catchup_positions = np.array(
                [int(current_positions[i]) - 1 for i in catchup_indices],
                dtype=np.int64,
            )
            catchup_batch_indices = np.array(
                [int(batch_indices[i]) for i in catchup_indices], dtype=np.int64
            )
            self.model(
                input_ids=catchup_input_ids,
                positions=catchup_positions,
                batch_indices=catchup_batch_indices,
                is_prompt=False,
                logits=self._decode_logits,
            )

        draft_token_ids: list[list[int]] = [[] for _ in range(num_reqs)]

        # batch_indices and num_reqs are loop-invariant; compute the slice once.
        decode_batch_indices = batch_indices[:num_reqs]  # shape: (num_reqs,)

        for step in range(self.num_spec_tokens):
            # Build decode inputs: shape (num_reqs,) flat arrays.
            # _run_decode expects flat input_ids of length
            # num_decodes * max_decode_tokens.
            # For the draft model, max_decode_tokens = 1, so shape is (num_reqs,).
            #
            # First-step invariant: at step=0, current_token_ids[i] is the sampled
            # (bonus) token produced by the target model's verification pass — i.e.
            # the token the target chose at the first unverified position. The draft
            # model reads the KV cache at current_positions[i]-1 and predicts
            # current_positions[i]+1. This invariant holds because catch-up above
            # ensures draft KV[current_positions[i]-1] is populated before the loop.
            decode_input_ids = current_token_ids  # shape: (num_reqs,)
            decode_positions = current_positions  # shape: (num_reqs,)

            # Run draft model decode; results are written into self._decode_logits.
            self.model(
                input_ids=decode_input_ids,
                positions=decode_positions,
                batch_indices=decode_batch_indices,
                is_prompt=False,
                logits=self._decode_logits,
            )

            # Greedy sampling: argmax over vocab dimension.
            # self._decode_logits shape: (decode_bsz, 1, vocab_size) or
            # (decode_bsz, vocab_size)
            raw = self._decode_logits[:num_reqs]
            next_tokens = np.argmax(
                raw[:, 0, :] if raw.ndim == 3 else raw, axis=-1
            ).astype(np.int64)

            # Collect draft tokens for this step.
            # Only store tokens for requests in decode phase; prefill requests
            # have in_decode_phase[i]=False and return [] (already initialized).
            for i in range(num_reqs):
                if in_decode_phase[i]:
                    draft_token_ids[i].append(int(next_tokens[i]))

            # Advance for next step.
            current_token_ids = next_tokens
            current_positions = current_positions + 1

        return draft_token_ids
