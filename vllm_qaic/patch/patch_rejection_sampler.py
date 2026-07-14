# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/sample/rejection_sampler.py
"""Monkey-patch for vllm.v1.sample.rejection_sampler.RejectionSampler.

QAIC-specific optimizations in the rejection-sampling forward pass:

  1. Skip-clone optimization:
     When all requests are greedy AND no logprobs are needed, the tensor
     clone before apply_logits_processors is unnecessary because the raw
     logits are never read again.  Skipping the clone avoids a memory
     allocation on the hot path.

  2. Skip-softmax optimization:
     For greedy decoding argmax(logits) == argmax(softmax(logits)), so
     computing the probability distribution is unnecessary.  Pass the
     logits directly as `target_probs` to rejection_sample, which uses
     target_probs only for argmax when all_greedy=True.
"""

from dataclasses import replace

import torch
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import (
    MAX_SPEC_LEN,
    RejectionSampler,
    apply_sampling_constraints,
    rejection_sample,
)


def _qaic_forward(
    self,
    metadata,
    draft_probs,
    logits,
    sampling_metadata,
) -> SamplerOutput:
    """Forward pass with QAIC greedy-path optimizations (skip-clone, skip-softmax)."""
    assert metadata.max_spec_len <= MAX_SPEC_LEN

    bonus_logits_indices = metadata.bonus_logits_indices
    target_logits_indices = metadata.target_logits_indices

    assert logits is not None
    bonus_logits = logits[bonus_logits_indices]
    bonus_sampler_output = self.sampler(
        logits=bonus_logits,
        sampling_metadata=replace(
            sampling_metadata,
            max_num_logprobs=-1,
        ),
        predict_bonus_token=True,
        logprobs_mode_override="processed_logits"
        if self.is_processed_logprobs_mode
        else "raw_logits",
    )
    bonus_token_ids = bonus_sampler_output.sampled_token_ids

    raw_target_logits = logits[target_logits_indices]
    raw_target_logits = raw_target_logits.to(torch.float32)
    target_logits = raw_target_logits
    if not self.is_processed_logprobs_mode:
        # Clone raw_target_logits before applying processors to preserve
        # the original raw logits for logprobs computation, since
        # apply_logits_processors modifies the tensor in-place.
        # --- BEGIN QAIC modification: skip-clone optimisation ---
        needs_raw_logits = sampling_metadata.max_num_logprobs is not None
        if not sampling_metadata.all_greedy or needs_raw_logits:
            target_logits = target_logits.clone()
        # --- END QAIC modification ---
    target_logits = self.apply_logits_processors(
        target_logits, sampling_metadata, metadata
    )
    target_logits = apply_sampling_constraints(
        target_logits,
        metadata.cu_num_draft_tokens,
        sampling_metadata,
    )
    # --- BEGIN QAIC modification: skip-softmax optimisation ---
    # For greedy decoding argmax(logits) == argmax(softmax(logits)).
    if sampling_metadata.all_greedy:
        target_probs = target_logits
    else:
        target_probs = target_logits.softmax(dim=-1, dtype=torch.float32)
    # --- END QAIC modification ---

    output_token_ids = rejection_sample(
        metadata.draft_token_ids,
        metadata.num_draft_tokens,
        metadata.max_spec_len,
        metadata.cu_num_draft_tokens,
        draft_probs,
        target_probs,
        bonus_token_ids,
        sampling_metadata,
    )

    logprobs_tensors = None
    if sampling_metadata.max_num_logprobs is not None:
        logprobs_tensors = self._get_logprobs_tensors(
            sampling_metadata.max_num_logprobs,
            metadata,
            logits,
            target_logits if self.is_processed_logprobs_mode else raw_target_logits,
            bonus_sampler_output.logprobs_tensors.logprobs,
            output_token_ids,
        )

    return SamplerOutput(
        sampled_token_ids=output_token_ids,
        logprobs_tensors=logprobs_tensors,
    )


RejectionSampler.forward = _qaic_forward
