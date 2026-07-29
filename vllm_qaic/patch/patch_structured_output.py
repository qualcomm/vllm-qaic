# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Monkey-patch for vllm.v1.structured_output.utils.apply_grammar_bitmask.

QAIC (PyT mode) is a non-CUDA device, so the upstream implementation fails
two ways when applying the xgrammar bitmask:

  1. It builds the index tensor with ``pin_memory=True``. Pinned host memory
     requires a CUDA allocator, which QAIC does not have, raising
     "Need to provide pin_memory allocator to use pin memory".

  2. It passes that tensor as ``indices`` to
     ``xgr.apply_token_bitmask_inplace``. On a non-CUDA device xgrammar
     dispatches to the CPU kernel (``apply_token_bitmask_inplace_cpu``),
     which only accepts a python ``Sequence[int]``, not a torch.Tensor.

How:
    Replace ``apply_grammar_bitmask`` with a QAIC-safe version that gates the
    pinned-tensor path on ``is_pin_memory_available()`` (False on QAIC) and
    passes the plain python list of indices otherwise. Behaviour is identical
    to upstream on CUDA.
"""

import numpy as np
import torch

from vllm.utils.import_utils import LazyLoader
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.structured_output import utils as _so_utils
from vllm.v1.worker.gpu_input_batch import InputBatch

xgr = LazyLoader("xgr", globals(), "xgrammar")


def _qaic_apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """
    Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.
    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask = np.full(
        shape=(logits.shape[0], grammar_bitmask.shape[1]),
        fill_value=-1,
        dtype=grammar_bitmask.dtype,
    )
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_spec_tokens = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + num_spec_tokens):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + num_spec_tokens

    # Copy async to device as tensor.
    grammar_bitmask = torch.from_numpy(sorted_bitmask).to(
        logits.device, non_blocking=True
    )

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    index_tensor = None
    if not skip_out_indices:
        # xgrammar expects a python list of indices but it will actually work with
        # a tensor. If we copy the tensor ourselves here we can do it in a non_blocking
        # manner and there should be no cpu sync within xgrammar.
        # QAIC (non-CUDA) has no pinned-memory allocator and its CPU xgrammar
        # kernel only accepts a python Sequence[int], so pass the plain list there.
        if is_pin_memory_available():
            index_tensor = torch.tensor(
                out_indices, dtype=torch.int32, device="cpu", pin_memory=True
            )
            index_tensor = index_tensor.to(logits.device, non_blocking=True)
        else:
            index_tensor = out_indices

    xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)


# Patch the definition module.
_so_utils.apply_grammar_bitmask = _qaic_apply_grammar_bitmask

# Callers that did `from ...utils import apply_grammar_bitmask` bind the name at
# import time, so also rebind it in any module that already imported it.
try:
    from vllm.v1.worker import gpu_model_runner as _gpu_mr

    if hasattr(_gpu_mr, "apply_grammar_bitmask"):
        _gpu_mr.apply_grammar_bitmask = _qaic_apply_grammar_bitmask
except Exception:  # pragma: no cover - defensive; gpu_model_runner may be absent
    pass
