# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Startup warm-up for the upstream n-gram proposer's Numba kernel (AOT).

Upstream ``NgramProposer.__init__`` warms up with empty ``sampled_token_ids``,
so ``batch_propose_numba`` is never called and its JIT compile (~1 s) lands on
the first real decode step.  Numba also specializes per argument type, and the
AOT runner swaps ``token_ids_cpu`` for an int64 (-1 padded) table, so the
warm-up must use arrays with the production dtype/ndim/layout.  It does so by
slicing the live buffers' first row into scratch copies.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm_qaic.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig
    from vllm.v1.spec_decode.ngram_proposer import NgramProposer

logger = init_logger(__name__)

# Short repeating context: the suffix "1 2 3" matches earlier, so the KMP
# search proposes a draft and the whole kernel body gets compiled.
_WARMUP_PATTERN = (1, 2, 3)
_WARMUP_REPEATS = 3


def warm_up_ngram_proposer(
    drafter: NgramProposer,
    token_ids_cpu: np.ndarray,
    num_tokens_no_spec: np.ndarray,
) -> float:
    """Compile ``batch_propose_numba`` for these arrays' types; returns seconds.

    Live arrays are never written: the drafter sees scratch copies of their
    first row, and its per-request draft buffers are restored afterwards.
    """
    t0 = time.perf_counter()
    tok = np.full_like(token_ids_cpu[:1], -1)
    ntok = np.zeros_like(num_tokens_no_spec[:1])
    context = np.tile(_WARMUP_PATTERN, _WARMUP_REPEATS)
    n = min(len(context), tok.shape[1], drafter.max_model_len - 1)
    tok[0, :n] = context[:n]
    ntok[0] = n

    saved_draft = drafter.valid_ngram_draft[:1].copy()
    saved_num_drafts = drafter.valid_ngram_num_drafts[:1].copy()
    try:
        drafter.propose([[0]], ntok, tok, slot_mappings=None)
    finally:
        drafter.valid_ngram_draft[:1] = saved_draft
        drafter.valid_ngram_num_drafts[:1] = saved_num_drafts
    dt = time.perf_counter() - t0
    logger.info("AOT ngram proposer numba warm-up: %.3f s", dt)
    return dt


def maybe_warm_up_ngram_proposer(
    speculative_config: SpeculativeConfig | None,
    drafter: Any,
    input_batch: Any,
) -> float | None:
    """Warm up the n-gram drafter if one is active; never raises.

    Returns the warm-up seconds, or None when skipped or failed.  A failure
    only means the first request compiles lazily, as it would without this
    warm-up, so it is logged rather than allowed to abort server start-up.
    """
    if (
        speculative_config is None
        or speculative_config.method != "ngram"
        or drafter is None
    ):
        return None
    try:
        return warm_up_ngram_proposer(
            drafter, input_batch.token_ids_cpu, input_batch.num_tokens_no_spec
        )
    except Exception:
        logger.warning(
            "AOT ngram proposer numba warm-up failed; the first request will "
            "compile the kernel instead.",
            exc_info=True,
        )
        return None
