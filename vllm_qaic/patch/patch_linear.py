# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Route the unquantized linear/GEMM family to vLLM's Triton GEMM.

The linear family (QKV / Column / Row / MergedColumn ParallelLinear) and the
LM-head / logits matmul all dispatch their GEMM through the ``apply`` method of
``UnquantizedLinearMethod`` (linear layers) and ``UnquantizedEmbeddingMethod``
(ParallelLMHead / VocabParallelEmbedding). Neither inherits ``CustomOp``, so
there is no ``forward_oot`` hook — instead we reassign ``apply`` (the plugin's
established monkey-patch idiom, cf. patch_rejection_sampler / patch_block_table)
to a wrapper that calls ``linear_batch_invariant`` when
``VLLM_QAIC_TRITON_LINEAR`` is set, and otherwise defers to the original method
(``dispatch_unquantized_gemm`` -> ``F.linear`` / QAIC path).

``linear_batch_invariant(x, weight, bias)`` does
``matmul_batch_invariant(x, weight.t())`` (+ bias), handling the ND x 2D linear
shape, so all four linear variants and the logits matmul route through it.

Note: this file is imported only when ``VLLM_QAIC_TRITON_LINEAR`` is set (see
vllm_qaic/patch/__init__.py), but the wrapper still re-checks the flag per call
so the flag remains authoritative. The QAIC platform must implement
``num_compute_units`` (see platform_base.py) or matmul_persistent raises
NotImplementedError at launch.
"""

from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

from vllm_qaic.ops._triton_flags import triton_op_enabled

_orig_linear_apply = UnquantizedLinearMethod.apply
_orig_embedding_apply = UnquantizedEmbeddingMethod.apply


def _qaic_linear_apply(self, layer, x, bias=None):
    if triton_op_enabled("LINEAR"):
        return linear_batch_invariant(x, layer.weight, bias)
    return _orig_linear_apply(self, layer, x, bias)


def _qaic_embedding_apply(self, layer, x, bias=None):
    if triton_op_enabled("LINEAR"):
        return linear_batch_invariant(x, layer.weight, bias)
    return _orig_embedding_apply(self, layer, x, bias)


UnquantizedLinearMethod.apply = _qaic_linear_apply
UnquantizedEmbeddingMethod.apply = _qaic_embedding_apply
