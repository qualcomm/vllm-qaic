# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""VocabParallelEmbedding routed to the fused QAIC mask+remap+gather kernel."""

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)


class QAicVocabParallelEmbedding(VocabParallelEmbedding):
    """VocabParallelEmbedding using ``torch.ops.qaic.vocab_parallel_embedding``.

    The op fuses the vocab-range mask, index remap and row gather (plus the
    zero-fill of rows owned by other ranks) into one NSP dispatch, replacing
    ``get_masked_input_and_mask`` + ``F.embedding`` + ``masked_fill_``.

    Benchmarked against the upstream path on QAIC (fp16, shard 64k-76k rows x
    3584-4096, 1..4096 tokens): 4.3x-7x faster for TP>1 and 1.1x-2.2x faster
    even for TP=1 (where the upstream path is a single ``F.embedding``), so the
    fused op is used for every TP size. Quantized embedding methods, odd dtypes
    and non-row-major weights keep the upstream implementation.
    """

    def _qaic_fused_ok(self, input_: torch.Tensor) -> bool:
        w = self.weight
        return (
            isinstance(self.quant_method, UnquantizedEmbeddingMethod)
            and input_.device.type == "qaic"
            and input_.dtype in (torch.int32, torch.int64)
            and w.dim() == 2
            and w.stride(-1) == 1
            and w.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )

    def forward(self, input_):
        if not self._qaic_fused_ok(input_):
            return super().forward(input_)

        si = self.shard_indices
        output_parallel = torch.ops.qaic.vocab_parallel_embedding(
            input_,
            self.weight,
            si.org_vocab_start_index,
            si.org_vocab_end_index,
            si.num_org_vocab_padding,
            si.added_vocab_start_index,
            si.added_vocab_end_index,
        )
        if self.tp_size == 1:
            return output_parallel
        if self.parallel_group is not None:
            return self.parallel_group.all_reduce(output_parallel)
        return tensor_model_parallel_all_reduce(output_parallel)
