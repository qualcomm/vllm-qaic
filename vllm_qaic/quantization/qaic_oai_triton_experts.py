# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""QAIC routing dtype overrides for the OAI Triton GPT-OSS experts."""

import torch
import triton

from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    OAITritonExperts,
    pack_bitmatrix,
)
from triton_kernels.matmul_ogs import GatherIndx, RoutingData, ScatterIndx
from triton_kernels.tensor import (
    SparseMatrix,
    make_ragged_tensor_metadata,
    wrap_torch_tensor,
)
from triton_kernels.tensor_details.dtype import BIT


def make_routing_data_qaic(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_local_experts: int,
    params_dtype: torch.dtype,
):
    """Build OAI routing data using tensor dtypes supported by QAIC."""
    # Upstream uses int16 IDs and BF16 weights as storage optimizations. QAIC
    # cannot materialize either dtype, so use its supported 32-bit index type
    # and the layer's resolved parameter/activation dtype.
    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(params_dtype)

    n_rows, num_topk = topk_ids.size()
    block_size_m = 512
    block_size_k = 32

    bm_cols = triton.cdiv(num_local_experts, block_size_k)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.int32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, block_size_m),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_K=block_size_k,
    )

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]
    bitmatrix = wrap_torch_tensor(
        bitmatrix,
        dtype=BIT,
        shape=bitmatrix_shape,
        shape_max=bitmatrix_shape_max,
    )

    # matmul_ogs expects invalid routing weights to contain -1.
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)

    sparse_logits = SparseMatrix(
        indx=topk_ids,
        vals=topk_weights,
        mask=bitmatrix,
    )
    dispatch_indx = sparse_logits.mask_metadata.row_sorted_indx
    combine_indx = sparse_logits.mask_metadata.col_sorted_indx
    ragged_batch_metadata = make_ragged_tensor_metadata(
        sparse_logits.mask_metadata.col_sum,
        dispatch_indx.shape[0],
    )
    gate_scal = sparse_logits.vals.flatten()[combine_indx]
    routing_data = RoutingData(
        gate_scal,
        ragged_batch_metadata.block_sizes,
        num_local_experts,
        num_topk,
        ragged_batch_metadata,
    )
    gather_indx = GatherIndx(combine_indx, dispatch_indx)
    scatter_indx = ScatterIndx(dispatch_indx, combine_indx)
    return routing_data, gather_indx, scatter_indx


class QaicOAITritonExperts(OAITritonExperts):
    """OAI Triton experts using QAIC-compatible routing tensor dtypes."""

    def _make_routing_data(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_local_experts: int,
    ):
        return make_routing_data_qaic(
            topk_ids,
            topk_weights,
            num_local_experts,
            self.moe_config.in_dtype,
        )
