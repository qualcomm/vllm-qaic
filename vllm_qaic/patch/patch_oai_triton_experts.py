# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Patch upstream OAITritonExperts for QAIC activation and routing dtypes.

The upstream wrappers require BF16 activations and use int16 routing IDs and
int64 remapping output. Replace ``_make_routing_data`` and ``apply`` to retain
the model's activation dtype and use int32 indices on QAIC. Both expert matmuls
still run the existing ``matmul_ogs`` kernels with fused SwiGLU and reduction.

Imported by QaicTritonGptOssMxfp4MoEMethod after validating triton_kernels,
when VLLM_QAIC_TRITON_FUSED_MOE selects the GPT-OSS Triton backend.
"""

import torch
import triton

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    OAITritonExperts,
    _remap_topk_to_local_kernel,
    pack_bitmatrix,
    use_legacy_triton_kernels,
)
from triton_kernels.matmul_ogs import (
    FnSpecs,
    FusedActivation,
    GatherIndx,
    RoutingData,
    ScatterIndx,
    matmul_ogs,
)
from triton_kernels.swiglu import swiglu_fn
from triton_kernels.tensor import BIT, Bitmatrix, wrap_torch_tensor

# vLLM's pinned v3.5.1 package predates SparseMatrix and ragged metadata.
# Follow its API selection so the QAIC expert patch works with
# both the bundled kernels and newer compatible installations.
if not use_legacy_triton_kernels:
    from triton_kernels.tensor import SparseMatrix, make_ragged_tensor_metadata


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

    # The packing kernel writes uint32 bit patterns, and triton_kernels'
    # vpopc helper requires loads from the bitmatrix to have tl.uint32 dtype.
    # Keep the QAIC-supported int32 allocation above, then reinterpret the
    # identically sized storage without copying it.
    bitmatrix = bitmatrix.view(torch.uint32)

    bitmatrix_shape = [n_rows, bm_cols * 32]
    bitmatrix_shape_max = [n_rows, None]

    # matmul_ogs expects invalid routing weights to contain -1.
    topk_weights = torch.where(topk_ids == -1, -1.0, topk_weights)

    if use_legacy_triton_kernels:
        from triton_kernels.routing import routing_from_bitmatrix

        # The legacy wrapper takes logical shape information in its
        # constructor; wrap_torch_tensor does not accept shape kwargs here.
        bitmatrix = Bitmatrix(
            bitmatrix,
            shape=bitmatrix_shape,
            shape_max=bitmatrix_shape_max,
            scratchpad=None,
        )
        return routing_from_bitmatrix(
            bitmatrix, topk_weights, topk_ids, num_local_experts, num_topk
        )

    bitmatrix = wrap_torch_tensor(
        bitmatrix,
        dtype=BIT,
        shape=bitmatrix_shape,
        shape_max=bitmatrix_shape_max,
    )

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


def _qaic_make_routing_data(
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


def _qaic_apply(
    self,
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1,
    w2,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
    a1q_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    expert_tokens_meta,
    apply_router_weight_on_input: bool,
) -> None:
    """Launch upstream matmul kernels without the wrapper's BF16 restriction.

    The gate/up matmul fuses SwiGLU, and the down matmul applies routing
    weights and scatter reduction. Reuse the upstream workspace and kernels
    while retaining the model's activation dtype and int32 routing storage.
    """
    assert activation == MoEActivation.SWIGLUOAI
    assert hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
    quant_config = self.quant_config
    assert quant_config is not None
    assert quant_config.w1_bias is None or quant_config.w1_bias.dtype == torch.float32
    assert quant_config.w2_bias is None or quant_config.w2_bias.dtype == torch.float32
    assert hidden_states.shape[-1] == w1.shape[-2]
    assert w2.shape[-1] == w1.shape[1]

    if expert_map is not None:
        # Upstream's remapping wrapper allocates int64 output. The same
        # kernel can store the local expert IDs in QAIC-supported int32.
        local_ids = torch.empty_like(topk_ids, dtype=torch.int32)
        n = topk_ids.numel()
        _remap_topk_to_local_kernel[(triton.cdiv(n, 1024),)](
            topk_ids, expert_map, local_ids, n, BLOCK=1024
        )
        topk_ids = local_ids

    local_num_experts, m, n, k, topk = self.moe_problem_size(
        hidden_states, w1, w2, topk_ids
    )
    routing_data, gather_indx, scatter_indx = self._make_routing_data(
        topk_ids, topk_weights, local_num_experts
    )

    intermediate_size = m * topk * (n // 2)
    intermediate = workspace2.view(-1)[:intermediate_size].view(1, m * topk, n // 2)
    output_view = output.view(1, m, k)
    alpha = quant_config.gemm1_alpha
    limit = quant_config.gemm1_clamp_limit
    activation_args = (
        1.702 if alpha is None else alpha,
        7.0 if limit is None else limit,
    )
    if use_legacy_triton_kernels:
        fused_activation = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), activation_args, 2
        )
    else:
        fused_activation = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            activation_args,
        )

    # Match OAITritonExperts: apply router weights after the down matmul.
    matmul_ogs(
        hidden_states,
        w1,
        quant_config.w1_bias,
        routing_data,
        gather_indx=gather_indx,
        precision_config=quant_config.w1_precision,
        fused_activation=fused_activation,
        y=intermediate,
    )
    matmul_ogs(
        intermediate.view(m * topk, n // 2),
        w2,
        quant_config.w2_bias,
        routing_data,
        scatter_indx=scatter_indx,
        precision_config=quant_config.w2_precision,
        gammas=routing_data.gate_scal,
        y=output_view,
    )


OAITritonExperts._make_routing_data = (  # type: ignore[method-assign]
    _qaic_make_routing_data
)
OAITritonExperts.apply = _qaic_apply  # type: ignore[method-assign]
