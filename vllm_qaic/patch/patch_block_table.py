# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""Patch BlockTable.compute_slot_mapping to bypass the Triton kernel on QAIC.

The upstream implementation uses a @triton.jit kernel launched with grid syntax
kernel[(grid)](args), which fails on QAIC because Triton is not available.
This replaces it with a pure PyTorch/numpy equivalent.
"""

import numpy as np
import torch
import vllm.v1.worker.block_table
from vllm.v1.worker.block_table import BlockTable, PAD_SLOT_ID


def _qaic_compute_slot_mapping(
    self,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    num_tokens = positions.shape[0]
    query_start_loc_np = query_start_loc.cpu().numpy()
    positions_np = positions.cpu().numpy()
    token_counts = np.diff(query_start_loc_np[: num_reqs + 1])
    req_indices = np.repeat(np.arange(num_reqs, dtype=np.int64), token_counts)

    # Pad remaining slots for v23's compute_slot_mapping contract.
    self.slot_mapping.np[num_tokens:] = PAD_SLOT_ID

    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
    # where K is the max_num_blocks_per_req and the block size is 2.
    # NOTE(woosuk): We can't simply use `token_indices // block_size`
    # here because M (max_model_len) is not necessarily divisible by
    # block_size.
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    if total_cp_world_size > 1:
        # Note(hc): The DCP implement store kvcache with an interleave
        # style, the kvcache for the token whose token_idx is i is
        # always stored on the GPU whose dcp_rank equals i % cp_world_size:

        # Use a "virtual block" which equals to world_size * block_size
        # for block_table_indices calculation.
        virtual_block_size = self.block_size * total_cp_world_size
        block_table_indices = (
            req_indices * self.max_num_blocks_per_req
            + positions_np // virtual_block_size
        )

        block_numbers = self.block_table.np.ravel()[block_table_indices]
        # Use virtual_block_size for mask calculation, which marks local
        # tokens.
        virtual_block_offsets = positions_np % virtual_block_size
        mask = (
            virtual_block_offsets
            // self.cp_kv_cache_interleave_size
            % total_cp_world_size
            == total_cp_rank
        )
        # Calculate local block_offsets
        block_offsets = (
            virtual_block_offsets
            // (total_cp_world_size * self.cp_kv_cache_interleave_size)
            * self.cp_kv_cache_interleave_size
            + virtual_block_offsets % self.cp_kv_cache_interleave_size
        )
        # Calculate slot_mapping
        slot_mapping = block_numbers * self.block_size + block_offsets
        # Write final slots, use -1 for not-local
        self.slot_mapping.np[: req_indices.shape[0]] = np.where(
            mask, slot_mapping, -1
        )
    else:
        block_table_indices = (
            req_indices * self.max_num_blocks_per_req + positions_np // self.block_size
        )

        block_numbers = self.block_table.np.ravel()[block_table_indices]
        block_offsets = positions_np % self.block_size
        np.add(
            block_numbers * self.block_size,
            block_offsets,
            out=self.slot_mapping.np[: req_indices.shape[0]],
        )

    self.slot_mapping.copy_to_gpu()


BlockTable.compute_slot_mapping = _qaic_compute_slot_mapping
