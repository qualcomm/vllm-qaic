# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
"""QAIC DFlash speculative decoding proposer."""

from typing import TYPE_CHECKING

import numpy as np

from vllm.config import VllmConfig
from vllm_qaic.logger import init_logger
from vllm_qaic.model_loader.qaic import load_qaic_model

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch

logger = init_logger(__name__)


def _extract_mask_token_id(dlm_hf) -> int:
    """Read mask_token_id from _dflash_mask_token_id or dflash_config."""
    val = getattr(dlm_hf, "_dflash_mask_token_id", None)
    if val is not None:
        return int(val)
    dflash_cfg = getattr(dlm_hf, "dflash_config", None)
    if dflash_cfg is not None and not isinstance(dflash_cfg, dict):
        dflash_cfg = dflash_cfg.to_dict()
    if isinstance(dflash_cfg, dict) and "mask_token_id" in dflash_cfg:
        return int(dflash_cfg["mask_token_id"])
    raise ValueError(
        "DFlash: could not determine mask_token_id from the DLM config "
        "(_dflash_mask_token_id / dflash_config.mask_token_id both missing)."
    )


class _ReqState:
    """Per-request DFlash state."""

    __slots__ = (
        "dlm_candidates",
        "position_counter",
        "candidates_from_prefill",
    )

    def __init__(self) -> None:
        self.dlm_candidates: np.ndarray | None = None
        self.position_counter: int = 0
        self.candidates_from_prefill: bool = False


class QaicDFlashProposer:
    """DFlash draft-model proposer for QAIC."""

    def __init__(self, draft_vllm_config: VllmConfig) -> None:
        assert not draft_vllm_config.scheduler_config.async_scheduling, (
            "QaicDFlashProposer requires synchronous scheduling "
            "(async_scheduling=False)."
        )
        self._draft_vllm_config = draft_vllm_config
        spec_config = draft_vllm_config.speculative_config
        self.block_size: int = spec_config.num_speculative_tokens
        self.decode_bsz: int = draft_vllm_config.scheduler_config.max_num_seqs

        # Runner sets tlm_prefill_seq_len from the loaded TLM before load_model().
        add_cfg = draft_vllm_config.additional_config or {}
        tgt_qaic_cfg = add_cfg.get("override_qaic_config") or {}
        self.tlm_prefill_seq_len: int = int(tgt_qaic_cfg.get("prefill_seq_len", 0))
        self.num_sub_blocks: int = 0

        dlm_hf = draft_vllm_config.model_config.hf_config
        self.mask_token_id: int = _extract_mask_token_id(dlm_hf)
        self.hidden_size: int = draft_vllm_config.model_config.get_hidden_size()
        self.vocab_size: int = draft_vllm_config.model_config.get_vocab_size()
        self.max_model_len: int = draft_vllm_config.model_config.max_model_len

        self._mask_row = np.full(
            (self.block_size,),
            self.mask_token_id,
            dtype=np.int64,
        )

        # Reusable DLM input buffers; inactive slots carry sentinels so the
        # hardware skips their KV writes.
        self._dlm_input_ids = np.tile(self._mask_row, (self.decode_bsz, 1))
        self._dlm_position_ids = np.full(
            (self.decode_bsz, self.block_size), -1, dtype=np.int64
        )
        self._dlm_position_ids_target = np.full(
            (self.decode_bsz, self.block_size), -1, dtype=np.int64
        )
        self._dlm_target_hidden = np.zeros(
            (self.decode_bsz, self.block_size, self.hidden_size),
            dtype=np.float32,
        )
        self._dlm_batch_index = np.full((self.decode_bsz, 1), -1, dtype=np.int64)

        self._dlm_logits_buf: np.ndarray | None = None
        self._req_state: dict[str, _ReqState] = {}
        self._prefill_pending: list[dict] = []

    def load_model(self) -> None:
        assert self.tlm_prefill_seq_len > 0 and (
            self.tlm_prefill_seq_len % self.block_size == 0
        ), (
            "DFlash requires a TLM prefill_seq_len that is a positive multiple of "
            f"block_size; got prefill_seq_len={self.tlm_prefill_seq_len}, "
            f"block_size={self.block_size}. The runner must set "
            "drafter.tlm_prefill_seq_len before load_model()."
        )
        self.num_sub_blocks = self.tlm_prefill_seq_len // self.block_size
        logger.info(
            "Loading DFlash DLM %s (block_size=%d, decode_bsz=%d, "
            "tlm_prefill_seq_len=%d, mask_token_id=%d) ...",
            self._draft_vllm_config.model_config.model,
            self.block_size,
            self.decode_bsz,
            self.tlm_prefill_seq_len,
            self.mask_token_id,
        )
        self.model = load_qaic_model(self._draft_vllm_config, "draft")
        # DLM logits output shape: (decode_bsz, block_size, vocab).
        self._dlm_logits_buf = np.zeros(
            (self.decode_bsz, self.block_size, self.vocab_size),
            dtype=np.float32,
        )

    def _state_for(self, req_id: str) -> _ReqState:
        st = self._req_state.get(req_id)
        if st is None:
            st = _ReqState()
            self._req_state[req_id] = st
        return st

    def _gc_finished_reqs(self, alive: set[str]) -> None:
        """Drop state for requests no longer in the input batch."""
        for req_id in list(self._req_state.keys()):
            if req_id not in alive:
                del self._req_state[req_id]

    def _reset_dlm_inputs(self) -> None:
        """Reset inactive-slot sentinels before active rows are filled."""
        self._dlm_input_ids[:] = self._mask_row
        self._dlm_position_ids[:] = -1
        self._dlm_position_ids_target[:] = -1
        self._dlm_batch_index[:] = -1

    def build_prefill_pending(
        self,
        prefill_cum_sum: np.ndarray,
        prefill_positions: np.ndarray,
        prefill_block_ids: np.ndarray,
        prefill_is_partial: np.ndarray,
        hidden_states_prefill: np.ndarray | None,
        hidden_state_chunks: list[np.ndarray],
        prefill_req_ids: list[str],
    ) -> None:
        """Map per-chunk TLM hidden states into DLM prefill work items."""
        pending: list[dict] = []
        chunk_idx = 0
        tok_start = 0
        for req_i, tok_end in enumerate(prefill_cum_sum):
            req_n_tokens = int(tok_end) - tok_start
            req_positions = prefill_positions[..., tok_start:tok_end]
            req_block_ids = prefill_block_ids[req_i : req_i + 1]
            req_logits = (
                hidden_states_prefill[req_i : req_i + 1]
                if hidden_states_prefill is not None
                else None
            )
            req_id = prefill_req_ids[req_i]
            pending.append(
                {
                    "target_hidden": hidden_state_chunks[chunk_idx],
                    "prefill_input_ids": None,
                    "prefill_positions": req_positions,
                    "prefill_cum_sum": np.array(
                        [req_n_tokens], dtype=prefill_cum_sum.dtype
                    ),
                    "batch_indices": req_block_ids,
                    "prefill_is_partial": prefill_is_partial[req_i : req_i + 1],
                    "last_chunk_logits": req_logits,
                    "req_id": req_id,
                }
            )
            chunk_idx += 1
            tok_start = int(tok_end)

        self._prefill_pending = pending

    def update_prefill_kv(self) -> None:
        """Drain pending DLM prefill work items; no-op when nothing pending."""
        for ctx in self._prefill_pending:
            self.prefill_step(**ctx)
        self._prefill_pending = []

    def prefill_step(
        self,
        target_hidden: np.ndarray,
        prefill_input_ids: np.ndarray,
        prefill_positions: np.ndarray,
        prefill_cum_sum: np.ndarray,
        batch_indices: np.ndarray,
        prefill_is_partial: np.ndarray,
        last_chunk_logits: np.ndarray | None,
        req_id: str,
    ) -> None:
        """Advance DLM KV cache for one TLM prefill chunk."""
        assert prefill_cum_sum.shape[0] == 1, (
            "DFlash prefill_step expects a single request per step; "
            f"got prefill_cum_sum={prefill_cum_sum}."
        )
        n_tokens = int(prefill_cum_sum[0])
        assert n_tokens <= self.tlm_prefill_seq_len, (
            f"DFlash chunk size {n_tokens} exceeds TLM prefill_seq_len "
            f"{self.tlm_prefill_seq_len}."
        )

        st = self._state_for(req_id)
        is_final_chunk = not bool(prefill_is_partial[0])
        block_size = self.block_size
        active_slot = 0
        active_kv_slot = int(batch_indices[0])

        positions_padded = np.full(
            (self.tlm_prefill_seq_len,),
            -1,
            dtype=np.int64,
        )
        positions_padded[:n_tokens] = prefill_positions[:n_tokens]

        def _run_dlm_one_subblock(
            target_hidden_slice: np.ndarray,
            position_ids_target_row: np.ndarray,
            position_ids_row: np.ndarray,
            input_ids_row: np.ndarray,
        ) -> np.ndarray:
            self._reset_dlm_inputs()
            self._dlm_input_ids[active_slot] = input_ids_row
            # Inactive slots use position_ids=-1 so the hardware skips their KV writes.
            self._dlm_position_ids[active_slot] = position_ids_row
            self._dlm_position_ids_target[active_slot] = position_ids_target_row
            self._dlm_target_hidden[active_slot] = target_hidden_slice
            self._dlm_batch_index[active_slot, 0] = active_kv_slot

            dlm_inputs = {
                "input_ids": self._dlm_input_ids,
                "position_ids": self._dlm_position_ids,
                "position_ids_target": self._dlm_position_ids_target,
                "target_hidden": self._dlm_target_hidden,
                "batch_index": self._dlm_batch_index,
                "logits": self._dlm_logits_buf,
            }
            exec_obj_idx = self.model.session.np_run(dlm_inputs, is_prefill=True)
            self.model.session.complete_inf(exec_obj_idx, is_prefill=True)
            return self._dlm_logits_buf[active_slot]

        if not is_final_chunk:
            n_sub = -(-n_tokens // block_size)  # ceil
            for sub_i in range(n_sub):
                sub_start = sub_i * block_size
                sub_end = sub_start + block_size
                _run_dlm_one_subblock(
                    np.ascontiguousarray(target_hidden[0, sub_start:sub_end, :]),
                    positions_padded[sub_start:sub_end],
                    positions_padded[sub_start:sub_end] + block_size,
                    np.full((block_size,), self.mask_token_id, dtype=np.int64),
                )
            st.position_counter = int(prefill_positions[n_tokens - 1])
            return

        # Final chunk: last_sub holds the last real token and receives the bonus.
        last_real_idx = n_tokens - 1
        last_sub = last_real_idx // block_size
        last_pos = int(prefill_positions[last_real_idx])
        assert last_sub < self.num_sub_blocks, (
            f"DFlash divisibility-only path: bonus sub-block index "
            f"{last_sub} out of range (num_sub_blocks={self.num_sub_blocks})."
        )

        for sub_i in range(last_sub):
            sub_start = sub_i * block_size
            sub_end = sub_start + block_size
            _run_dlm_one_subblock(
                np.ascontiguousarray(target_hidden[0, sub_start:sub_end, :]),
                positions_padded[sub_start:sub_end],
                positions_padded[sub_start:sub_end] + block_size,
                np.full((block_size,), self.mask_token_id, dtype=np.int64),
            )

        sub_start = last_sub * block_size
        sub_end = sub_start + block_size
        target_hidden_slice = np.ascontiguousarray(target_hidden[0, sub_start:sub_end, :])
        position_ids_target_row = positions_padded[sub_start:sub_end]
        # DLM writes its KV at the block_size positions after the last real token.
        position_ids_row = np.arange(
            last_pos + 1,
            last_pos + 1 + block_size,
            dtype=np.int64,
        )
        input_ids_row = np.full((block_size,), self.mask_token_id, dtype=np.int64)
        if last_chunk_logits is not None:
            new_tlm_token = int(np.argmax(last_chunk_logits[0]))
            input_ids_row[0] = new_tlm_token

        active_logits = _run_dlm_one_subblock(
            target_hidden_slice,
            position_ids_target_row,
            position_ids_row,
            input_ids_row,
        )

        st.position_counter = last_pos
        # Slot 0 is the bonus seed; slots 1..block-1 are the drafts.
        st.dlm_candidates = active_logits.argmax(axis=-1).astype(np.int64)
        st.candidates_from_prefill = True

    def propose(
        self,
        input_batch: "InputBatch",
        sampled_token_ids: list[list[int]],
        batch_indices: np.ndarray,
        target_hidden: np.ndarray,
        commit: bool = True,
    ) -> list[list[int]]:
        """Batched DLM forward; commit=False still advances KV but discards drafts."""
        num_reqs = input_batch.num_reqs
        req_ids = input_batch.req_ids[:num_reqs]
        in_decode_phase = (
            input_batch.num_tokens_no_spec[:num_reqs]
            > input_batch.num_prompt_tokens[:num_reqs]
        )

        self._gc_finished_reqs(set(req_ids[:num_reqs]))

        if num_reqs == 0:
            return []

        block_size = self.block_size
        self._reset_dlm_inputs()
        input_ids = self._dlm_input_ids
        position_ids = self._dlm_position_ids
        position_ids_target = self._dlm_position_ids_target
        target_hidden_block = self._dlm_target_hidden
        batch_index = self._dlm_batch_index

        draft_token_ids: list[list[int]] = [[] for _ in range(num_reqs)]
        active_slots: list[int] = []

        for i in range(num_reqs):
            if not in_decode_phase[i]:
                continue
            req_id = req_ids[i]
            st = self._state_for(req_id)
            accepted_seq = sampled_token_ids[i]
            if not accepted_seq:
                continue

            if st.candidates_from_prefill:
                if commit:
                    # Gate open: serve prefill-computed drafts once, then decode.
                    st.candidates_from_prefill = False
                    assert st.dlm_candidates is not None
                    draft_token_ids[i] = st.dlm_candidates[1:].tolist()
                    continue
                # Gate closed: can't serve cached drafts; discard them and fall
                # through to the discarded-decode path so the DLM KV (and
                # position_counter) keeps tracking the TLM instead of freezing.
                st.candidates_from_prefill = False
                st.dlm_candidates = None

            n_advanced = len(accepted_seq)
            new_tlm_token = int(accepted_seq[-1])

            accepted_length = n_advanced - 1
            prev_counter = st.position_counter
            row_pids_target = np.arange(
                prev_counter + 1,
                prev_counter + 1 + block_size,
                dtype=np.int64,
            )
            row_pids_target[accepted_length + 1 :] = -1
            position_ids_target[i] = row_pids_target

            st.position_counter = prev_counter + n_advanced
            position_ids[i] = np.arange(
                st.position_counter + 1,
                st.position_counter + 1 + block_size,
                dtype=np.int64,
            )

            input_ids[i, 0] = new_tlm_token
            target_hidden_block[i] = target_hidden[i, :block_size, :]
            batch_index[i, 0] = int(batch_indices[i])
            active_slots.append(i)

        if active_slots:
            dlm_inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "position_ids_target": position_ids_target,
                "target_hidden": target_hidden_block,
                "batch_index": batch_index,
                "logits": self._dlm_logits_buf,
            }
            # DLM is compiled prefill_only=True, so every call is a prefill.
            exec_obj_idx = self.model.session.np_run(dlm_inputs, is_prefill=True)
            self.model.session.complete_inf(exec_obj_idx, is_prefill=True)

            if commit:
                all_candidates = self._dlm_logits_buf.argmax(axis=-1).astype(np.int64)
                for slot in active_slots:
                    st = self._req_state[req_ids[slot]]
                    st.dlm_candidates = all_candidates[slot]
                    draft_token_ids[slot] = all_candidates[slot, 1:].tolist()

        return draft_token_ids
