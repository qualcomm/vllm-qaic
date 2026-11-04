# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_qaic.worker import model_runner as runner_module
from vllm_qaic.worker.model_runner import QaicModelRunnerAoT


@pytest.mark.parametrize(
    "uses_mrope,with_prefill", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("signal_count", [1, 7, 16])
@pytest.mark.parametrize("method", ["dflash", "draft_model", "ngram"])
def test_decode_padding_ignores_stale_positive_token_ids(
    monkeypatch, uses_mrope, with_prefill, signal_count, method
):
    block_size, ctx_len = 16, 4096
    num_decodes = 1 if uses_mrope else 2
    req_ids = [f"decode_{i}" for i in range(num_decodes)]
    counts = [signal_count] + ([3] if num_decodes == 2 else [])
    computed = [ctx_len - signal_count] + ([20] if num_decodes == 2 else [])
    if with_prefill:
        req_ids.append("prefill")
        counts.append(5)
        computed.append(0)
    num_reqs = len(req_ids)
    calls = []

    class RecordingModel(SimpleNamespace):
        def __call__(self, **inputs):
            calls.append(
                {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in inputs.items()
                }
            )

    # Include stale positive values even in the host table's extra K columns.
    token_table = torch.full(
        (num_reqs, ctx_len + block_size - 1), 42, dtype=torch.int64
    )
    if num_decodes == 2:
        token_table[1, computed[1] + 1] = -1
    original_table = token_table.clone()
    computed_tensor = torch.tensor(computed, dtype=torch.int64)
    mrope = torch.full((3, 128), 7777, dtype=torch.int64)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        num_computed_tokens_cpu=computed_tensor.numpy(),
        num_computed_tokens_cpu_tensor=computed_tensor,
        token_ids_cpu_tensor=token_table,
        token_ids_cpu=token_table.numpy(),
        block_table=[
            SimpleNamespace(
                get_numpy_array=lambda: np.arange(1, num_reqs + 1).reshape(-1, 1)
            )
        ],
        prev_sampled_token_ids=None,
        num_prompt_tokens=np.full(num_reqs, 10),
    )
    spec = SimpleNamespace(method=method, use_dflash=lambda: method == "dflash")
    runner = SimpleNamespace(
        input_batch=batch,
        num_decodes=num_decodes,
        max_decode_tokens=block_size,
        active_k=15,
        decode_ks=[15],
        signal_num_scheduled_tokens=np.zeros(num_reqs, dtype=np.int32),
        arange_np=np.arange(128),
        query_pos=SimpleNamespace(np=np.zeros(128, dtype=np.int64)),
        positions_np=np.zeros(128, dtype=np.int64),
        uses_mrope=uses_mrope,
        mrope_positions=SimpleNamespace(cpu=mrope, np=mrope.numpy()),
        mrope_positions_np=mrope.numpy(),
        input_ids=SimpleNamespace(cpu=torch.empty(128, dtype=torch.int64)),
        optimistic_seq_lens_cpu=torch.zeros(num_reqs, dtype=torch.int64),
        discard_request_mask=SimpleNamespace(np=np.zeros(num_reqs, dtype=bool)),
        requests={
            r: SimpleNamespace(
                num_tokens=c + n,
                prompt_token_ids=list(range(10)),
                prompt_embeds=None,
                mrope_positions=torch.zeros((3, 10), dtype=torch.int64),
                mrope_position_delta=0,
            )
            for r, c, n in zip(req_ids, computed, counts, strict=True)
        },
        synchronize_input_prep=nullcontext,
        execute_model_state=None,
        _update_states=lambda *args: None,
        _determine_active_k=lambda *args: 15,
        _calc_spec_decode_metadata=lambda *args: None,
        is_kv_consumer=False,
        is_kv_producer=False,
        is_async_kv_producer=False,
        supports_mm_inputs=False,
        speculative_config=spec,
        use_async_scheduling=False,
        is_pooling_model=False,
        lora_config=None,
        vllm_config=None,
        kv_cache_info=None,
        kv_caches=None,
        maybe_get_kv_connector_output=lambda *args, **kwargs: nullcontext(None),
        create_logits_np=lambda n, vocab, width=1: np.zeros(
            (n, width, vocab), dtype=np.float32
        ),
        _compute_hidden_states_and_logits=lambda *args, **kwargs: (None, None),
        _tlm_prefill_hidden_chunks=[],
        _dflash_hidden_dtype=np.float32,
        drafter=SimpleNamespace(build_prefill_pending=lambda *args: None),
        model=RecordingModel(
            is_vision_encoder=False,
            vocab_size=32,
            decode_bsz=num_decodes,
            uses_mrope=uses_mrope,
            num_prefill_chunks=lambda cum: len(cum),
            tlm_hidden_size=8,
            prefill_seq_len=128,
        ),
    )
    for name in (
        "_prepare_qaic_inputs",
        "_get_cumsum_and_arange",
        "_calc_mrope_positions",
    ):
        setattr(runner, name, MethodType(getattr(QaicModelRunnerAoT, name), runner))
    monkeypatch.setattr(
        runner_module, "set_forward_context", lambda *a, **kw: nullcontext()
    )
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={}),
    )
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=sum(counts),
        num_scheduled_tokens=dict(zip(req_ids, counts, strict=True)),
        scheduled_spec_decode_tokens={},
    )
    QaicModelRunnerAoT.execute_model(runner, scheduler)

    decode = next(call for call in calls if not call["is_prompt"])
    positions = decode["positions"].reshape((-1, num_decodes, block_size))
    ids = decode["input_ids"].reshape(num_decodes, block_size)
    for row in range(num_decodes):
        expected = computed[row] + np.arange(block_size)
        if method == "dflash":
            expected[counts[row] :] = -1
        expected[ids[row] == -1] = -1
        np.testing.assert_array_equal(positions[0, row], expected)
        if method == "dflash":
            np.testing.assert_array_equal(
                positions[:, row], np.broadcast_to(expected, positions[:, row].shape)
            )
            assert np.all(positions[:, row, counts[row] :] == -1)
            assert np.all((positions[0, row] == -1) | (positions[0, row] < ctx_len))
    if with_prefill:
        prefill = next(call for call in calls if call["is_prompt"])
        np.testing.assert_array_equal(prefill["positions"], np.arange(5))
        np.testing.assert_array_equal(prefill["input_ids"], np.full(5, 42))
    torch.testing.assert_close(token_table, original_table)
