import torch

from vllm_qaic.attention.backends.qaic_attn import (
    QAicAttentionBackendImpl,
    QAicAttentionMetadata,
)


def _metadata(
    request_ids: list[str], seq_lens: list[int], query_lens: list[int]
) -> QAicAttentionMetadata:
    return QAicAttentionMetadata(
        isa="vec",
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens),
        query_start_loc=torch.tensor([0, *torch.tensor(query_lens).cumsum(0)]),
        max_seq_len=max(seq_lens),
        seq_lens=torch.tensor(seq_lens),
        block_table=torch.empty(0),
        slot_mapping=torch.empty(0),
        max_num_seqs=2,
        max_model_len=16,
        scheduler_metadata=None,
        req_ids=request_ids,
    )


def _run_decode(
    attention: QAicAttentionBackendImpl,
    request_ids: list[str],
    seq_lens: list[int],
    query_lens: list[int],
    key_values: list[float],
) -> None:
    num_tokens = sum(query_lens)
    query = torch.zeros(num_tokens, 1, 2)
    key = torch.tensor(key_values, dtype=torch.float32).view(num_tokens, 1, 1)
    key = key.repeat(1, 1, 2)
    value = key.clone()
    output = torch.empty_like(query)
    attention._run_sdpa_decode_forward(
        query,
        key,
        value,
        _metadata(request_ids, seq_lens, query_lens),
        output,
    )


def test_speculative_rejection_overwrites_and_truncates_request_kv_cache():
    attention = QAicAttentionBackendImpl(
        num_heads=1,
        head_size=2,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    _run_decode(
        attention,
        ["request-a", "request-b"],
        [3, 3],
        [3, 3],
        [1, 2, 3, 10, 11, 12],
    )
    _run_decode(
        attention,
        ["request-a", "request-b"],
        [6, 6],
        [3, 3],
        [4, 5, 6, 13, 14, 15],
    )
    _run_decode(attention, ["request-b", "request-a"], [4, 4], [1, 1], [20, 30])

    a_key, _, a_cached = attention._kv_cache["request-a"]
    b_key, _, b_cached = attention._kv_cache["request-b"]
    assert a_cached == b_cached == 4
    assert a_key[:a_cached, 0, 0].tolist() == [1, 2, 3, 30]
    assert b_key[:b_cached, 0, 0].tolist() == [10, 11, 12, 20]
