# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""qaic::paged_attention with attention sinks and a sliding window (gpt-oss),
on vLLM 0.30.0's native LBNHC cache, against an fp32 reference."""

import pytest
import torch

from tests.unit.ops.moe_test_utils import requires_qaic
from tests.unit.ops.paged_attention_test_utils import lbnhc_kv_cache, ref_attention

H, HKV, D, B = 8, 2, 64, 16
NUM_BLOCKS = 64


def _lbnhc_cache(device):
    return lbnhc_kv_cache(NUM_BLOCKS, B, HKV, D, device).split(D, dim=-1)


def _input_views(q, k, v, layout, device):
    """Upload first, then create views so setup copies stay outside profiling."""
    tensors = (q, k, v)
    if layout == "contiguous":
        return [x.to(device) for x in tensors]
    if layout == "qkv_split":
        # A leading head also exercises nonzero Q, K and V storage offsets.
        projection = torch.cat([q[:, 0], *(x.flatten(1) for x in tensors)], 1)
        parts = projection.to(device).split([D, H * D, HKV * D, HKV * D], 1)
        return [x.view(q.shape[0], -1, D) for x in parts[1:]]
    if layout == "head_transpose":
        return [
            x.transpose(0, 1).contiguous().to(device).transpose(0, 1) for x in tensors
        ]
    if layout == "independent":
        # Different token/head strides for every input, including a packed K.
        return [
            _input_views(q, k, v, "head_transpose", device)[0],
            k.to(device),
            _input_views(q, k, v, "qkv_split", device)[2],
        ]
    if layout == "dim_slice":
        return [
            torch.stack([x, x], -1).flatten(-2).to(device)[..., ::2] for x in tensors
        ]
    if layout == "wide_token_stride":
        views = []
        for x in tensors:
            padded = torch.zeros(x.shape[0], 33024, dtype=x.dtype)
            padded[:, : x.shape[1] * D] = x.flatten(1)
            views.append(padded.to(device)[:, : x.shape[1] * D].view(x.shape))
        return views
    raise ValueError(layout)


def _make_inputs(
    q_lens,
    seq_lens,
    sinks,
    window_left,
    input_layout="contiguous",
    write_cache=True,
):
    dev = "qaic:0"
    max_blocks = max(-(-s // B) for s in seq_lens)
    block_table = torch.randperm(NUM_BLOCKS)[: len(seq_lens) * max_blocks]
    block_table = block_table.view(len(seq_lens), max_blocks).to(torch.int32)

    key_cache, value_cache = _lbnhc_cache(dev)
    k_full = [torch.randn(s, HKV, D, dtype=torch.float16) for s in seq_lens]
    v_full = [torch.randn(s, HKV, D, dtype=torch.float16) for s in seq_lens]

    # Write the cached prefix of every request with the store path, then let
    # the attention call write the new tokens itself.
    def slots(req, positions):
        return [int(block_table[req, p // B]) * B + p % B for p in positions]

    prefix_slots, prefix_k, prefix_v = [], [], []
    new_slots, new_k, new_v = [], [], []
    for req, (q_len, s) in enumerate(zip(q_lens, seq_lens, strict=True)):
        cached_len = s - q_len if write_cache else s
        prefix_slots += slots(req, range(cached_len))
        prefix_k.append(k_full[req][:cached_len])
        prefix_v.append(v_full[req][:cached_len])
        new_slots += slots(req, range(s - q_len, s))
        new_k.append(k_full[req][s - q_len :])
        new_v.append(v_full[req][s - q_len :])
    kc_cpu = key_cache.cpu()
    vc_cpu = value_cache.cpu()
    for slot, k, v in zip(
        prefix_slots, torch.cat(prefix_k), torch.cat(prefix_v), strict=True
    ):
        kc_cpu[slot // B, :, slot % B] = k
        vc_cpu[slot // B, :, slot % B] = v
    key_cache.copy_(kc_cpu)
    value_cache.copy_(vc_cpu)

    num_tokens = sum(q_lens)
    q = torch.randn(num_tokens, H, D, dtype=torch.float16)
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32)
    scale = D**-0.5
    inputs = (
        *_input_views(q, torch.cat(new_k), torch.cat(new_v), input_layout, dev),
        key_cache,
        value_cache,
        block_table.to(dev),
        torch.tensor(new_slots, dtype=torch.int32).to(dev),
        qsl.to(dev),
        torch.tensor(seq_lens, dtype=torch.int32).to(dev),
        scale,
        True,
        write_cache,
        None if sinks is None else sinks.to(dev),
        window_left,
    )
    expected = ref_attention(
        q, k_full, v_full, q_lens, seq_lens, scale, sinks, window_left
    )
    return inputs, expected, k_full, v_full, block_table


def _run(q_lens, seq_lens, backend, sinks, window_left, monkeypatch):
    from vllm_qaic import _custom_ops as qaic_ops

    monkeypatch.setenv("QAIC_PAGED_ATTN_BACKEND", backend)
    inputs, expected, k_full, v_full, block_table = _make_inputs(
        q_lens, seq_lens, sinks, window_left
    )
    out = qaic_ops.paged_attention(*inputs).cpu()
    key_cache, value_cache = inputs[3:5]
    return out, expected, key_cache, value_cache, k_full, v_full, block_table


# gpt-oss stores sinks ~N(0, 1)-ish; the large case makes the sink dominate
# every row, which exercises the rebased denominator.
_SINKS = {
    "none": None,
    "normal": lambda: torch.randn(H),
    "dominant": lambda: torch.full((H,), 12.0),
}


@requires_qaic
@pytest.mark.parametrize("backend", ["hmx", "hvx"])
@pytest.mark.parametrize("sinks", list(_SINKS))
@pytest.mark.parametrize("window_left", [-1, 0, 19, 127])
@pytest.mark.parametrize(
    "q_lens, seq_lens",
    [
        ([1, 1, 1], [5, 50, 200]),  # decode
        ([37, 70], [37, 70]),  # prefill
        ([1, 20, 9], [50, 70, 9]),  # mixed decode + chunked prefill
    ],
    ids=["decode", "prefill", "mixed"],
)
def test_paged_attention_sinks_window(
    monkeypatch, backend, sinks, window_left, q_lens, seq_lens
):
    torch.manual_seed(0)
    sinks_t = None if _SINKS[sinks] is None else _SINKS[sinks]()
    out, expected, *_ = _run(
        q_lens, seq_lens, backend, sinks_t, window_left, monkeypatch
    )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)


@requires_qaic
def test_paged_attention_writes_lbnhc_cache_in_place(monkeypatch):
    """New tokens land in the interleaved K/V halves of the right page rows."""
    torch.manual_seed(0)
    q_lens, seq_lens = [3, 1], [3, 20]
    _, _, key_cache, value_cache, k_full, v_full, block_table = _run(
        q_lens, seq_lens, "hmx", torch.randn(H), 19, monkeypatch
    )
    kc, vc = key_cache.cpu(), value_cache.cpu()
    for req, s in enumerate(seq_lens):
        for p in range(s):
            blk = int(block_table[req, p // B])
            torch.testing.assert_close(
                kc[blk, :, p % B], k_full[req][p], atol=0, rtol=0
            )
            torch.testing.assert_close(
                vc[blk, :, p % B], v_full[req][p], atol=0, rtol=0
            )


@requires_qaic
@pytest.mark.parametrize("layout", ["qkv_split", "head_transpose", "independent"])
@pytest.mark.parametrize("write_cache", [True, False])
@pytest.mark.parametrize("with_sinks_window", [True, False])
@pytest.mark.parametrize(
    "q_lens, seq_lens",
    [
        ([1, 1, 1], [5, 50, 200]),
        ([37, 70], [37, 70]),
        ([1, 20, 9], [50, 70, 9]),
    ],
    ids=["decode", "prefill", "mixed"],
)
def test_paged_attention_input_views(
    monkeypatch, layout, write_cache, with_sinks_window, q_lens, seq_lens
):
    """HMX reads views without packing; cache writes respect both layouts."""
    from vllm_qaic import _custom_ops as qaic_ops

    monkeypatch.setenv("QAIC_PAGED_ATTN_BACKEND", "hmx")
    torch.manual_seed(0)
    sinks = torch.randn(H) if with_sinks_window else None
    inputs, expected, k_full, v_full, block_table = _make_inputs(
        q_lens,
        seq_lens,
        sinks,
        19 if with_sinks_window else -1,
        layout,
        write_cache,
    )
    # Load the kernel before profiling only the device-resident operator.
    qaic_ops.paged_attention(*inputs)
    torch.qaic.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        out = qaic_ops.paged_attention(*inputs)
        torch.qaic.synchronize()
    copies = sum(e.count for e in prof.key_averages() if e.key == "aten::_copy_from")
    assert copies == 0
    assert out.is_contiguous()
    torch.testing.assert_close(out.cpu().float(), expected, atol=2e-2, rtol=2e-2)
    kc, vc = (x.cpu() for x in inputs[3:5])
    for req, s in enumerate(seq_lens):
        for p in range(s):
            blk = int(block_table[req, p // B])
            torch.testing.assert_close(
                kc[blk, :, p % B], k_full[req][p], atol=0, rtol=0
            )
            torch.testing.assert_close(
                vc[blk, :, p % B], v_full[req][p], atol=0, rtol=0
            )


@requires_qaic
@pytest.mark.parametrize(
    "backend, layout, expected_copies",
    [
        ("hmx", "dim_slice", 3),
        ("hmx", "wide_token_stride", 2),
        ("hvx", "qkv_split", 3),
    ],
)
def test_paged_attention_input_packing_fallback(
    monkeypatch, backend, layout, expected_copies
):
    from vllm_qaic import _custom_ops as qaic_ops

    monkeypatch.setenv("QAIC_PAGED_ATTN_BACKEND", backend)
    torch.manual_seed(0)
    inputs, expected, *_ = _make_inputs([3, 1], [3, 20], None, -1, layout)
    qaic_ops.paged_attention(*inputs)
    torch.qaic.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        out = qaic_ops.paged_attention(*inputs)
        torch.qaic.synchronize()
    copies = sum(e.count for e in prof.key_averages() if e.key == "aten::_copy_from")
    assert copies == expected_copies
    torch.testing.assert_close(out.cpu().float(), expected, atol=2e-2, rtol=2e-2)
