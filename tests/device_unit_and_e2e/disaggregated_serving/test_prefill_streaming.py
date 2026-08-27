# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Benchmark test comparing prefill latency metrics with and without stream_prefill.

Both runners act as kv_producer (prefill-only) connected to a shared KV handoff
server.  They run simultaneously on separate device groups (4 devices each, 8
total).  3 iterations are collected per runner and mean/median/p99 are reported.

Run example:
    pytest tests/e2e/disaggregated_serving/test_prefill_streaming.py -v \
        --device-id 0,1,2,3,4,5,6,7 \
        --override-qaic-config "qpc_path:/path/to/qpc stages:4"
"""

import random

import numpy as np
import pytest

from vllm import SamplingParams
from vllm.config import KVTransferConfig

NUM_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _extract_metrics(req_outputs):
    ttfts, ttfts_from_arrival, decode_times, e2e_latencies, output_tokens = (
        [],
        [],
        [],
        [],
        [],
    )
    for req in req_outputs:
        m = req.metrics
        if m is None:
            continue
        ttfts.append(m.first_token_ts - m.scheduled_ts)
        if m.first_token_latency > 0:
            ttfts_from_arrival.append(m.first_token_latency)
        if m.first_token_ts > 0 and m.last_token_ts > 0:
            decode_times.append(m.last_token_ts - m.first_token_ts)
            e2e_latencies.append(m.last_token_ts - m.scheduled_ts)
        output_tokens.append(m.num_generation_tokens)
    return ttfts, ttfts_from_arrival, decode_times, e2e_latencies, output_tokens


def _compute_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p99": float("nan")}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
    }


def _throughput(e2e_list, toks_list):
    if not e2e_list:
        return float("nan")
    return sum(toks_list) / max(e2e_list)


def _print_metrics_table(
    label_a: str,
    reqs_a,
    label_b: str,
    reqs_b,
    tput_override: tuple[float, float] | None = None,
):
    ttft_a, ttft_arrival_a, _, e2e_a, toks_a = _extract_metrics(reqs_a)
    ttft_b, ttft_arrival_b, _, e2e_b, toks_b = _extract_metrics(reqs_b)

    sa_ttft = _compute_stats([v * 1000 for v in ttft_a])
    sb_ttft = _compute_stats([v * 1000 for v in ttft_b])
    sa_ttft_arrival = _compute_stats([v * 1000 for v in ttft_arrival_a])
    sb_ttft_arrival = _compute_stats([v * 1000 for v in ttft_arrival_b])
    sa_e2e = _compute_stats([v * 1000 for v in e2e_a])
    sb_e2e = _compute_stats([v * 1000 for v in e2e_b])
    if tput_override is not None:
        tput_a, tput_b = tput_override
    else:
        tput_a = _throughput(e2e_a, toks_a)
        tput_b = _throughput(e2e_b, toks_b)

    W = 71
    col = 18
    print(f"\n{'=' * W}")
    print(f"  Prefill Streaming Benchmark — {label_a} vs {label_b}")
    print(f"{'=' * W}")
    print(f"  {'Metric':<30} {label_a:>{col}} {label_b:>{col}}")
    print(f"  {'-' * (W - 2)}")

    def row(name, va, vb, unit="ms"):
        print(f"  {name:<30} {va:>{col}.2f} {vb:>{col}.2f}  ({unit})")

    row("TTFT mean", sa_ttft["mean"], sb_ttft["mean"])
    row("TTFT median", sa_ttft["median"], sb_ttft["median"])
    row("TTFT p99", sa_ttft["p99"], sb_ttft["p99"])
    row("TTFT (from arrival) mean", sa_ttft_arrival["mean"], sb_ttft_arrival["mean"])
    row(
        "TTFT (from arrival) median",
        sa_ttft_arrival["median"],
        sb_ttft_arrival["median"],
    )
    row("TTFT (from arrival) p99", sa_ttft_arrival["p99"], sb_ttft_arrival["p99"])
    row("E2E mean", sa_e2e["mean"], sb_e2e["mean"])
    row("E2E median", sa_e2e["median"], sb_e2e["median"])
    row("E2E p99", sa_e2e["p99"], sb_e2e["p99"])
    row("Throughput", tput_a, tput_b, unit="tok/s")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=2048,
    seq_len=256,
    dtype="mxfp6",
    decode_bsz=2,
    num_device_groups=2,
    device_group_size=2,
)
def test_prefill_streaming_metrics(
    ctx_len,
    seq_len,
    device_groups,
    make_runner,
    override_qaic_config,
    sharegpt_prompts,
    num_prompt,
    kv_port,
    kv_handoff_server,
):
    prompts = sharegpt_prompts(num_prompt, in_len=ctx_len)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)

    def _make(disable_stream_prefill: bool, dg: list):
        override = dict(override_qaic_config or {})
        override.setdefault("prefill_only", True)
        override.setdefault("stages", len(dg))
        override.setdefault("prefill_seq_len", seq_len)

        ktc = KVTransferConfig(
            kv_connector="QaicConnector",
            kv_role="kv_producer",
            kv_rank=0,
            kv_port=kv_port,
        )
        extra_kwargs = {}
        if disable_stream_prefill:
            long_prefill_token_threshold = seq_len
        else:
            long_prefill_token_threshold = 0
            extra_kwargs["max_num_batched_tokens"] = seq_len

        return make_runner(
            async_scheduling=not disable_stream_prefill,
            dg=dg,
            enable_chunked_prefill=True,
            override_qaic_config=override,
            long_prefill_token_threshold=long_prefill_token_threshold,
            kv_transfer_config=ktc,
            **extra_kwargs,
        )

    # Shared handoff server sized to absorb all KV blocks from both runners
    handoff_size = 2 * NUM_ITERATIONS * len(prompts)
    kv_handoff_server(kv_port, handoff_size)

    base_outputs: list = []
    stream_outputs: list = []
    iter_tput_base: list[float] = []
    iter_tput_stream: list[float] = []
    with (
        _make(True, device_groups[0]) as runner_base,
        _make(False, device_groups[1]) as runner_stream,
    ):
        for i in range(NUM_ITERATIONS):
            random.shuffle(prompts)
            base_iter_outputs = runner_base.llm.generate(
                prompts, sampling_params=sampling_params
            )
            stream_iter_outputs = runner_stream.llm.generate(
                prompts, sampling_params=sampling_params
            )
            print(f"\n--- iteration {i + 1}/{NUM_ITERATIONS} done ---")
            _print_metrics_table(
                f"no-stream iter{i + 1}",
                base_iter_outputs,
                f"stream iter{i + 1}",
                stream_iter_outputs,
            )
            _, _, _, e2e_b, toks_b = _extract_metrics(base_iter_outputs)
            _, _, _, e2e_s, toks_s = _extract_metrics(stream_iter_outputs)
            iter_tput_base.append(_throughput(e2e_b, toks_b))
            iter_tput_stream.append(_throughput(e2e_s, toks_s))
            base_outputs.extend(base_iter_outputs)
            stream_outputs.extend(stream_iter_outputs)

    agg_tput_base = float(np.mean(iter_tput_base)) if iter_tput_base else float("nan")
    agg_tput_stream = (
        float(np.mean(iter_tput_stream)) if iter_tput_stream else float("nan")
    )
    print("\n--- Aggregate across all iterations ---")
    _print_metrics_table(
        "no-stream",
        base_outputs,
        "stream",
        stream_outputs,
        tput_override=(agg_tput_base, agg_tput_stream),
    )

    assert len(base_outputs) == NUM_ITERATIONS * len(prompts), (
        f"Expected {NUM_ITERATIONS * len(prompts)} baseline outputs, "
        f"got {len(base_outputs)}"
    )
    assert len(stream_outputs) == NUM_ITERATIONS * len(prompts), (
        f"Expected {NUM_ITERATIONS * len(prompts)} stream outputs, "
        f"got {len(stream_outputs)}"
    )
