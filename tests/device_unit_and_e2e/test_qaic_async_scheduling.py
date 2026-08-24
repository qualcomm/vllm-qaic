# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
import random
import time

import pytest
from vllm import SamplingParams

from .conftest import check_outputs_equal

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# KPI helpers
# ---------------------------------------------------------------------------


def _extract_metrics(req_outputs):
    """Extract per-request KPIs from RequestOutput.metrics (RequestStateStats)."""
    ttfts, decode_times, e2e_latencies, output_tokens = [], [], [], []
    for req in req_outputs:
        m = req.metrics
        if m is None:
            continue
        ttfts.append(m.first_token_ts - m.scheduled_ts)
        if m.first_token_ts > 0 and m.last_token_ts > 0:
            decode_times.append(m.last_token_ts - m.first_token_ts)
            e2e_latencies.append(m.last_token_ts - m.scheduled_ts)
        output_tokens.append(m.num_generation_tokens)
    return ttfts, decode_times, e2e_latencies, output_tokens


def _print_kpi_comparison(
    iter_idx, sync_reqs, async_reqs, sync_total_time, async_total_time
):
    """Print a side-by-side KPI table for sync vs async scheduling."""
    s_ttft, s_dec, s_e2e, s_toks = _extract_metrics(sync_reqs)
    a_ttft, a_dec, a_e2e, a_toks = _extract_metrics(async_reqs)

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    def tpot(dec_times, toks):
        vals = [d / (t - 1) for d, t in zip(dec_times, toks, strict=False) if t > 1]
        return avg(vals) * 1000  # ms

    def throughput(e2e_list, toks_list):
        total_tokens = sum(toks_list)
        total_time = max(e2e_list) if e2e_list else float("nan")
        return total_tokens / total_time if total_time else float("nan")

    def decode_throughput(dec_list, toks_list):
        total_tokens = sum(toks_list)
        total_time = max(dec_list) if dec_list else float("nan")
        return total_tokens / total_time if total_time else float("nan")

    print(f"\n{'=' * 62}")
    print(f"  Iteration {iter_idx + 1} — Async Scheduling KPI Comparison")
    print(f"{'=' * 62}")
    print(f"  {'Metric':<28} {'non-async':>14} {'async':>14}")
    print(f"  {'-' * 56}")
    print(
        f"  {'Avg TTFT (ms)':<28} {avg(s_ttft) * 1000:>13.1f}  "
        f"{avg(a_ttft) * 1000:>13.1f}"
    )
    print(
        f"  {'Avg Decode Time (ms)':<28} {avg(s_dec) * 1000:>13.1f}  "
        f"{avg(a_dec) * 1000:>13.1f}"
    )
    print(
        f"  {'Avg E2E Latency (ms)':<28} {avg(s_e2e) * 1000:>13.1f}  "
        f"{avg(a_e2e) * 1000:>13.1f}"
    )
    print(
        f"  {'Avg TPOT (ms/tok)':<28} {tpot(s_dec, s_toks):>13.2f}  "
        f"{tpot(a_dec, a_toks):>13.2f}"
    )
    print(
        f"  {'Throughput (tok/s)':<28} {throughput(s_e2e, s_toks):>13.1f}  "
        f"{throughput(a_e2e, a_toks):>13.1f}"
    )
    print(
        f"  {'Decode Throughput (tok/s)':<28} "
        f"{decode_throughput(s_dec, s_toks):>13.1f}  "
        f"{decode_throughput(a_dec, a_toks):>13.1f}"
    )
    print(f"  {'Avg Output Tokens':<28} {avg(s_toks):>13.1f}  {avg(a_toks):>13.1f}")
    print(f"  {'-' * 56}")
    print(
        f"  {'Total Exec Time (s)':<28} {sync_total_time:>13.2f}  "
        f"{async_total_time:>13.2f}"
    )
    print(f"{'=' * 62}\n")


def _run_consistency_check(
    prompts: list[str],
    runner,
    runner_async,
    sp: SamplingParams,
    num_iterations: int = 3,
) -> None:
    for i in range(num_iterations):
        random.shuffle(prompts)

        t0 = time.time()
        outputs_raw = runner.llm.generate(prompts, sampling_params=sp)
        sync_total_time = time.time() - t0

        t0 = time.time()
        outputs_async_raw = runner_async.llm.generate(prompts, sampling_params=sp)
        async_total_time = time.time() - t0

        outputs_cmp = [
            (list(s.token_ids), s.text) for req in outputs_raw for s in req.outputs[:1]
        ]
        outputs_async_cmp = [
            (list(s.token_ids), s.text)
            for req in outputs_async_raw
            for s in req.outputs[:1]
        ]

        check_outputs_equal(outputs_cmp, outputs_async_cmp, "non_async", "async")

        _print_kpi_comparison(
            i, outputs_raw, outputs_async_raw, sync_total_time, async_total_time
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_runner_chunked(make_runner, async_scheduling: bool, dg: list, seq_len: int):
    if async_scheduling:
        return make_runner(
            async_scheduling,
            dg,
            enable_chunked_prefill=True,
            max_num_batched_tokens=seq_len,
        )
    return make_runner(async_scheduling, dg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=1024,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
)
def test_output_consistency_database(
    make_runner, sharegpt_prompts, ctx_len, device_groups, sampling_params
):
    prompts = sharegpt_prompts(20, in_len=ctx_len)
    with (
        make_runner(False, device_groups[0]) as runner,
        make_runner(True, device_groups[1]) as runner_async,
    ):
        _run_consistency_check(prompts, runner, runner_async, sampling_params)


@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=1024,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
)
def test_output_consistency_database_chunked_prefill(
    make_runner,
    sharegpt_prompts,
    ctx_len,
    decode_bsz,
    seq_len,
    device_groups,
    sampling_params,
):
    prompts = sharegpt_prompts(3 * decode_bsz, in_len=ctx_len)
    with (
        _make_runner_chunked(make_runner, False, device_groups[0], seq_len) as runner,
        _make_runner_chunked(
            make_runner, True, device_groups[1], seq_len
        ) as runner_async,
    ):
        _run_consistency_check(prompts, runner, runner_async, sampling_params)
