# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import pytest

from vllm import SamplingParams

ITERATIONS = 6
MAX_END_TOKENS = 20
KV_CACHE_SZ = 160


def _run_multichat(
    make_runner,
    dg,
    sharegpt_prompts,
    decode_bsz,
    prefix_caching,
):
    sampling_params = SamplingParams(temperature=0, max_tokens=MAX_END_TOKENS)

    with make_runner(
        async_scheduling=False,
        dg=dg,
        enable_prefix_caching=prefix_caching,
        num_gpu_blocks_override=KV_CACHE_SZ if prefix_caching else None,
    ) as runner:
        prompts = None
        avg_ttfts = []
        for _ in range(ITERATIONS):
            new_turns = sharegpt_prompts(decode_bsz)
            prompts = (
                new_turns
                if prompts is None
                else [p + " " + t for p, t in zip(prompts, new_turns, strict=False)]
            )

            outputs = runner.llm.generate(prompts, sampling_params)
            iter_ttfts = [
                op.metrics.first_token_ts - op.metrics.scheduled_ts for op in outputs
            ]
            avg_ttfts.append(sum(iter_ttfts) / len(iter_ttfts))

            prompts = [
                p + op.outputs[0].text for p, op in zip(prompts, outputs, strict=False)
            ]

    return sum(avg_ttfts) / len(avg_ttfts)


@pytest.mark.qaic_test_config(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    seq_len=32,
    ctx_len=1024,
    decode_bsz=16,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_multichat_correctness(
    make_runner, device_group, decode_bsz, seq_len, ctx_len, sharegpt_prompts
):
    """Multi-turn chat (each turn appends a new user message to the
    conversation so far) run with and without prefix caching. Prefix
    caching must not increase average TTFT."""
    assert decode_bsz <= KV_CACHE_SZ, "kv cache size should >= decode_bsz"
    assert (seq_len + MAX_END_TOKENS) * ITERATIONS <= ctx_len

    avg_ttft_with_caching = _run_multichat(
        make_runner, device_group, sharegpt_prompts, decode_bsz, prefix_caching=True
    )
    avg_ttft_without_caching = _run_multichat(
        make_runner, device_group, sharegpt_prompts, decode_bsz, prefix_caching=False
    )

    print(
        f"Prefix Caching E2E speed-up: "
        f"{avg_ttft_without_caching / avg_ttft_with_caching:.3f}x"
    )
    assert avg_ttft_with_caching <= avg_ttft_without_caching, (
        "No prefix caching benefit observed in TTFT"
    )
