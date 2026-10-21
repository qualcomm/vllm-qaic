# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import pytest

from vllm import SamplingParams

ITERATIONS = 6
MAX_END_TOKENS = 20
KV_CACHE_SZ = 512
SEED_STRIDE = 42


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
        # Warmup the device without seeding the prefix cache with text reused by
        # the measured turns.
        runner.llm.generate(
            sharegpt_prompts(decode_bsz, seed=123456),
            sampling_params=sampling_params,
        )
        prompts = None
        total_prompt_tokens = 0
        total_cached_tokens = 0
        per_turn_hit_rates = []
        for i in range(ITERATIONS):
            new_turns = sharegpt_prompts(decode_bsz, seed=i * SEED_STRIDE)
            prompts = (
                new_turns
                if prompts is None
                else [p + " " + t for p, t in zip(prompts, new_turns, strict=False)]
            )

            outputs = runner.llm.generate(prompts, sampling_params=sampling_params)
            n_prompt = [len(op.prompt_token_ids) for op in outputs]
            n_cached = [(getattr(op, "num_cached_tokens", None) or 0) for op in outputs]
            turn_prompt_tokens = sum(n_prompt)
            turn_cached_tokens = sum(n_cached)
            turn_hit_rate = turn_cached_tokens / max(turn_prompt_tokens, 1)
            total_prompt_tokens += turn_prompt_tokens
            total_cached_tokens += turn_cached_tokens
            per_turn_hit_rates.append(turn_hit_rate)

            print(
                f"[multichat] caching={prefix_caching} turn={i} "
                f"cached_tokens(avg/max)={sum(n_cached) / len(n_cached):.0f}/"
                f"{max(n_cached)} "
                f"prefix_hit={turn_hit_rate:.2%}",
                flush=True,
            )

            prompts = [
                p + op.outputs[0].text for p, op in zip(prompts, outputs, strict=False)
            ]

    cache_hit_rate = total_cached_tokens / max(total_prompt_tokens, 1)
    print(
        f"[Multi-turn chat] caching={prefix_caching} totals over {ITERATIONS} turns: "
        f"prompt_tokens={total_prompt_tokens} "
        f"cached_tokens={total_cached_tokens} "
        f"cache_hit_rate={cache_hit_rate:.2%} "
        f"per_turn_hit_rates={[f'{rate:.2%}' for rate in per_turn_hit_rates]}",
        flush=True,
    )
    return total_cached_tokens, cache_hit_rate


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
    """Verify that multi-turn chat reuses prompt tokens with prefix caching."""
    assert decode_bsz <= KV_CACHE_SZ, "kv cache size should >= decode_bsz"
    assert (seq_len + MAX_END_TOKENS) * ITERATIONS <= ctx_len

    cached_with, hit_rate_with = _run_multichat(
        make_runner, device_group, sharegpt_prompts, decode_bsz, prefix_caching=True
    )
    cached_without, hit_rate_without = _run_multichat(
        make_runner, device_group, sharegpt_prompts, decode_bsz, prefix_caching=False
    )

    print(
        "Prefix cache result: "
        f"enabled={cached_with} cached tokens ({hit_rate_with:.2%} hit rate), "
        f"disabled={cached_without} cached tokens ({hit_rate_without:.2%} hit rate)"
    )
    assert cached_without == 0, (
        f"Prefix caching disabled but {cached_without} cached tokens were reported"
    )
    assert cached_with > cached_without, (
        "Prefix caching did not increase the number of cached prompt tokens"
    )
    assert hit_rate_with > hit_rate_without, (
        "Prefix caching did not increase the prompt-token cache hit rate"
    )
