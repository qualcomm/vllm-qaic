# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from pathlib import Path

import pytest
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

from ...conftest import VllmRunner


def _resolve(request, pytestconfig, key, default=None):
    """Resolve a QAIC run-config value: closest `qaic_test_config` marker
    kwarg first, then the matching CLI option, then `default`."""
    marker = request.node.get_closest_marker("qaic_test_config")
    if marker is not None and key in marker.kwargs:
        return marker.kwargs[key]
    cli_value = pytestconfig.getoption(key, default=None)
    if cli_value is not None:
        return cli_value
    return default


@pytest.fixture
def qaic_test_config(request, pytestconfig):
    """Per-test QAIC run configuration, resolved from the `qaic_test_config`
    marker kwargs first, then matching CLI options, then defaults."""
    defaults = {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "seq_len": 32,
        "ctx_len": 256,
        "decode_bsz": 4,
        "dtype": "mxfp6",
        "kv_dtype": "mxint8",
        "override_qaic_config": None,
    }
    return {
        key: _resolve(request, pytestconfig, key, default=default)
        for key, default in defaults.items()
    }


@pytest.fixture
def make_runner(qaic_test_config):
    def _make(
        dg: list,
        ctx_len: int | None = None,
        seq_len: int | None = None,
        decode_bsz: int | None = None,
        async_scheduling: bool = False,
        enable_chunked_prefill: bool = False,
        prefix_caching: bool = False,
        **kwargs,
    ) -> VllmRunner:
        _seq_len = seq_len if seq_len is not None else qaic_test_config["seq_len"]
        _override = dict(qaic_test_config["override_qaic_config"] or {})
        _override["prefill_seq_len"] = _seq_len
        return VllmRunner(
            qaic_test_config["model_name"],
            max_num_seqs=decode_bsz
            if decode_bsz is not None
            else qaic_test_config["decode_bsz"],
            max_model_len=ctx_len
            if ctx_len is not None
            else qaic_test_config["ctx_len"],
            quantization=qaic_test_config["dtype"],
            kv_cache_dtype=qaic_test_config["kv_dtype"],
            additional_config={
                "override_qaic_config": _override,
                "device_group": dg,
            },
            enable_prefix_caching=prefix_caching,
            async_scheduling=async_scheduling,
            disable_log_stats=False,
            enable_chunked_prefill=enable_chunked_prefill,
            **kwargs,
        )

    return _make


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


# T1: Validate PA model with different prompt lengths
def test_with_different_prompt_lengths(make_runner, device_group, qaic_test_config):
    """
    Validate how block_table and slot_id updates happen for the following
    three scenarios: a) prompt_len < block_size b) prompt_len = block_size
    c) prompt_len > block_size. We have three prompts such that P1 < 32
    tokens, P2 = 32 tokens and P3 > 32 tokens and we test with a
    seq_len = block_size = 32.
    """
    base = (
        "My name is John Smith and I am a software engineer. "
        "I have been working in the industry for the past 5 years "
        "and have been fortunate"
    )
    prompts = [
        base,
        base + " enough to",
        base + " enough to work on some",
    ]
    sampling_params = SamplingParams(temperature=0)
    with make_runner(
        dg=device_group,
        ctx_len=256,
        seq_len=32,
        enable_chunked_prefill=True,
        prefix_caching=True,
    ) as runner:
        outputs = runner.llm.generate(prompts, sampling_params=sampling_params)

    final_outputs = [output.prompt + output.outputs[0].text for output in outputs]
    tokenizer = AutoTokenizer.from_pretrained(qaic_test_config["model_name"])
    min_tokens_to_check = len(
        tokenizer.encode(final_outputs[0], add_special_tokens=False)
    )
    for output in final_outputs:
        assert output[:min_tokens_to_check] == final_outputs[0][:min_tokens_to_check], (
            "Outputs do not match for different prompt lengths!!"
        )


# T2: Validate PA model outputs across various batch sizes, num_kv_blocks,
# and ctx_lengths - this also validates batch invariance
def test_pa_model_outputs(make_runner, device_group):
    prompts = ["My name is"] * 16
    batch_sizes = [1, 4, 16]
    ctx_lengths = [512, 1024, 2048]
    seq_lens = [32, 128]
    sampling_params = SamplingParams(temperature=0)

    results = []
    for decode_bsz in batch_sizes:
        for ctx_len in ctx_lengths:
            for seq_len in seq_lens:
                with make_runner(
                    dg=device_group,
                    decode_bsz=decode_bsz,
                    ctx_len=ctx_len,
                    seq_len=seq_len,
                    enable_chunked_prefill=True,
                    prefix_caching=True,
                ) as runner:
                    outputs = runner.llm.generate(
                        prompts, sampling_params=sampling_params
                    )
                for output in outputs:
                    results.append(
                        {"prompt": output.prompt, "output": output.outputs[0].text}
                    )

    base_prompt, base_text = results[0]["prompt"], results[0]["output"]
    for res in results:
        assert res["prompt"] == base_prompt, "Prompts are not matching"
        assert res["output"] == base_text, "Generated text is not same"


# T3: Performance gains on prefix caching
@pytest.mark.qaic_test_config(num_device_groups=2, device_group_size=1)
def test_prefix_caching_performance(make_runner, device_groups, qaic_test_config):
    sampling_params = SamplingParams(temperature=0)
    sys_prompt = (Path(__file__).resolve().parents[4] / "15k_prompt.txt").read_text()
    tokenizer = AutoTokenizer.from_pretrained(qaic_test_config["model_name"])
    sys_tokens = tokenizer.encode(sys_prompt, add_special_tokens=False)
    sys_prompt = tokenizer.decode(sys_tokens[:8000])
    sub_prompt_1 = tokenizer.decode(sys_tokens[:2500])
    sub_prompt_2 = tokenizer.decode(sys_tokens[:5000])
    sub_prompt_3 = tokenizer.decode(sys_tokens[:7500])
    warmup_prompts = [sys_prompt] * qaic_test_config["decode_bsz"]
    prompts = [sys_prompt, sub_prompt_1, sub_prompt_2, sub_prompt_3]

    with make_runner(
        dg=device_groups[0], enable_chunked_prefill=True, prefix_caching=True
    ) as runner:
        # Warmup run to cache the KV blocks.
        runner.llm.generate(warmup_prompts, sampling_params=sampling_params)
        outputs_1 = runner.llm.generate(prompts, sampling_params=sampling_params)

    with make_runner(
        dg=device_groups[1], enable_chunked_prefill=True, prefix_caching=False
    ) as runner:
        outputs_2 = runner.llm.generate(prompts, sampling_params=sampling_params)

    ttfts_1, _, _, _ = _extract_metrics(outputs_1)
    ttfts_2, _, _, _ = _extract_metrics(outputs_2)
    avg_ttft_1 = abs(sum(ttfts_1) / len(ttfts_1))
    avg_ttft_2 = abs(sum(ttfts_2) / len(ttfts_2))
    print(
        f"Avg TTFT with prefix_cache: {avg_ttft_1:.3f}s | "
        f"Avg TTFT without prefix_cache: {avg_ttft_2:.3f}s"
    )
    print(f"Prefix Caching Speedup : {avg_ttft_2 / avg_ttft_1:.3f}x")
    assert avg_ttft_1 < avg_ttft_2, "No prefix caching benefit observed in TTFT"


@pytest.mark.qaic_test_config(num_device_groups=2, device_group_size=1)
def test_paged_attention_spd(device_groups, qaic_test_config):
    baseline_device_group, test_device_group = device_groups
    model_name = qaic_test_config["model_name"]
    ctx_len = qaic_test_config["ctx_len"]
    seq_len = qaic_test_config["seq_len"]
    decode_bsz = qaic_test_config["decode_bsz"]

    qllm_base = LLM(
        model=model_name,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        async_scheduling=False,
        disable_log_stats=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=1.0,
        additional_config={"device_group": baseline_device_group},
    )
    qllm_spd = LLM(
        model=model_name,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        disable_log_stats=False,
        async_scheduling=False,
        enable_prefix_caching=True,
        gpu_memory_utilization=1.0,
        speculative_config={
            "method": "draft_model",
            "num_speculative_tokens": 5,
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "quantization": "mxfp6",
        },
        additional_config={
            "device_group": baseline_device_group,
            "draft_override_qaic_config": {"device_group": test_device_group},
        },
    )

    sampling_params = SamplingParams(temperature=0)
    prompts = [
        "The history of artificial intelligence dates back",
        "Speculative decoding accelerates inference by",
        "Large language models are trained on",
        "The key difference between supervised and unsupervised learning is",
    ] * 5
    outputs_base = qllm_base.generate(prompts, sampling_params)
    outputs_spd = qllm_spd.generate(prompts, sampling_params)

    for op_base, op_spd in zip(outputs_base, outputs_spd, strict=False):
        assert op_base.prompt == op_spd.prompt
        assert op_base.outputs[0].text == op_spd.outputs[0].text
        assert op_base.outputs[0].token_ids == op_spd.outputs[0].token_ids
