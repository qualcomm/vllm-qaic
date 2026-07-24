# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/qaic.py

"""
Offline QAIC on-device sampling (ODS) example.
"""

import gc

from vllm import LLM, SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTX_LEN = 256
SEQ_LEN = 128
DECODE_BSZ = 2


def run_baseline_config() -> None:
    llm = None
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_num_seqs=DECODE_BSZ,
            max_model_len=CTX_LEN,
            quantization="mxfp6",
            kv_cache_dtype="mxint8",
            disable_log_stats=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=1.0,
            async_scheduling=False,
            long_prefill_token_threshold=SEQ_LEN,
            additional_config={
                "override_qaic_config": {
                    "aic_include_sampler": 1,
                },
            },
        )
    except Exception as err:
        print(f"[FAIL] Baseline setup failed: {err}")
        return

    greedy_prompts = [
        "My name is",
        "On-device sampling can reduce host-side token processing by",
    ]
    greedy_params = SamplingParams(temperature=0.0, max_tokens=40)
    try:
        outputs = llm.generate(greedy_prompts, greedy_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_generated_tokens = len(output.outputs[0].token_ids)
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
                f"Num generated tokens: {num_generated_tokens!r}"
            )
    except Exception as err:
        print(f"[FAIL] Baseline greedy request failed: {err}")

    non_greedy_prompts = [
        "The key benefit of accelerator-side token selection is",
        "A practical QAIC deployment should monitor",
    ]
    non_greedy_params = SamplingParams(
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        max_tokens=40,
    )
    try:
        outputs = llm.generate(non_greedy_prompts, non_greedy_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_generated_tokens = len(output.outputs[0].token_ids)
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
                f"Num generated tokens: {num_generated_tokens!r}"
            )
    except Exception as err:
        print(f"[FAIL] Baseline non-greedy request failed: {err}")

    del llm
    gc.collect()


def run_max_top_k_limit_config() -> None:
    llm = None
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_num_seqs=DECODE_BSZ,
            max_model_len=CTX_LEN,
            quantization="mxfp6",
            kv_cache_dtype="mxint8",
            disable_log_stats=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=1.0,
            async_scheduling=False,
            long_prefill_token_threshold=SEQ_LEN,
            additional_config={
                "override_qaic_config": {
                    "aic_include_sampler": 1,
                    "max_top_k_ids": 64,
                },
            },
        )
    except Exception as err:
        print(f"[FAIL] max_top_k_ids setup failed: {err}")
        return

    in_limit_params = SamplingParams(
        temperature=0.7,
        top_k=32,
        top_p=0.9,
        max_tokens=40,
    )
    try:
        outputs = llm.generate(
            ["This request uses a top_k within the configured ODS limit."],
            in_limit_params,
        )
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_generated_tokens = len(output.outputs[0].token_ids)
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
                f"Num generated tokens: {num_generated_tokens!r}"
            )
    except Exception as err:
        print(f"[FAIL] In-limit top_k request failed unexpectedly: {err}")

    over_limit_params = SamplingParams(
        temperature=0.7,
        top_k=999,
        top_p=0.9,
        max_tokens=40,
    )
    try:
        llm.generate(
            ["This request intentionally exceeds max_top_k_ids."],
            over_limit_params,
        )
        print("[FAIL] Over-limit top_k request unexpectedly succeeded.")
    except Exception as err:
        print(f"[PASS] Over-limit top_k request was rejected as expected: {err}")

    del llm
    gc.collect()


def run_debug_submode_config() -> None:
    llm = None
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_num_seqs=DECODE_BSZ,
            max_model_len=CTX_LEN,
            quantization="mxfp6",
            kv_cache_dtype="mxint8",
            disable_log_stats=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=1.0,
            async_scheduling=False,
            long_prefill_token_threshold=SEQ_LEN,
            additional_config={
                "override_qaic_config": {
                    "aic_include_sampler": 1,
                    "aic_return_pdfs": 1,
                },
            },
        )
    except Exception as err:
        print(f"[FAIL] Debug sub-mode setup failed: {err}")
        return

    '''
    NOTE: aic_return_pdfs debug outputs (ods_debug_last_decode_probs/
    ods_debug_last_prefill_probs) are internal worker-process attributes 
    and are not exposed through the public LLM()/RequestOutput API. This 
    example only validates that debug sub-mode loads and runs without 
    error. This is intentional and aligns with the feature guide warning 
    that this mode is not suitable for production.
    '''

    debug_params = SamplingParams(temperature=0.0, max_tokens=40)
    try:
        outputs = llm.generate(
            ["Run one decode step in ODS debug sub-mode."],
            debug_params,
        )
        print("[PASS] Debug sub-mode request succeeded.")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_generated_tokens = len(output.outputs[0].token_ids)
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
                f"Num generated tokens: {num_generated_tokens!r}"
            )
    except Exception as err:
        print(f"[FAIL] Debug sub-mode request failed: {err}")

    del llm
    gc.collect()


def run_async_scheduling_config() -> None:
    llm = None
    try:
        llm = LLM(
            model=MODEL_NAME,
            max_num_seqs=DECODE_BSZ,
            max_model_len=CTX_LEN,
            quantization="mxfp6",
            kv_cache_dtype="mxint8",
            disable_log_stats=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=1.0,
            async_scheduling=True,
            long_prefill_token_threshold=SEQ_LEN,
            additional_config={
                "override_qaic_config": {
                    "aic_include_sampler": 1,
                },
            },
        )
    except Exception as err:
        print(f"[FAIL] async_scheduling setup failed: {err}")
        return

    greedy_prompts = [
        "My name is",
        "On-device sampling can reduce host-side token processing by",
    ]
    greedy_params = SamplingParams(temperature=0.0, max_tokens=40)
    try:
        outputs = llm.generate(greedy_prompts, greedy_params)
        print("[PASS] async_scheduling greedy request succeeded.")
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            num_generated_tokens = len(output.outputs[0].token_ids)
            print(
                f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
                f"Num generated tokens: {num_generated_tokens!r}"
            )
    except Exception as err:
        print(f"[FAIL] async_scheduling greedy request failed: {err}")

    del llm
    gc.collect()


def main() -> None:
    print("=" * 60)
    print("ODS OFFLINE EXAMPLE: BASELINE CONFIG")
    print("=" * 60)
    run_baseline_config()

    print("=" * 60)
    print("ODS OFFLINE EXAMPLE: max_top_k_ids LIMIT CHECK")
    print("=" * 60)
    run_max_top_k_limit_config()

    print("=" * 60)
    print("ODS OFFLINE EXAMPLE: DEBUG SUB-MODE (aic_return_pdfs)")
    print("=" * 60)
    run_debug_submode_config()

    print("=" * 60)
    print("ODS OFFLINE EXAMPLE: ASYNC SCHEDULING (async_scheduling=True)")
    print("=" * 60)
    run_async_scheduling_config()


if __name__ == "__main__":
    main()
