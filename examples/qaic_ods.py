# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0


"""
QAIC on-device sampling example
"""

import gc
from vllm import LLM, SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTX_LEN = 256
SEQ_LEN = 128
DECODE_BSZ = 2


def run() -> None:
    llm = LLM(
        model=MODEL_NAME,
        max_num_seqs=DECODE_BSZ,
        max_model_len=CTX_LEN,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        long_prefill_token_threshold=SEQ_LEN,
        additional_config={
            "override_qaic_config": {
                "aic_include_sampler": 1,
                "aic_return_pdfs": 1,
                "max_top_k_ids": 64,
            }
        },
    )

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
        top_k=32,
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


if __name__ == "__main__":
    run()


"""
Online QAIC using vllm serve

Server command:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 4 \
    --long-prefill-token-threshold 128 \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --additional-config \
    '{"override_qaic_config":{"aic_include_sampler":1,"max_top_k_ids":64}}'

Sample curl command:

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize why ODS helps QAIC serving."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 64
    }'
"""
