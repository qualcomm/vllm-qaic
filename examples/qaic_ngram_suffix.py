# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/basic/basic.py

import gc
import random

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "My name is",
    # Add more prompts here
] * 20

random.shuffle(prompts)
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)
# Only Greedy Sampling (temperature < 1e-5) or
# Random sampling with best_of==1 is supported.
# best_of >1 or beam search not supported in current qaic implementation
# Beam search not supported.

# define qpc parameters
ctx_len = 256
seq_len = 128
decode_bsz = 16

print("N-gram spd run....\n")
# Create a LLM.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=decode_bsz,  # determines decode batch size
    max_model_len=ctx_len,  # ctx_len (prompt + generated tokens, no padding)
    long_prefill_token_threshold=seq_len,  # seq_len
    quantization="mxfp6",  # Preferred quantization
    kv_cache_dtype="mxint8",  # Preferred option to save KV cache performance
    disable_log_stats=False,
    async_scheduling=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=1.0,
    speculative_config={
        "num_speculative_tokens": 5,
        "method": "ngram",
    },
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_tokens = output.outputs[0].token_ids
    num_generated_tokens = len(generated_tokens)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
        f"Num generated tokens: {num_generated_tokens!r}"
    )

del llm

gc.collect()

print("Suffix spd run....\n")
# Create a LLM.
llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=decode_bsz,  # determines decode batch size
    max_model_len=ctx_len,  # ctx_len (prompt + generated tokens, no padding)
    long_prefill_token_threshold=seq_len,  # seq_len
    quantization="mxfp6",  # Preferred quantization
    kv_cache_dtype="mxint8",  # Preferred option to save KV cache performance
    disable_log_stats=False,
    async_scheduling=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=1.0,
    speculative_config={
        "num_speculative_tokens": 5,
        "method": "suffix",
    },
    additional_config={"device_group": [0]},
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_tokens = output.outputs[0].token_ids
    num_generated_tokens = len(generated_tokens)
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
        f"Num generated tokens: {num_generated_tokens!r}"
    )
