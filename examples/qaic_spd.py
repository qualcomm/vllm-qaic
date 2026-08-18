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


def main() -> None:
    # Sample prompts.
    prompts = [
        "The history of artificial intelligence dates back",
        "Speculative decoding accelerates inference by",
        "Large language models are trained on",
        "The key difference between supervised and unsupervised learning is",
    ] * 5

    random.shuffle(prompts)

    # Create a sampling params object.
    # Only Greedy Sampling (temperature < 1e-5) or
    # Random sampling with best_of==1 is supported.
    # best_of >1 or beam search not supported in current qaic implementation.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # QPC parameters
    ctx_len = 2048
    seq_len = 128
    decode_bsz = 4

    qid = 0
    tlm_cores = 10
    dlm_cores = 6  # = 16 - tlm_cores

    print("LLM draft-model SpD run (Llama-3.1-8B → Llama-3.2-1B)...\n")

    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_num_seqs=decode_bsz,  # determines decode batch size
        max_model_len=ctx_len,  # ctx_len (prompt + generated tokens, no padding)
        long_prefill_token_threshold=seq_len,  # seq_len
        quantization="mxfp6",  # Preferred quantization
        kv_cache_dtype="mxint8",  # Preferred option to save KV cache performance
        disable_log_stats=False,
        gpu_memory_utilization=1.0,
        async_scheduling=False,
        additional_config={
            "override_qaic_config": {
                "device_group": [qid],
                "num_cores": tlm_cores,
            },
            "draft_override_qaic_config": {
                "device_group": [qid],  # same device as TLM
                "num_cores": dlm_cores,
            },
        },
        speculative_config={
            "method": "draft_model",
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "num_speculative_tokens": 3,
        },
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_generated_tokens = len(output.outputs[0].token_ids)
        print(
            f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
            f"Num generated tokens: {num_generated_tokens!r}"
        )

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
