# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/qaic.py to demonstrate weight-free export.

import os

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "My name is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=50)
# Only Greedy Sampling (temperature < 1e-5) or
# Random sampling with best_of==1 is supported.
# best_of >1 or beam search not supported in current qaic implementation
# Beam search not supported.

# define qpc parameters
ctx_len = 128
seq_len = 32
decode_bsz = 16

# set QAIC specific environment variables through environment variable
os.environ["QAIC_VISIBLE_DEVICES"] = (
    "0,1,2,3"  # for multiple devices, separate them through comma
)
# Alternatively: additional_config={"device_group": [0, 1, 2, 3]}
# device_group uses local ordinals (0..N-1), not absolute physical IDs.


def main():
    # weight_free export traces on the meta device, skipping real weight loads.
    llm = LLM(
        model="openai/gpt-oss-20b",
        dtype="float16",
        max_num_seqs=decode_bsz,  # determines decode batch size
        # ctx_len (does not account for padding, but does account for prompt
        # and generated tokens)
        max_model_len=ctx_len,
        quantization="mxfp6",  # Preferred quantization
        # Preferred option to same KV cache and increase performance
        kv_cache_dtype="mxint8",
        disable_log_stats=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=1.0,
        async_scheduling=False,
        long_prefill_token_threshold=seq_len,
        additional_config={
            "override_qaic_config": {
                "weight_free": True,
                # Not implied by weight_free; QEfficient's docs/tests pair it with True.
                "use_onnx_subfunctions": True,
            },
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


if __name__ == "__main__":
    main()
