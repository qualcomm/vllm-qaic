# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

# QAIC on-device sampling example. Serve equivalent:
#   vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-model-len 256 \
#     --max-num-seqs 2 --long-prefill-token-threshold 128 \
#     --quantization mxfp6 --kv-cache-dtype mxint8 --additional-config \
#     '{"override_qaic_config":{"aic_include_sampler":true,"max_top_k_ids":64}}'

import gc

from vllm import LLM, SamplingParams

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def run() -> None:
    llm = LLM(
        model=MODEL_NAME,
        max_num_seqs=2,
        max_model_len=256,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        long_prefill_token_threshold=128,
        additional_config={
            "override_qaic_config": {"aic_include_sampler": True, "max_top_k_ids": 64}
        },
    )
    prompts = [
        "My name is",
        "On-device sampling can reduce host-side token processing by",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=40)
    for output in llm.generate(prompts, params):
        print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")

    sampled_params = SamplingParams(temperature=0.7, top_k=32, top_p=0.9, max_tokens=40)
    for output in llm.generate(
        ["The key benefit of accelerator-side token selection is"], sampled_params
    ):
        print(f"Prompt: {output.prompt!r}, Generated: {output.outputs[0].text!r}")

    del llm
    gc.collect()


if __name__ == "__main__":
    run()
