# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import json
import sys

from vllm import LLM, SamplingParams


def main() -> int:
    method = sys.argv[1]
    output_path = sys.argv[2]
    # Optional third argument "sample" switches to temperature>0 so the
    # random-rejection and recovered-token kernels run, not just greedy.
    temperature = 0.8 if len(sys.argv) > 3 and sys.argv[3] == "sample" else 0.0
    sampling_params = SamplingParams(temperature=temperature, max_tokens=32, seed=42)
    kwargs = dict(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype="float16",  # QAIC has no bf16 support
        max_num_seqs=16,
        max_model_len=256,
        long_prefill_token_threshold=128,
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.9,
    )
    if method != "baseline":
        kwargs["speculative_config"] = {
            "num_speculative_tokens": 5,
            "method": method,
        }

    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(
            [
                "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
                "My name is",
            ],
            sampling_params,
        )
        token_ids = [list(output.outputs[0].token_ids) for output in outputs]
        with open(output_path, "w") as output_file:
            json.dump(token_ids, output_file)
    finally:
        del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
