# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/basic/basic.py

import argparse
import os
import random

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = ["My name is"]

random.shuffle(prompts)
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=20)
# Only Greedy Sampling (temperature < 1e-5) or
# Random sampling with best_of==1 is supported.
# best_of >1 or beam search not supported in current qaic implementation
# Beam search not supported.

# define qpc parameters
ctx_len = 256
seq_len = 128
decode_bsz = 2
# An image adds a few hundred soft tokens to the prompt.
image_ctx_len = 1024

# set QAIC specific environment variables
os.environ["QAIC_VISIBLE_DEVICES"] = (
    "0"  # for multiple devices, separate them through comma, i.e. 0,1,2,3
)


def main(args):
    use_image = args.image is not None
    # Create a LLM.
    llm = LLM(
        model="google/gemma-4-26B-A4B-it",
        max_num_seqs=decode_bsz,  # determines decode batch size
        # ctx_len (does not account for padding, but does account for prompt
        # and generated tokens)
        max_model_len=image_ctx_len if use_image else ctx_len,
        enable_prefix_caching=False,
        dtype="float16",
        gpu_memory_utilization=0.98,
        # The image prompt is prefilled in one step.
        max_num_batched_tokens=image_ctx_len if use_image else seq_len,
        enforce_eager=True,
        async_scheduling=False,
        long_prefill_token_threshold=image_ctx_len if use_image else seq_len,
        tensor_parallel_size=8,
        # Fewer multimodal items keep the vision profiling run small.
        limit_mm_per_prompt=(
            {"image": 1, "video": 0}
            if use_image
            else {"video": {"count": 1, "num_frames": 2}}
        ),
        # buffers
        # Truncate the text model (None keeps all 30 layers).
        hf_overrides=(
            {"text_config": {"num_hidden_layers": args.num_layers}}
            if args.num_layers is not None
            else None
        ),
        # Cap the KV cache size in blocks (None: size from free memory).
        num_gpu_blocks_override=args.num_blocks,
    )
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    if use_image:
        from PIL import Image

        image = Image.open(args.image).convert("RGB")
        # llm.chat applies Gemma 4's chat template, which inserts the image
        # placeholder tokens.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": args.question},
                ],
            }
        ]
        outputs = llm.chat(messages, sampling_params)
    else:
        outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        num_generated_tokens = len(generated_tokens)
        print(
            f"Prompt: {prompt!r}, Generated text: {generated_text!r},\
             Num generated tokens: {num_generated_tokens!r}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QAIC eager mode: text prompt, or an image with --image"
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a local image to ask about; without the flag, run the "
        "text prompt.",
    )
    parser.add_argument(
        "--question",
        default="Describe this image in one sentence.",
        help="Question asked about the image.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate the text model to this many decoder layers "
        "(hf_overrides num_hidden_layers); default: all 30.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Cap the KV cache at this many blocks (num_gpu_blocks_override).",
    )
    main(parser.parse_args())
