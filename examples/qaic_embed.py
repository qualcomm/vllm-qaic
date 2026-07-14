# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/basic/embed.py

from vllm import LLM

# Set example specific arguments
# For cpu pooling pass 'override_qaic_config={"pooling_device":"cpu"}' and for
# qaic pooling an example would be:
# 'override_qaic_config={"pooling_device":"qaic", "pooling_method":"mean"}'
# To compile for multiple sequence lengths pass multi_seq_lens in the following
# format: 'override_qaic_config={"embed_seq_len":[32,512]}'. Always include
# max_model_len in embed_seq_len.


def print_embeds(prompts, outputs):
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs, strict=False):
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(f"Prompt: {prompt!r} \nEmbeddings: {embeds_trimmed} (size={len(embeds)})")
        print("-" * 60)


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
    ] * 10
    # Create an LLM.
    # You should pass runner="pooling" for embedding models
    print("running multi specialization with pooling on cpu")
    model = LLM(
        model="intfloat/multilingual-e5-large",
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=256,
        additional_config={
            "device_group": [0],
            "override_qaic_config": {
                "pooling_device": "cpu",
                "embed_seq_len": [32, 256],
            },
        },
    )
    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = model.embed(prompts)

    # Print the outputs.
    print_embeds(prompts, outputs)

    print("running single specialization with pooling on qaic")
    model = LLM(
        model="intfloat/multilingual-e5-large",
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=256,
        additional_config={
            "device_group": [0],
            "override_qaic_config": {
                "pooling_device": "qaic",
                "pooling_method": "mean",
                "normalize": True,
            },
        },
    )

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    outputs = model.embed(prompts)

    # Print the outputs.
    print_embeds(prompts, outputs)


if __name__ == "__main__":
    main()
