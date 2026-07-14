# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/basic/classify.py

from vllm import LLM

# Classification example for QAIC.
# llm.classify(prompts) runs a sequence-classification model and returns a
# probability/logit vector over the label set for each prompt.
#
# Use 'task: "classify"' in override_qaic_config to select
# QEFFAutoModelForSequenceClassification (the correct model class for classifiers).
# For QAIC pooling, also pass 'softmax: True' to get probabilities instead of
# raw logits.


def print_classifications(prompts, outputs):
    print("\nGenerated Classifications:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs, strict=True):
        probs = output.outputs.probs
        probs_trimmed = (
            (str(probs[:8])[:-1] + ", ...]") if len(probs) > 8 else probs
        )
        print(
            f"Prompt: {prompt!r}\n"
            f"Probs:  {probs_trimmed} (num_labels={len(probs)})"
        )
        print("-" * 60)


def main():
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    print("running classify with CPU pooling")
    model = LLM(
        model="BAAI/bge-reranker-v2-m3",
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=512,
        additional_config={
            "device_group": [5],
            "override_qaic_config": {"pooling_device": "cpu", "task": "classify"},
        },
    )
    outputs = model.classify(prompts)
    print_classifications(prompts, outputs)

    print("running classify with QAIC pooling")
    model = LLM(
        model="BAAI/bge-reranker-v2-m3",
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=512,
        additional_config={
            "device_group": [6],
            "override_qaic_config": {"pooling_device": "qaic", "task": "classify"},
        },
    )
    outputs = model.classify(prompts)
    print_classifications(prompts, outputs)


if __name__ == "__main__":
    main()
