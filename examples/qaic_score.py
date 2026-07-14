# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/basic/score.py

from vllm import LLM

# Score (reranking) example for QAIC.
# llm.score(query, passages) runs a cross-encoder model that produces a
# relevance score for each (query, passage) pair.
#
# Use 'task: "score"' in override_qaic_config to select
# QEFFAutoModelForSequenceClassification (the correct model class for rerankers).


def print_scores(query, passages, outputs):
    print("\nGenerated Scores:\n" + "-" * 60)
    print(f"Query: {query!r}")
    for passage, output in zip(passages, outputs, strict=True):
        score = output.outputs.score
        print(f"Passage: {passage!r}\nScore:   {score:.4f}")
        print("-" * 60)


def main():
    # A single query and a list of candidate passages to rank.
    query = "What is the capital of France?"
    passages = [
        "Paris is the capital and most populous city of France.",
        "The Eiffel Tower is located in Paris.",
        "Berlin is the capital of Germany.",
        "The capital of France is Paris.",
    ]

    print("running score with CPU pooling")
    model = LLM(
        model="BAAI/bge-reranker-v2-m3",
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=512,
        additional_config={
            "device_group": [5],
            "override_qaic_config": {"pooling_device": "cpu", "task": "score"},
        },
    )
    outputs = model.score(query, passages)
    print_scores(query, passages, outputs)

    print("running score with QAIC pooling")
    model = LLM(
        model="BAAI/bge-reranker-v2-m3",
        runner="pooling",
        enforce_eager=True,
        max_num_seqs=4,
        max_model_len=512,
        additional_config={
            "device_group": [6],
            "override_qaic_config": {"pooling_device": "qaic", "task": "score"},
        },
    )
    outputs = model.score(query, passages)
    print_scores(query, passages, outputs)


if __name__ == "__main__":
    main()
