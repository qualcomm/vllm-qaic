# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import argparse
import os
import random

import discover_tasks
import model_configs_embed
from hf_cache import delete_hf_checkpoint
from vllm import LLM

# Sample prompts.
prompts = [
    "My name is"
    # Add more prompts here
]

# Sample query/passage pair for cross-encoder/reranker (score) models
query_texts = ["What is the capital of France?"]
passage_texts = ["Paris is the capital of France."]

random.shuffle(prompts)

# set QAIC specific environment variables
# setdefault, not assignment, so ci_fallback_ops.sh's export wins.
os.environ.setdefault("QAIC_VISIBLE_DEVICES", "60,61,62,63")


def cleanup(model_name=None, delete_checkpoint=False):
    if delete_checkpoint and model_name:
        delete_hf_checkpoint(model_name)


def run_task(llm, task):
    results = llm.encode(prompts, pooling_task=task)
    for r in results:
        print(f"[{task}] Prompt Token Id Shape: {len(r.prompt_token_ids)}")
        print(f"[{task}] Output shape: {tuple(r.outputs.data.shape)}")

    # Cross-encoder rerankers (classify + num_labels == 1) additionally
    # expose the two-input scoring API -- exercise that code path too.
    num_labels = getattr(llm.model_config.hf_config, "num_labels", None)
    if task == "classify" and num_labels == 1:
        score_results = llm.score(query_texts, passage_texts)
        for r in score_results:
            print(f"[score] Score: {r.outputs.score}")


def test_embed_vllm(
    model_name: str, tp_size: int, gen_len: int, delete_hf_checkpoint=False
):
    model_name = model_name.replace("--", "/")
    tp_size = model_configs_embed.get_tp_size(model_name, tp_size)
    print(f"Model:{model_name}, TP_SIZE:{tp_size}")

    llm = None
    try:
        if model_configs_embed.get_discover_tasks(model_name):
            tasks, llm = discover_tasks.discover(model_name, tp_size)
            if not tasks:
                print(f"SKIP: {model_name}")
                return
            if len(tasks) == 1:
                # discovery's own build already auto-picked this one task
                # (it was the only candidate) -- reuse it, no rebuild. The
                # top-level finally below still tears it down.
                run_task(llm, tasks[0])
                return
            model_configs_embed.teardown_llm(llm)
            llm = None
        else:
            tasks = model_configs_embed.get_tasks(model_name)
            if not tasks:
                print(f"SKIP (cached): {model_name} -- see its discovery log for why")
                return

        for task in tasks:
            llm = LLM(**model_configs_embed.build_llm_kwargs(model_name, tp_size, task))
            run_task(llm, task)
            model_configs_embed.teardown_llm(llm)
            llm = None
    finally:
        if llm is not None:
            model_configs_embed.teardown_llm(llm)
            llm = None
        cleanup(model_name=model_name, delete_checkpoint=delete_hf_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--gen-len", type=int, default=2)
    parser.add_argument(
        "--delete-hf-checkpoint",
        action="store_true",
        help="Delete the model's cached HF checkpoint once the run is done",
    )

    args = parser.parse_args()
    test_embed_vllm(**(args.__dict__))
