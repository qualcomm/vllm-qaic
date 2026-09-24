# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Figures out which pooling tasks an embed model actually supports, once, and
caches the result via model_configs_embed.mark_discovered() so no later
sweep run ever has to rebuild the model just to re-learn something already
known.

Used two ways:
  - Imported by run_embed.py: discover(model_name, tp_size) is called
    inline the first time a model is seen.
  - Run standalone as a batch pre-pass over a scraped model list, e.g.
    right after scrape_models.py writes a new embed_models_<ref>.json:
      python discover_tasks.py --ref v0.23.0
"""

import argparse
import json
from pathlib import Path

import model_configs_embed
from vllm import LLM


def discover(model_name, tp_size):
    """
    One build, no pooler_config -- let vLLM auto-pick a task, then read the
    full set the checkpoint actually supports. Persists the result either
    way. Returns (tasks, llm): tasks is [] if the model is unsupported or
    has no usable pooling task; llm is the just-built instance when tasks
    is non-empty (the caller can reuse it directly when len(tasks) == 1,
    since that was the only task vLLM could have picked), else None.
    """
    llm_kwargs = model_configs_embed.build_llm_kwargs(model_name, tp_size)
    try:
        llm = LLM(**llm_kwargs)
    except Exception as e:
        print(f"[discover] {model_name}: build failed: {type(e).__name__}: {e}")
        model_configs_embed.mark_discovered(model_name, tasks=[])
        return [], None

    if llm.runner_type != "pooling":
        print(f"[discover] {model_name}: resolved to runner_type={llm.runner_type!r}")
        model_configs_embed.teardown_llm(llm)
        model_configs_embed.mark_discovered(model_name, tasks=[])
        return [], None

    tasks = sorted(set(llm.supported_tasks) & model_configs_embed.POOLING_TASKS)
    if not tasks:
        print(f"[discover] {model_name}: no usable pooling task")
        model_configs_embed.teardown_llm(llm)
        model_configs_embed.mark_discovered(model_name, tasks=[])
        return [], None

    print(f"[discover] {model_name}: tasks={tasks}")
    model_configs_embed.mark_discovered(model_name, tasks=tasks)
    return tasks, llm


def _discover_all(ref, models_dir, force):
    json_path = Path(models_dir) / f"embed_models_{ref.replace('/', '_')}.json"
    with open(json_path) as f:
        models = json.load(f)

    for entry in models:
        model_name = entry["model"]
        if not force and not model_configs_embed.get_discover_tasks(model_name):
            continue
        tasks, llm = discover(model_name, tp_size=1)
        if llm is not None:
            model_configs_embed.teardown_llm(llm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch-discover pooling tasks for a scraped embed model list."
    )
    parser.add_argument("--ref", required=True, help="vLLM ref, e.g. v0.23.0")
    parser.add_argument(
        "--models-dir", default=".", help="Directory containing embed_models_<ref>.json"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-discover even already-cached models"
    )
    args = parser.parse_args()
    _discover_all(args.ref, args.models_dir, args.force)
