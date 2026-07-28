# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
CONFIGS = {
    "ByteDance-Seed/Seed-OSS-36B-Instruct": {"tp_size": 8},
    "chuhac/TeleChat2-35B": {"tp_size": 8},
    "CofeAI/FLM-2-52B-Instruct-2407": {"tp_size": 8},
    "CofeAI/Tele-FLM": {"tp_size": 8, "gpu_memory_utilization": 0.98},
    "facebook/opt-66b": {"tp_size": 8},
    "facebook/opt-iml-max-30b": {"tp_size": 8},
    "google/gemma-2-9b": {"dtype": "float32"},
    "google/gemma-2-27b": {"dtype": "float32"},
    "google/gemma-3-1b-it": {"dtype": "float32"},
    "xverse/XVERSE-7B-Chat": {"tokenizer_mode": "slow"},
    "zai-org/GLM-4-32B-0414": {"dtype": "float32"},
    "tiiuae/falcon-7b": {"tp_size": 1, "trust_remote_code": False},
    "tiiuae/falcon-40b": {"tp_size": 8, "trust_remote_code": False},
    "tiiuae/falcon-rw-7b": {"tp_size": 1, "trust_remote_code": False},
    "tiiuae/Falcon-H1-34B-Base": {"tp_size": 8},
    "tiiuae/Falcon-H1-34B-Instruct": {"tp_size": 8},
    "pfnet/plamo-2-1b": {"dtype": "bfloat16"},
    "pfnet/plamo-2-8b": {"dtype": "bfloat16"},
    "EssentialAI/rnj-1-instruct": {"dtype": "float32"},
    "google/gemma-2b": {"dtype": "float32"},
    "google/gemma-3n-E2B-it": {"dtype": "float32"},
    "upstage/solar-pro-preview-instruct": {"tp_size": 1},
    "Zyphra/Zamba2-7B-instruct": {"tp_size": 1},
    "microsoft/Phi-3.5-MoE-instruct": {"tp_size": 8, "gpu_memory_utilization": 0.98},
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "tp_size": 8,
        "gpu_memory_utilization": 0.98,
    },
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1": {
        "tp_size": 8,
        "gpu_memory_utilization": 0.98,
    },
}


def get_config(model_name):
    """
    Returns the model-specific configuration dictionary for LLMs.
    Returns an empty dict if the model is not found, allowing
    the main script to apply its own default settings.
    """
    return CONFIGS.get(model_name, {})
