# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
CONFIGS = {
    "AIDC-AI/Ovis1.6-Llama3.2-3B": {"tp_size": 1, "max_model_len": 2348},
    "AIDC-AI/Ovis2-1B": {"tp_size": 1, "max_model_len": 2348},
    "AIDC-AI/Ovis2.5-9B": {
        "tp_size": 8,
        "max_model_len": 2048,
        "prompt": "<image>\nDescribe the image in detail.",
        "gpu_memory_utilization": 0.98,
    },
    "MiniMaxAI/MiniMax-VL-01": {"max_model_len": 4096},
    "PaddlePaddle/PaddleOCR-VL": {
        "max_model_len": 4096,
        "prompt": "<|begin_of_sentence|>User: Describe the image in detail."
        "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|\nAssistant:",
    },  # 2063+output tokens
    "Qwen/Qwen-VL": {
        "max_model_len": 2048,
        "prompt": "Describe the image in detail.Picture 1: <img></img>\n",
        "hf_overrides": {"architectures": ["QwenVLForConditionalGeneration"]},
    },
    "Qwen/Qwen-VL-Chat": {
        "max_model_len": 2048,
        "prompt": "Describe the image in detail.Picture 1: <img></img>\n",
        "hf_overrides": {"architectures": ["QwenVLForConditionalGeneration"]},
    },
    "Qwen/Qwen2-Audio-7B-Instruct": {"max_model_len": 4096},
    "Qwen/Qwen2.5-Omni-3B": {"max_model_len": 2064},
    "Qwen/Qwen2.5-Omni-7B": {"max_model_len": 2064},
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "max_model_len": 4096,
        "prompt": "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Describe the image in detail."
        "<|im_end|>\n"
        "<|im_start|>assistant\n",
    },  # 2072+output tokens
    "Qwen/Qwen2-VL-7B-Instruct": {"max_model_len": 4096},  # 2060+output tokens
    "allenai/Molmo-7B-D-0924": {"max_model_len": 4096},
    "allenai/Molmo-7B-O-0924": {"max_model_len": 4096},
    "allenai/Molmo2-4B": {
        "prompt": "<|image|><|im_start|>user\nDescribe the image in detail."
        "<|im_end|>\n<|im_start|>assistant\n"
    },
    "allenai/Molmo2-8B": {
        "prompt": "<|image|><|im_start|>user\nDescribe the image in detail."
        "<|im_end|>\\n<|im_start|>assistant\n"
    },
    "allenai/Molmo2-O-7B": {
        "prompt": "<|image|><|im_start|>user\nDescribe the image in detail.<|im_end|>"
        "           \n<|im_start|>assistant\n"
    },
    "google/gemma-3-4b-it": {
        "dtype": "float32",
        "prompt": "<image>\nDescribe the image in detail.",
    },
    "google/gemma-3n-E2B-it": {"dtype": "float32"},
    "google/paligemma-3b-mix-224": {"max_model_len": 4096},
    "google/paligemma-3b-pt-224": {"max_model_len": 4096},
    "google/paligemma2-3b-ft-docci-448": {"max_model_len": 4096},
    "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B": {
        "prompt": "<|im_start|>user (vector)\n"
        "<|dummy3|><|im_end|>\n"
        "<|im_start|>user\n"
        "Describe the image in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    },
    "Kwai-Keye/Keye-VL-8B-Preview": {
        "prompt": "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Describe the image in detail."
    },
    "Kwai-Keye/Keye-VL-1_5-8B": {
        "prompt": "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Describe the image in detail.",
        "max_model_len": 2060,
    },
    "llava-hf/LLaVA-NeXT-Video-7B-hf": {
        "max_model_len": 4096,
        "limit_mm_per_prompt": {
            "image": {
                "count": 1,
                "width": 512,
                "height": 512,
            },  # Overriding default 224 resolution
            "video": {
                "count": 1,
                "num_frames": 32,
                "fps": 2,
                "width": 640,
                "height": 640,
            },
        },
    },
    "llava-hf/llava-1.5-7b-hf": {"max_model_len": 4096},
    "llava-hf/llava-v1.6-mistral-7b-hf": {"max_model_len": 4096},  # 2097+output tokens
    "llava-hf/llava-onevision-qwen2-7b-ov-hf": {
        "max_model_len": 8000
    },  # 7253+output tokens
    "TIGER-Lab/Mantis-8B-siglip-llama3": {"max_model_len": 4096},
    "microsoft/Phi-3-vision-128k-instruct": {
        "prompt": "<|user|>\n<|image_1|>\n"
        "Describe the image in detail.<|end|>"
        "\n<|assistant|>\n"
    },
    "microsoft/Phi-3.5-vision-instruct": {
        "prompt": "<|user|>\\n<|image_1|>\\n"
        "Describe the image in detail."
        "<|end|>\\n<|assistant|>\\n"
    },
    "microsoft/Phi-4-reasoning-vision-15B": {
        "tp_size": 1,
        "gpu_memory_utilization": 0.98,
    },
    "microsoft/Phi-4-multimodal-instruct": {
        "prompt": "<|user|>\n"
        "<|image_1|>\n"
        "Describe the image in detail.\n"
        "<|end|>\n"
        "<|assistant|>\n"
    },
    "mistral-community/pixtral-12b": {"max_model_len": 4096, "mistral_tokenize": True},
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": {
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "mistral_tokenize": True,
    },
    "nvidia/Cosmos3-Nano": {
        "prompt": "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        "Describe the image in detail."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    },
    "openbmb/MiniCPM-o-2_6": {
        "max_model_len": 4096,
        "prompt": "(<image>./</image>)\nDescribe the image in detail.",
    },
    "Open-Bee/Bee-8B-RL": {"max_model_len": 6000},  # 5056+output tokens
    "rhymes-ai/Aria": {
        "prompt": "<|im_start|>user\n"
        "<fim_prefix><|img|><fim_suffix>"
        "Describe the image in detail."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    },
    "stepfun-ai/Step3-VL-10B": {
        "prompt": "<|begin_of_sentence|> You are a helpful assistant."
        "<|BOT|>user\\n<im_patch>Describe the image"
        "<|EOT|><|BOT|>assistant\\n<think>\\n"
    },
    "Salesforce/blip2-opt-2.7b": {"max_model_len": 2048},
    "deepseek-ai/DeepSeek-OCR": {"tp_size": 1},
    "deepseek-ai/DeepSeek-OCR-2": {"tp_size": 2},
    "OpenGVLab/InternVL3-1B-hf": {
        "tp_size": 2,
        "max_model_len": 4096,  # 2835+output tokens
        "prompt": "<|im_start|>user\n"
        "<img><IMG_CONTEXT></img>\n"
        "Describe the image in detail."
        "<|im_end|>\n"
        "<|im_start|>assistant\n",
    },
    "openvla/openvla-7b": {"max_model_len": 2048},
    "xlangai/OpenCUA-7B": {
        "max_model_len": 2105,
        "prompt": "<|im_start|>system\\nYou are a GUI agent."
        "You are given a task and a screenshot of the screen."
        "You need to perform actions to complete the task."
        "<|im_end|>\\n<|im_start|>user\\n<|media_placeholder|>"
        "\\nDescribe the image.\\n<|im_end|>\\n<|im_start|>assistant\\n",
    },
    "YannQi/R-4B": {"max_model_len": 4096},  # 2680+output tokens
    "zai-org/GLM-4.1V-9B-Thinking": {
        "prompt": "[gMASK]<sop><|user|><|begin_of_image|>"
        "<|image|><|end_of_image|>Describe the image in detail."
        "<|assistant|>"
    },
    "zai-org/GLM-OCR": {"prompt": "<|image|>\nDescribe the image in detail."},
}


def get_config(model_name):
    """
    Returns the model-specific configuration dictionary.
    Returns an empty dict if the model is not found, allowing
    main script to handle its own defaults.
    """
    return CONFIGS.get(model_name, {})
