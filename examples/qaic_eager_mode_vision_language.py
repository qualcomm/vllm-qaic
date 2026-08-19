# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/vision_language.py

import argparse
from PIL import Image
from vllm import LLM, SamplingParams

def run_llava(questions: list[str], modality: str) -> dict:
    if modality == "image":
        placeholder = "<image>"
    elif modality == "video":
        placeholder = "<video>"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    prompts = [
        f"USER: {placeholder}\n{question}\nASSISTANT:" for question in questions
    ]

    return {
        "model": "llava-hf/llava-1.5-7b-hf",
        "prompts": prompts,
        "engine_params": {
            "limit_mm_per_prompt": {
                "image": {"count": 1, "width": 160, "height": 160},
                "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
            },
        },
    }

def run_qwen3_vl(questions: list[str], modality: str) -> dict:
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    prompts = [
        (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<|vision_start|>{placeholder}<|vision_end|>\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "prompts": prompts,
        "engine_params": {
            "mm_processor_kwargs": {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
            "limit_mm_per_prompt": {
                "image": {"count": 1, "width": 160, "height": 160},
                "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
            },
        },
    }

def run_qwen2_5_vl(questions: list[str], modality: str) -> dict:
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    prompts = [
        (
            "<|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"<|vision_start|>{placeholder}<|vision_end|>\n"
            f"{question}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return {
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "prompts": prompts,
        "engine_params": {
            "mm_processor_kwargs": {
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
                "fps": 1,
            },
            "limit_mm_per_prompt": {
                "image": {"count": 1, "width": 160, "height": 160},
                "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
            },
        },
    }

MODEL_MAP = {
    "qwen3_vl": run_qwen3_vl,
    "llava": run_llava,
    "qwen2_5_vl": run_qwen2_5_vl,
}

def build_engine_params(args) -> dict:
    return {
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tp_size,
        "enforce_eager": True,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "trust_remote_code": True,
        "gpu_memory_utilization": 0.9,
    }

def main(args):
    if args.model_type not in MODEL_MAP:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    req_data = MODEL_MAP[args.model_type]([args.question], args.modality)

    engine_params = build_engine_params(args)
    engine_params["model"] = req_data["model"]
    engine_params.update(req_data.get("engine_params", {}))

    prompts = req_data["prompts"]

    llm = LLM(**engine_params)

    image = Image.open(args.image_path)

    inputs = {
        "prompt": prompts[0],
        "multi_modal_data": {
            args.modality: [image],
        },
    }

    outputs = llm.generate(
        inputs,
        sampling_params=SamplingParams(
            temperature=args.temperature,
            max_tokens=args.gen_len,
        ),
    )

    print("------ OUTPUT ------")
    for o in outputs:
        print(o.outputs[0].text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo on using vLLM for QAIC eager mode on "
        "vision language models for text generation"
    )

    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        required=True,
        help='Huggingface "model_type".',
    )

    parser.add_argument(
        "--tp-size",
        "-tp",
        type=int,
        default=4,
        help="Tensor parallel size to override the model's default setting.",
    )

    parser.add_argument(
        "--gen-len",
        type=int,
        default=64,
        help="Number of tokens to generate.",
    )

    parser.add_argument(
        "--model-impl",
        type=str,
        default="vllm",
        help="Model implementation backend.",
    )

    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--question",
        type=str,
        default="Describe the image.",
        help="Input question/prompt for the model.",
    )

    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Modality of the input.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length.",
    )

    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of sequences to process in parallel.",
    )

    # parser.add_argument(
    #     "--min-pixels",
    #     type=int,
    #     default=28 * 28,
    #     help="Minimum pixel count for image processing.",
    # )

    # parser.add_argument(
    #     "--max-pixels",
    #     type=int,
    #     default=1280 * 28 * 28,
    #     help="Maximum pixel count for image processing.",
    # )

    # parser.add_argument(
    #     "--fps",
    #     type=int,
    #     default=1,
    #     help="Frames per second for video processing.",
    # )

    # parser.add_argument(
    #     "--limit-mm-per-prompt",
    #     type=int,
    #     default=1,
    #     help="Limit multimodal inputs per prompt.",
    # )

    # parser.add_argument(
    #     "--num-frames",
    #     type=int,
    #     default=16,
    #     help="Number of frames to extract from the video.",
    # )

    args = parser.parse_args()
    main(args)
