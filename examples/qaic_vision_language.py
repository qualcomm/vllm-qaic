# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/vision_language.py

"""
This example shows how to use vLLM for running offline inference with
the correct prompt format on vision language models for text generation
on QAIC backend with AOT.

For most models, the prompt format should follow corresponding examples
on HuggingFace model repository.
"""

import copy
import random
from contextlib import contextmanager
from dataclasses import asdict
from typing import NamedTuple

import requests
import torch
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None
    sampling_params: list[SamplingParams] | None = None
    image_grid_thw: torch.Tensor | None = None


seq_len = 128
ctx_len = 4096
# Though on QAIC vision encoders run at batch size = 1,
# vision_bsz can be set higher here to allow vLLM to batch
# pre-process multimodal data before sending to the encoder.
# TODO: Gather performance data on whether this is beneficial or not.
vision_bsz = 4
decode_bsz = 4


# Gemma 3
def run_gemma3(questions: list[str], modality: str) -> ModelRequestData:
    assert modality in ("image", "text")

    engine_args = EngineArgs(
        model="google/gemma-3-4b-it",
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
    )

    placeholder = "<start_of_image>" if modality == "image" else ""
    prompts = [
        (
            "<bos><start_of_turn>user\n"
            f"{placeholder}{question}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# InternVL
def run_internvl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality in ("image", "text")

    model_name = "OpenGVLab/InternVL2_5-1B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_dynamic_patch": 12},
        # Default is 12; with the thumbnail, it becomes 13 patches.
    )

    placeholder = "<image>\n" if modality == "image" else ""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    messages = [
        [{"role": "user", "content": f"{placeholder}{question}"}]
        for question in questions
    ]
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B/blob/main/conversation.py
    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
    stop_token_ids = [token_id for token_id in stop_token_ids if token_id is not None]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        stop_token_ids=stop_token_ids,
    )


# LLaVA-1.5
def run_llava(questions: list[str], modality: str) -> ModelRequestData:
    assert modality in ("image", "text")

    engine_args = EngineArgs(
        model="llava-hf/llava-1.5-7b-hf",
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
    )

    placeholder = "<image>\n" if modality == "image" else ""
    prompts = [f"USER: {placeholder}{question}\nASSISTANT:" for question in questions]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
    )


# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality in ("image", "text")

    model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Provide lists of height and width that are used to compile the model.
    # If an input resolution does not match a precompiled QPC exactly,
    # it resizes to the best resolution from the supported list.
    height = [364, 512]
    width = [532, 910]

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
            },
        },
    )

    if modality == "image":
        placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        # image_grid_thw is required by Qwen2.5VL when vision embeddings are passed
        # as input, but its value depends on the post-resize image dimensions which
        # are determined server-side and are difficult for the client to calculate in
        # advance. A dummy placeholder [-1, -1, -1] is used here; the actual values
        # are recovered from image_grid_thw_lookup in
        # QaicQwen2VLMultiModalDataParser._parse_image_data.
        image_grid_thw = torch.tensor([[-1, -1, -1]])
    else:
        placeholder = ""
        image_grid_thw = None

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholder}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        image_grid_thw=image_grid_thw,
    )


# Qwen3-VL
def run_qwen3_vl(questions: list[str], modality: str) -> ModelRequestData:
    assert modality in ("image", "text")

    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    # Provide lists of height and width that are used to compile the model.
    # If an input resolution does not match a precompiled QPC exactly,
    # it resizes to the best resolution from the supported list.
    height = [364, 512]
    width = [532, 910]

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        limit_mm_per_prompt={"image": 1},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
            },
        },
    )

    if modality == "image":
        placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
        # image_grid_thw is required by Qwen3VL when vision embeddings are passed
        # as input, but its value depends on the post-resize image dimensions which
        # are determined server-side and are difficult for the client to calculate in
        # advance. A dummy placeholder [-1, -1, -1] is used here; the actual values
        # are recovered from image_grid_thw_lookup in
        # QaicQwen2VLMultiModalDataParser._parse_image_data.
        image_grid_thw = torch.tensor([[-1, -1, -1]])
    else:
        placeholder = ""
        image_grid_thw = None

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholder}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        image_grid_thw=image_grid_thw,
    )


model_example_map = {
    "gemma3": run_gemma3,
    "internvl_chat": run_internvl,
    "llava": run_llava,
    "qwen2_5_vl": run_qwen2_5_vl,
    "qwen3_vl": run_qwen3_vl,
}


def load_images(args) -> list[Image.Image]:
    """Load one or more images from --image-urls or --file-paths."""
    images = []
    if args.image_urls:
        for url in args.image_urls:
            img = Image.open(requests.get(url, stream=True).raw)
            images.append(convert_image_mode(img, "RGB"))
    if args.file_paths:
        for path in args.file_paths:
            images.append(convert_image_mode(Image.open(path), "RGB"))
    return images


def get_multi_modal_input(args):
    """
    return {
        "data": list of images or None,
        "questions": list of questions,
    }
    """
    if not args.questions:
        raise ValueError("Provide at least one question via --questions.")

    if args.modality == "image":
        images = load_images(args)
        if not images:
            raise ValueError(
                "For image modality, provide at least one image via "
                "--image-urls or --file-paths."
            )
        return {"data": images, "questions": args.questions}

    if args.modality == "text":
        return {"data": None, "questions": args.questions}

    raise ValueError(f"Modality {args.modality} is not supported.")


def apply_image_repeat(
    image_repeat_prob, num_prompts, images: list, prompts: list[str], modality
):
    """Repeats images with provided probability of "image_repeat_prob".
    Used to simulate hit/miss for the MM preprocessor cache.
    """
    assert 0 <= image_repeat_prob <= 1.0
    no_yes = [0, 1]
    probs = [1.0 - image_repeat_prob, image_repeat_prob]

    inputs = []
    inputs_with_empty_media = []
    cur_image = images[0]
    for i in range(num_prompts):
        if image_repeat_prob is not None:
            res = random.choices(no_yes, probs)[0]
            if res == 0:
                # No repeat => Modify one pixel
                cur_image = cur_image.copy()
                new_val = (i // 256 // 256, i // 256, i % 256)
                cur_image.putpixel((0, 0), new_val)

        uuid = "uuid_{}".format(i)
        inputs.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: cur_image},
                "multi_modal_uuids": {modality: uuid},
            }
        )
        inputs_with_empty_media.append(
            {
                "prompt": prompts[i % len(prompts)],
                "multi_modal_data": {modality: None},
                "multi_modal_uuids": {modality: uuid},
            }
        )

    return inputs, inputs_with_empty_media


@contextmanager
def time_counter(enable: bool):
    if enable:
        import time

        start_time = time.time()
        yield
        elapsed_time = time.time() - start_time
        print("-" * 50)
        print("-- generate time = {}".format(elapsed_time))
        print("-" * 50)
    else:
        yield


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "vision language models for text generation"
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="llava",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to run. Questions and images are cycled "
        "if num-prompts exceeds the number provided.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="image",
        choices=["image", "text"],
        help="Modality of the input.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        required=True,
        help="One or more questions/prompts to send to the model. "
        "Example: --questions 'What is this?' 'Describe the image.'",
    )
    parser.add_argument(
        "--image-urls",
        type=str,
        nargs="+",
        default=None,
        help="One or more image URLs (for image modality). "
        "Example: --image-urls 'http://a.com/1.jpg' 'http://a.com/2.jpg'",
    )
    parser.add_argument(
        "--file-paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more local image file paths (for image modality). "
        "Example: --file-paths img1.jpg img2.jpg",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--image-repeat-prob",
        type=float,
        default=None,
        help="Simulates the hit-ratio for multi-modal preprocessor cache (if enabled).",
    )
    parser.add_argument(
        "--disable-mm-processor-cache",
        action="store_true",
        help="If True, disables caching of multi-modal processor.",
    )
    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time.",
    )
    parser.add_argument(
        "--verify-mm-cache-hit-with-uuids",
        action="store_true",
        help="If True, will send all requests in a second batch with empty mm "
        "data to verify cache hits with UUIDs.",
    )
    parser.add_argument(
        "--device-group-vision",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[0],
        help="QAIC device IDs for the vision encoder in kv_offload mode. "
        "Example: --device-group-vision 0,1 (default: 0)",
    )
    parser.add_argument(
        "--device-group-lang",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[1],
        help="QAIC device IDs for the language decoder (pre-fill and decode) "
        "in kv_offload mode. Example: --device-group-lang 2,3 (default: 1)",
    )
    return parser.parse_args()


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    modality = args.modality
    mm_input = get_multi_modal_input(args)
    images = mm_input["data"]  # list of images or None
    questions = mm_input["questions"]

    req_data = model_example_map[model](questions, modality)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
        "mm_processor_cache_gb": 0 if args.disable_mm_processor_cache else 4,
    }

    # Set engine args specific to vision encoder
    engine_args_vision = copy.deepcopy(engine_args)
    engine_args_vision["runner"] = "pooling"
    engine_args_vision["additional_config"]["device_group"] = args.device_group_vision
    engine_args_vision["max_num_seqs"] = vision_bsz
    # Async scheduling is not supported for vision encoder
    engine_args_vision["async_scheduling"] = False

    # Set engine args specific to vllm instance that runs language pre-fill and decode
    engine_args["enable_mm_embeds"] = True
    engine_args["additional_config"]["device_group"] = args.device_group_lang
    engine_args["quantization"] = "mxfp6"
    engine_args["kv_cache_dtype"] = "mxint8"
    engine_args["max_num_seqs"] = decode_bsz

    llm_lang = LLM(**engine_args)
    llm_vision = LLM(**engine_args_vision) if modality == "image" else None

    prompts = req_data.prompts

    sampling_params = (
        SamplingParams(
            temperature=0.0, max_tokens=64, stop_token_ids=req_data.stop_token_ids
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )

    assert args.num_prompts > 0

    # --- Text-only path ---
    if modality == "text":
        text_inputs = [
            {"prompt": prompts[i % len(prompts)]} for i in range(args.num_prompts)
        ]
        with time_counter(args.time_generate):
            outputs = llm_lang.generate(text_inputs, sampling_params=sampling_params)

        print("-" * 50)
        for o in outputs:
            print(o.outputs[0].text)
            print("-" * 50)
        return

    # --- Image path ---
    if args.num_prompts == 1:
        # Single inference
        uuid = "uuid_0"
        inputs = [
            {
                "prompt": prompts[0],
                "multi_modal_data": {modality: images[0]},
                "multi_modal_uuids": {modality: uuid},
            }
        ]
        inputs_with_empty_media = [
            {
                "prompt": prompts[0],
                "multi_modal_data": {modality: None},
                "multi_modal_uuids": {modality: uuid},
            }
        ]
    else:
        # Batch inference
        if args.image_repeat_prob is not None:
            # Repeat images with specified probability of "image_repeat_prob"
            inputs, inputs_with_empty_media = apply_image_repeat(
                args.image_repeat_prob,
                args.num_prompts,
                images,
                prompts,
                modality,
            )
        else:
            # Cycle through images and prompts across requests
            inputs = []
            inputs_with_empty_media = []
            for i in range(args.num_prompts):
                uuid = "uuid_{}".format(i)
                inputs.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: images[i % len(images)]},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )
                inputs_with_empty_media.append(
                    {
                        "prompt": prompts[i % len(prompts)],
                        "multi_modal_data": {modality: None},
                        "multi_modal_uuids": {modality: uuid},
                    }
                )

    with time_counter(args.time_generate):
        embeddings = []
        outputs = llm_vision.encode(inputs, pooling_task="embed")
        for output in outputs:
            embed = output.outputs.data
            print(f"Embedding shape: {embed.shape}")
            embeddings.append(embed)

        for i, input_item in enumerate(inputs):
            # Replace the original image data with the pre-computed vision
            # embeddings so the language model receives features, not raw pixels.
            if req_data.image_grid_thw is None:
                input_item["multi_modal_data"][modality] = embeddings[i]
            else:
                input_item["multi_modal_data"][modality] = {
                    "image_embeds": embeddings[i],
                    "image_grid_thw": req_data.image_grid_thw,
                }

        outputs = llm_lang.generate(inputs, sampling_params=sampling_params)

    print("-" * 50)
    for o in outputs:
        print(o.outputs[0].text)
        print("-" * 50)

    if args.verify_mm_cache_hit_with_uuids:
        try:
            # MM cache is for the vision encoder. If the input image has already
            # been seen, vision embeddings are returned directly from cache
            # without needing to run the encoder on device.
            print(
                "Sending a second batch of requests with empty media and "
                "matching UUIDs."
            )
            with time_counter(args.time_generate):
                outputs = llm_vision.encode(
                    inputs_with_empty_media, pooling_task="embed"
                )
                embeddings = []
                for output in outputs:
                    embed = output.outputs.data
                    print(f"Embedding shape: {embed.shape}")
                    embeddings.append(embed)

                for i, input_item in enumerate(inputs_with_empty_media):
                    # Replace the original image data with the new embeddings
                    # so the language model receives pre-computed vision features.
                    if req_data.image_grid_thw is None:
                        inputs_with_empty_media[i]["multi_modal_data"][modality] = (
                            embeddings[i]
                        )
                    else:
                        inputs_with_empty_media[i]["multi_modal_data"][modality] = {
                            "image_embeds": embeddings[i],
                            "image_grid_thw": req_data.image_grid_thw,
                        }

                outputs = llm_lang.generate(
                    inputs_with_empty_media, sampling_params=sampling_params
                )
            print("-" * 50)
            for o in outputs:
                print(o.outputs[0].text)
                print("-" * 50)
        except Exception as e:
            print(f"Failed to verify cache hits with UUIDs. Error: {e}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
