# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/offline_inference/vision_language_multi_image.py

"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models for text generation on QAIC.

Each request is composed of N images (controlled by --num-images) and a
single question.
"""

import copy
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from multiprocessing import Event, Process, Queue
from typing import NamedTuple

import requests
import torch
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs, SamplingParams
from vllm.config import KVTransferConfig
from vllm.multimodal.image import convert_image_mode
from vllm.utils.argparse_utils import FlexibleArgumentParser


class ModelRequestData(NamedTuple):
    engine_args: EngineArgs
    prompts: list[str]
    stop_token_ids: list[int] | None = None
    sampling_params: list[SamplingParams] | None = None
    image_grid_thw: torch.Tensor | None = None
    use_disagg_lang: bool = False


seq_len = 128
ctx_len = 4096
# Though on QAIC vision encoders run at batch size = 1,
# vision_bsz can be set higher here to allow vLLM to batch
# pre-process multimodal data before sending to the encoder.
# TODO: Gather performance data on whether this is beneficial or not.
vision_bsz = 1
prefill_bsz = 1
decode_bsz = 4


# InternVL
def run_internvl(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "OpenGVLab/InternVL2_5-1B"

    engine_args = EngineArgs(
        model=model_name,
        trust_remote_code=True,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        mm_processor_kwargs={"max_dynamic_patch": 4},
        limit_mm_per_prompt={"image": num_images},
    )

    placeholders = "\n".join(f"Image-{i}: <image>\n" for i in range(1, num_images + 1))
    messages = [
        [
            {"role": "user", "content": f"{placeholders}\n{question}"}
            for question in questions
        ]
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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


# Qwen2.5-VL
def run_qwen2_5_vl(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Provide lists of height and width that are used to compile the model.
    # If an input resolution does not match a precompiled QPC exactly,
    # it resizes to the best resolution from the supported list.
    height = [354, 512]
    width = [536, 910]

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
        limit_mm_per_prompt={"image": num_images},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
                "use_onnx_subfunctions": False,
                "split_model_io": True,
            },
        },
    )

    # One placeholder token-pair per image in the request.
    placeholders = "<|vision_start|><|image_pad|><|vision_end|>" * num_images

    # image_grid_thw: one dummy row [-1, -1, -1] per image.
    # The actual values are recovered from image_grid_thw_lookup in
    # QaicQwen2VLMultiModalDataParser._parse_image_data.
    image_grid_thw = torch.tensor([[-1, -1, -1]] * num_images)

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
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
def run_qwen3_vl(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-32B-Instruct"

    # Provide lists of height and width that are used to compile the model.
    # If an input resolution does not match a precompiled QPC exactly,
    # it resizes to the best resolution from the supported list.
    height = [354, 512]
    width = [536, 910]

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
        limit_mm_per_prompt={"image": num_images},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
                "use_onnx_subfunctions": False,
                "split_model_io": True,
            },
        },
    )

    # One placeholder token-pair per image in the request.
    placeholders = "<|vision_start|><|image_pad|><|vision_end|>" * num_images

    # image_grid_thw: one dummy row [-1, -1, -1] per image.
    # The actual values are recovered from image_grid_thw_lookup in
    # QaicQwen2VLMultiModalDataParser._parse_image_data.
    image_grid_thw = torch.tensor([[-1, -1, -1]] * num_images)

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        image_grid_thw=image_grid_thw,
    )


# Qwen3-VL-MoE
def run_qwen3_vl_moe(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    # Provide lists of height and width that are used to compile the model.
    # If an input resolution does not match a precompiled QPC exactly,
    # it resizes to the best resolution from the supported list.
    height = [354, 512]
    width = [536, 910]

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
        limit_mm_per_prompt={"image": num_images},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
                "use_onnx_subfunctions": False,
                "split_model_io": True,
            },
        },
    )

    # One placeholder token-pair per image in the request.
    placeholders = "<|vision_start|><|image_pad|><|vision_end|>" * num_images

    # image_grid_thw: one dummy row [-1, -1, -1] per image.
    # The actual values are recovered from image_grid_thw_lookup in
    # QaicQwen2VLMultiModalDataParser._parse_image_data.
    image_grid_thw = torch.tensor([[-1, -1, -1]] * num_images)

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        image_grid_thw=image_grid_thw,
        use_disagg_lang=True,
    )


# Qwen3.5-VL-MoE
def run_qwen3_5_vl_moe(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "Qwen/Qwen3.5-35B-A3B"

    height = [354, 512]
    width = [536, 910]

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
        limit_mm_per_prompt={"image": num_images},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
                "use_onnx_subfunctions": True,
                "split_model_io": True,
            },
        },
    )

    placeholders = "<|vision_start|><|image_pad|><|vision_end|>" * num_images
    image_grid_thw = torch.tensor([[-1, -1, -1]] * num_images)

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        image_grid_thw=image_grid_thw,
        use_disagg_lang=True,
    )


# Qwen3.6-VL-MoE
def run_qwen3_6_vl_moe(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "Qwen/Qwen3.6-35B-A3B"

    height = [354, 512]
    width = [536, 910]

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
        limit_mm_per_prompt={"image": num_images},
        additional_config={
            "override_qaic_config": {
                "height": height,
                "width": width,
                "use_onnx_subfunctions": True,
                "split_model_io": True,
            },
        },
    )

    placeholders = "<|vision_start|><|image_pad|><|vision_end|>" * num_images
    image_grid_thw = torch.tensor([[-1, -1, -1]] * num_images)

    prompts = [
        (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{placeholders}{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        image_grid_thw=image_grid_thw,
        use_disagg_lang=True,
    )


# Gemma4
def run_gemma4(questions: list[str], num_images: int) -> ModelRequestData:
    model_name = "google/gemma-4-E2B"

    engine_args = EngineArgs(
        model=model_name,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": num_images},
        additional_config={
            "override_qaic_config": {
                "use_onnx_subfunctions": False,
                "split_model_io": True,
            }
        },
    )

    # One placeholder token per image in the request.
    placeholders = "<|image|>" * num_images

    prompts = [
        (f"<bos><|turn>user\n{placeholders}{question}<turn|>\n<|turn>model\n")
        for question in questions
    ]

    return ModelRequestData(
        engine_args=engine_args,
        prompts=prompts,
        use_disagg_lang=True,
    )


model_example_map = {
    "qwen2_5_vl": run_qwen2_5_vl,
    "qwen3_vl": run_qwen3_vl,
    "qwen3_vl_moe": run_qwen3_vl_moe,
    "qwen3_5_vl_moe": run_qwen3_5_vl_moe,
    "qwen3_6_vl_moe": run_qwen3_6_vl_moe,
    "internvl_chat": run_internvl,
    "gemma4": run_gemma4,
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


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def parse_args():
    parser = FlexibleArgumentParser(
        description="Demo on using vLLM for offline inference with "
        "multi-image vision language models on QAIC hardware."
    )
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        default="qwen2_5_vl",
        choices=model_example_map.keys(),
        help='Huggingface "model_type".',
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of independent requests to send. Questions and images "
        "are cycled if num-prompts exceeds the number provided.",
    )
    parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=2,
        help="Number of images bundled into each request.",
    )
    parser.add_argument(
        "--questions",
        type=str,
        nargs="+",
        required=True,
        help="One or more questions/prompts to send to the model. "
        "Example: --questions 'What is this?' 'Describe the images.'",
    )
    parser.add_argument(
        "--image-urls",
        type=str,
        nargs="+",
        default=None,
        help="One or more image URLs. "
        "Example: --image-urls 'http://a.com/1.jpg' 'http://a.com/2.jpg'",
    )
    parser.add_argument(
        "--file-paths",
        type=str,
        nargs="+",
        default=None,
        help="One or more local image file paths. "
        "Example: --file-paths img1.jpg img2.jpg",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed when initializing `vllm.LLM`.",
    )
    parser.add_argument(
        "--time-generate",
        action="store_true",
        help="If True, then print the total generate() call time.",
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
    parser.add_argument(
        "--device-group-prefill",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[1],
        help="QAIC device IDs for the disaggregated prefill LLM (MoE 3-QPC "
        "mode only). Example: --device-group-prefill 1,2,3,4 (default: 1)",
    )
    parser.add_argument(
        "--device-group-decode",
        type=lambda device_ids: [int(x) for x in device_ids.split(",")],
        default=[2],
        help="QAIC device IDs for the disaggregated decode LLM (MoE 3-QPC "
        "mode only). Example: --device-group-decode 5,6,7,8 (default: 2)",
    )
    parser.add_argument(
        "--kv-port",
        type=int,
        default=14579,
        help="Port for the QaicConnector KV handoff server (MoE 3-QPC mode "
        "only). Default: 14579",
    )
    return parser.parse_args()


def _run_disagg_vision_language(
    args,
    req_data: ModelRequestData,
    engine_args_vision: dict,
    engine_args_lang: dict,
    vision_inputs: list,
    request_prompts: list[str],
    sampling_params: SamplingParams,
) -> list:
    """Three-stage offline inference for MoE models requiring separate QPCs.

    Stage 1 - Vision encoder (main process, runner=pooling): encodes raw
              images into vision embeddings.
    Stage 2 - Prefill LLM (subprocess, kv_producer): processes the full
              prompt + embeddings and writes KV cache to the handoff server.
    Stage 3 - Decode LLM  (subprocess, kv_consumer): loads KV cache from
              the handoff server and auto-regressively generates tokens.
    """
    # ---- Stage 1: vision encoding (runs in the main process) ----
    llm_vision = LLM(**engine_args_vision)
    vision_outputs = llm_vision.encode(vision_inputs, pooling_task="embed")

    # Build language-model inputs by replacing raw images with embeddings.
    # Each output already contains the concatenated embeddings for all
    # images in that request; no manual stacking is needed.
    lang_inputs = []
    for i, output in enumerate(vision_outputs):
        embed = output.outputs.data
        print(f"Embedding shape: {embed.shape}")

        if req_data.image_grid_thw is None:
            mm_data = embed
        else:
            mm_data = {
                "image_embeds": embed,
                "image_grid_thw": req_data.image_grid_thw,
            }
        lang_inputs.append(
            {
                "prompt": request_prompts[i],
                "multi_modal_data": {"image": mm_data},
            }
        )

    # ---- Start the KV handoff server ----
    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "qaic_disagg.kv_handoff.server",
            "--port",
            str(args.kv_port),
            "--size",
            "64",
        ]
    )
    time.sleep(10)

    # ---- Stage 2: Prefill LLM config (kv_producer) ----
    engine_args_prefill = copy.deepcopy(engine_args_lang)
    engine_args_prefill["additional_config"]["device_group"] = args.device_group_prefill
    engine_args_prefill["additional_config"]["override_qaic_config"].update(
        {
            "prefill_only": True,
            "stages": len(args.device_group_prefill),
            "allow-mxint8-mdp-io": True,
            "kv_offload": True,
            "use_onnx_subfunctions": True,
            "split_model_io": True,
        }
    )
    engine_args_prefill["max_num_seqs"] = prefill_bsz

    # ---- Stage 3: Decode LLM config (kv_consumer) ----
    engine_args_decode = copy.deepcopy(engine_args_lang)
    engine_args_decode["additional_config"]["device_group"] = args.device_group_decode
    engine_args_decode["additional_config"]["override_qaic_config"].update(
        {
            "prefill_only": False,
            "stages": 1,
            "allow-mxint8-mdp-io": True,
            "kv_offload": True,
            "use_onnx_subfunctions": False,
            "split_model_io": True,
        }
    )
    engine_args_decode["max_num_seqs"] = decode_bsz

    sampling_params_prefill = SamplingParams(temperature=0.0, max_tokens=1)
    prefill_done = Event()
    prefill_ready = Event()  # set after prefill QPC is loaded, before generate()
    result_queue: Queue = Queue()

    def _prefill_worker(ea, li, sp, done, ready):
        try:
            ea["kv_transfer_config"] = KVTransferConfig(
                kv_connector="QaicConnector",
                kv_role="kv_producer",
                kv_rank=0,
                kv_port=args.kv_port,
            )
            llm = LLM(**ea)
        except Exception as exc:
            print(exc)
        finally:
            ready.set()
        try:
            llm.generate(li, sampling_params=sp)
        except Exception as exc:
            print(exc)
        finally:
            done.set()

    def _decode_worker(ea, li, sp, done, queue, ready):
        try:
            ea["kv_transfer_config"] = KVTransferConfig(
                kv_connector="QaicConnector",
                kv_role="kv_consumer",
                kv_rank=1,
                kv_port=args.kv_port,
            )
            ready.wait()
            llm = LLM(**ea)
            done.wait()
            outputs = llm.generate(li, sampling_params=sp)
            queue.put(outputs)
        except Exception as exc:
            queue.put(exc)

    p_prefill = Process(
        target=_prefill_worker,
        args=(
            engine_args_prefill,
            lang_inputs,
            sampling_params_prefill,
            prefill_done,
            prefill_ready,
        ),
    )
    p_decode = Process(
        target=_decode_worker,
        args=(
            engine_args_decode,
            lang_inputs,
            sampling_params,
            prefill_done,
            result_queue,
            prefill_ready,
        ),
    )

    p_prefill.start()
    p_decode.start()

    p_prefill.join()
    p_decode.join()
    server_proc.terminate()
    while server_proc.poll() is None:
        time.sleep(1)

    result = result_queue.get()
    if isinstance(result, Exception):
        raise result

    return result


def main(args):
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")

    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1.")
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be >= 1.")

    image_pool = load_images(args)
    if not image_pool:
        raise ValueError("Provide at least one image via --image-urls or --file-paths.")
    if len(image_pool) < args.num_images:
        raise ValueError(
            f"Requested {args.num_images} images per request but only "
            f"{len(image_pool)} image(s) were provided. "
            "Supply more via --image-urls or --file-paths."
        )

    req_data = model_example_map[model](args.questions, args.num_images)

    # Disable other modalities to save memory
    default_limits = {"image": 0, "video": 0, "audio": 0}
    req_data.engine_args.limit_mm_per_prompt = default_limits | dict(
        req_data.engine_args.limit_mm_per_prompt or {}
    )

    engine_args = asdict(req_data.engine_args) | {
        "seed": args.seed,
    }

    engine_args_vision = copy.deepcopy(engine_args)
    engine_args_vision["runner"] = "pooling"
    engine_args_vision["additional_config"]["device_group"] = args.device_group_vision
    engine_args_vision["max_num_seqs"] = vision_bsz
    # Async scheduling is not supported for vision encoder
    engine_args_vision["async_scheduling"] = False

    engine_args["enable_mm_embeds"] = True
    engine_args["additional_config"]["device_group"] = args.device_group_lang
    engine_args["quantization"] = "mxfp6"
    engine_args["kv_cache_dtype"] = "mxint8"
    engine_args["max_num_seqs"] = decode_bsz

    llm_lang = LLM(**engine_args) if not req_data.use_disagg_lang else None
    llm_vision = LLM(**engine_args_vision) if not req_data.use_disagg_lang else None

    # Cycle through the formatted prompts from req_data across requests.
    request_prompts = [
        req_data.prompts[i % len(req_data.prompts)] for i in range(args.num_prompts)
    ]
    # Each request gets num_images images, cycling through the pool.
    request_image_lists: list[list[Image.Image]] = [
        [
            image_pool[(i * args.num_images + j) % len(image_pool)]
            for j in range(args.num_images)
        ]
        for i in range(args.num_prompts)
    ]

    # One encode request per prompt, carrying all images for that request as a
    # list.  vLLM runs the vision encoder once per image and returns a single
    # concatenated embedding tensor for the whole request.
    vision_inputs = [
        {
            "prompt": request_prompts[i],
            "multi_modal_data": {"image": request_image_lists[i]},
        }
        for i in range(args.num_prompts)
    ]

    sampling_params = (
        SamplingParams(
            temperature=0.0,
            max_tokens=200,
            stop_token_ids=req_data.stop_token_ids,
        )
        if req_data.sampling_params is None
        else req_data.sampling_params
    )

    # --- Disaggregated path (3-QPC, for MoE models like Qwen3-VL-MoE) ---
    if req_data.use_disagg_lang:
        with time_counter(args.time_generate):
            outputs = _run_disagg_vision_language(
                args,
                req_data,
                engine_args_vision,
                engine_args,
                vision_inputs,
                request_prompts,
                sampling_params,
            )

    else:
        with time_counter(args.time_generate):
            vision_outputs = llm_vision.encode(vision_inputs, pooling_task="embed")

            # Each output already contains the concatenated embeddings for all
            # images in that request; no manual stacking is needed.
            lang_inputs = []
            for i, output in enumerate(vision_outputs):
                embed = output.outputs.data
                print(f"Embedding shape: {embed.shape}")

                if req_data.image_grid_thw is None:
                    mm_data = embed
                else:
                    # Qwen2.5-VL / Qwen3-VL: pass both the concatenated embeddings
                    # and the per-image grid metadata (one dummy row per image).
                    mm_data = {
                        "image_embeds": embed,
                        "image_grid_thw": req_data.image_grid_thw,
                    }

                lang_inputs.append(
                    {
                        "prompt": request_prompts[i],
                        "multi_modal_data": {"image": mm_data},
                    }
                )

            outputs = llm_lang.generate(lang_inputs, sampling_params=sampling_params)

    # ---- Print results ---------------------------------------------------
    print("-" * 50)
    for o in outputs:
        print(o.outputs[0].text)
        print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
