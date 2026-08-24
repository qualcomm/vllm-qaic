# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared multimodal test helpers"""

import random
from itertools import product

import pytest
import requests
import torch
from datasets import load_dataset
from PIL import Image
from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from transformers import AutoConfig, AutoProcessor

from vllm.model_executor.models.internvl import InternVLProcessor

QWEN_IMAGE_HEIGHT = 140
QWEN_IMAGE_WIDTH = 140
# mm_processor_kwargs for Qwen VL models: used both in the vision LLM and QEfficient
QWEN_PROCESSOR_KWARGS = {"min_pixels": 4 * 28 * 28, "max_pixels": 1280 * 28 * 28}
INTERNVL_NUM_PATCHES = 7
GRANITE_IMAGE_HEIGHT = 1109
GRANITE_IMAGE_WIDTH = 1610
NUM_CORES = 16
MAX_END_TOKENS = 40


def is_internvl(model_name: str) -> bool:
    return "InternVL" in model_name


def is_llama4(model_name: str) -> bool:
    return model_name == "meta-llama/Llama-4-Scout-17B-16E-Instruct"


def is_qwenvl(model_name: str) -> bool:
    return "Qwen" in model_name


def is_gemma(model_name: str) -> bool:
    return "gemma" in model_name


def is_granite(model_name: str) -> bool:
    return "granite" in model_name


def update_qaic_config(model_name: str, base_cfg: dict | None, **updates) -> dict:
    cfg = {} if base_cfg is None else dict(base_cfg)
    cfg.update({k: v for k, v in updates.items() if v is not None})
    if is_qwenvl(model_name):
        cfg["height"] = QWEN_IMAGE_HEIGHT
        cfg["width"] = QWEN_IMAGE_WIDTH
    return cfg


def create_qeff_model_and_processor(
    model_name: str,
    device_group: list,
    seq_len: int,
    ctx_len: int,
    dtype: str,
    kv_dtype: str,
    kv_offload: bool,
    override_cfg: dict,
    tokenizer_obj,
):
    compile_args = dict(
        num_devices=len(device_group),
        num_cores=NUM_CORES,
        prefill_seq_len=seq_len,
        ctx_len=ctx_len,
        mxfp6_matmul=dtype == "mxfp6",
        mxint8_kv_cache=kv_dtype == "mxint8",
        aic_enable_depth_first="dfs" not in override_cfg,
    )
    if is_internvl(model_name):
        hf_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        max_dynamic_patch = INTERNVL_NUM_PATCHES - 1
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(
            model_name,
            kv_offload=kv_offload,
            trust_remote_code=True,
            config=hf_cfg,
        )
        compile_args["num_patches"] = INTERNVL_NUM_PATCHES
        processor = InternVLProcessor(
            hf_cfg, tokenizer_obj, max_dynamic_patch=max_dynamic_patch
        )
    else:
        qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name, kv_offload=kv_offload
        )
        processor = AutoProcessor.from_pretrained(model_name)
        if is_qwenvl(model_name):
            compile_args["height"] = QWEN_IMAGE_HEIGHT
            compile_args["width"] = QWEN_IMAGE_WIDTH
            compile_args["mm_processor_kwargs"] = QWEN_PROCESSOR_KWARGS
    qeff_model.compile(**compile_args)
    return qeff_model, processor


def build_model_input(
    model_name: str, mm_input, tokenizer_obj, is_system_prompt=False, num_images=1
):
    updated_input = []
    for data, question in mm_input:
        if is_internvl(model_name):
            image_tokens = "<image>\n" * num_images
            if is_system_prompt:
                messages = [{"role": "system", "content": question}]
                if image_tokens:
                    messages.append({"role": "user", "content": image_tokens})
            else:
                messages = [{"role": "user", "content": f"{image_tokens}{question}"}]
            prompt = tokenizer_obj.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            image_items = [{"type": "image"}] * num_images
            if is_system_prompt:
                if is_granite(model_name):
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": question}],
                        }
                    ]
                else:
                    messages = [{"role": "system", "content": question}]
                if image_items:
                    messages.append({"role": "user", "content": image_items})
            else:
                messages = [
                    {
                        "role": "user",
                        "content": image_items + [{"type": "text", "text": question}],
                    }
                ]
            prompt = tokenizer_obj.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        updated_input.append((data, prompt))
    return updated_input


def prepare_qeff_inputs(model_name, processor, image, prompt, tokenizer_obj):
    processor_kwargs = dict(
        images=image,
        text=prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    if is_qwenvl(model_name):
        processor_kwargs.update(QWEN_PROCESSOR_KWARGS)
    if is_internvl(model_name):
        processor_kwargs.pop("add_special_tokens")

    inputs = processor(**processor_kwargs)

    if is_internvl(model_name):
        pv = inputs.pop("pixel_values_flat")
        if pv.shape[0] < INTERNVL_NUM_PATCHES:
            pad_size = INTERNVL_NUM_PATCHES - pv.shape[0]
            padding = torch.zeros(pad_size, *pv.shape[1:], dtype=pv.dtype)
            pv = torch.cat([pv, padding], 0)
        inputs["pixel_values"] = pv
    elif is_llama4(model_name):
        max_patches = 17
        pv = inputs["pixel_values"]
        if pv.shape[0] < max_patches:
            pad_size = max_patches - pv.shape[0]
            padding = pv[0].unsqueeze(0).expand(pad_size, *[-1] * pv[0].ndim)
            pv = torch.cat([pv, padding], 0)
            inputs["pixel_values"] = pv
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    return inputs


def encode_if_mm(qllm_vision, inputs, model_name):
    gen_inputs = []
    for item in inputs:
        if (
            "multi_modal_data" in item
            and item["multi_modal_data"].get("image") is not None
        ):
            op = qllm_vision.llm.encode(item, pooling_task="embed")[0]
            embed = op.outputs.data
            prompt = op.prompt_token_ids if is_llama4(model_name) else item["prompt"]
            if is_qwenvl(model_name):
                # Qwen VL requires image_grid_thw alongside the embeddings;
                # a dummy [-1, -1, -1] placeholder is used here — the actual
                # values are recovered server-side from image_grid_thw_lookup.
                raw_images = item["multi_modal_data"]["image"]
                num_images = len(raw_images) if isinstance(raw_images, list) else 1
                embed = {
                    "image_embeds": embed,
                    "image_grid_thw": torch.tensor([[-1, -1, -1]] * num_images),
                }
            gen_inputs.append({"prompt": prompt, "multi_modal_data": {"image": embed}})
        else:
            gen_inputs.append(item)
    return gen_inputs


@pytest.fixture
def mm_input(model_name: str):
    urls = [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg",
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png",
        "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
    ]
    data = [Image.open(requests.get(url, stream=True).raw) for url in urls]
    if is_qwenvl(model_name):
        data = [image.resize((QWEN_IMAGE_WIDTH, QWEN_IMAGE_HEIGHT)) for image in data]
    if is_granite(model_name):
        data = [
            image.resize((GRANITE_IMAGE_WIDTH, GRANITE_IMAGE_HEIGHT)) for image in data
        ]

    prompts = [
        "What's in this image?",
        "What is the content of this image",
        "Describe this image in detail please.",
    ]

    inputs = list(product(data, prompts))
    random.shuffle(inputs)
    return inputs


@pytest.fixture
def text_input():
    prompts = [
        "My name is",
        "How many people died in World War II",
        "Hello ",
        "When it snowfalls in San Diego",
        "What is deep learning?",
        "Tell me a very long story",
    ]
    random.shuffle(prompts)
    return prompts


@pytest.fixture
def audio_data():
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    inputs = []
    for i in range(10):
        data = ds[i]["audio"]["array"]
        # reshape to so shape corresponds to data with batch size 1
        data = data.reshape(-1)
        sample_rate = ds[i]["audio"]["sampling_rate"]
        inputs.append((data, sample_rate))
    random.shuffle(inputs)
    return inputs
