# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Pretrained QAIC DFlash vs vanilla greedy parity, including 256-token output.

Requires checkpoints, a compiler and eight free QAIC devices. Qwen3-VL also
requires VLLM_QAIC_QWEN3_VL_DRAFT to point to its local draft checkpoint.
The default unquantized comparison deliberately requires exact token equality;
matching compiler settings alone does not guarantee numerical equivalence.
"""

import copy
import os

import pytest
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from vllm import SamplingParams


@pytest.mark.qaic_test_config(num_device_groups=2, device_group_size=4)
@pytest.mark.parametrize(
    "target,draft",
    [
        ("Qwen/Qwen3.5-4B", "z-lab/Qwen3.5-4B-DFlash"),
        ("Qwen/Qwen3.6-35B-A3B", "z-lab/Qwen3.6-35B-A3B-DFlash"),
        ("google/gemma-4-31B-it", "z-lab/gemma-4-31B-it-DFlash"),
        ("google/gemma-4-26B-A4B-it", "z-lab/gemma-4-26B-A4B-it-DFlash"),
        ("Qwen/Qwen3-VL-32B-Instruct", None),
    ],
)
def test_dflash_vlm_matches_vanilla(vllm_runner, device_groups, target, draft):
    if draft is None:
        draft = os.environ.get("VLLM_QAIC_QWEN3_VL_DRAFT")
        if not draft:
            pytest.skip("Set VLLM_QAIC_QWEN3_VL_DRAFT to the local Qwen3-VL draft")

    is_qwen = target.startswith("Qwen/")
    processor = AutoProcessor.from_pretrained(target, trust_remote_code=True)
    processor_kwargs = {"min_pixels": 448**2, "max_pixels": 448**2} if is_qwen else {}
    compile_config = {
        "num_cores": 8,
        "prefill_seq_len": 128,
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "aic_enable_depth_first": False,
        **({"height": 448, "width": 448} if is_qwen else {}),
    }
    common = dict(
        max_num_seqs=1,
        max_model_len=4096,
        async_scheduling=False,
        enable_prefix_caching=False,
        quantization=None,
        kv_cache_dtype="auto",
        seed=0,
        disable_log_stats=False,
        enable_mm_embeds=True,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_kwargs=processor_kwargs,
        long_prefill_token_threshold=128,
    )
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 200, 200), fill="red")
    draw.ellipse((240, 240, 410, 410), fill="blue")
    question = "Describe the shapes and colors, then explain how to draw them. "
    question += "Give a detailed step-by-step explanation with concrete examples. " * 24
    requests = []
    for with_image in (False, True):
        content = ([{"type": "image"}] if with_image else []) + [
            {"type": "text", "text": question}
        ]
        requests.append(
            {
                "prompt": processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            }
        )
    # Encode once and give both language runs identical image embeddings.
    with vllm_runner(
        target,
        runner="pooling",
        **common,
        additional_config={
            "override_qaic_config": {
                **compile_config,
                "device_group": device_groups[0],
            }
        },
    ) as vision:
        embeddings = vision.llm.encode(
            [
                {
                    "prompt": requests[1]["prompt"],
                    "multi_modal_data": {"image": image},
                }
            ],
            pooling_task="embed",
        )[0].outputs.data
    if is_qwen:
        grid = processor(
            text=requests[1]["prompt"],
            images=image,
            return_tensors="pt",
            **processor_kwargs,
        )["image_grid_thw"]
        embeddings = {"image_embeds": embeddings, "image_grid_thw": grid}
    requests[1]["multi_modal_data"] = {"image": embeddings}

    results = []
    for speculative in (False, True):
        additional = {
            "override_qaic_config": {
                **compile_config,
                "device_group": device_groups[0],
            }
        }
        if speculative:
            additional["draft_override_qaic_config"] = {
                "device_group": device_groups[1],
                "num_cores": 8,
                "prefill_seq_len": 16,
                "mxfp6_matmul": False,
                "mxint8_kv_cache": False,
                "aic_enable_depth_first": False,
            }
        with vllm_runner(
            target,
            **common,
            additional_config=additional,
            speculative_config={
                "method": "dflash",
                "model": draft,
                "num_speculative_tokens": 15,
            }
            if speculative
            else None,
        ) as runner:
            runs = []
            for length in (32, 256):
                for prompts in (requests, requests[::-1]):
                    output = runner.llm.generate(
                        copy.deepcopy(prompts),
                        SamplingParams(
                            temperature=0.0,
                            max_tokens=length,
                            ignore_eos=True,
                        ),
                    )
                    assert len(output) == len(prompts)
                    ids = [list(item.outputs[0].token_ids) for item in output]
                    assert all(len(tokens) == length for tokens in ids)
                    assert all(len(item.prompt_token_ids) > 128 for item in output)
                    runs.append(ids)
            if speculative:
                for name in (
                    "vllm:spec_decode_num_draft_tokens",
                    "vllm:spec_decode_num_accepted_tokens",
                ):
                    count = sum(
                        metric.value
                        for metric in runner.llm.get_metrics()
                        if metric.name == name
                    )
                    assert count > 0, f"DFlash did not exercise {name}"
            results.append(runs)
    for run, (base, spec) in enumerate(zip(*results, strict=True)):
        for request, (expected, actual) in enumerate(zip(base, spec, strict=True)):
            assert actual == expected, (
                f"{target}: run {run}, request {request} diverged"
            )
    for runs in results:
        assert runs[0] == runs[1][::-1]
        assert runs[2] == runs[3][::-1]
