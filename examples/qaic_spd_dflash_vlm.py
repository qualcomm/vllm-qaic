# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import argparse
from contextlib import ExitStack
import os
from pathlib import Path

from PIL import Image
from transformers import AutoConfig, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    parser = argparse.ArgumentParser(
        description="Non-disaggregated DFlash/DFlash2 VLM example."
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3.5-4B-DFlash")
    parser.add_argument(
        "--image",
        default="https://vllm-public-assets.s3.us-west-2.amazonaws.com/stop_sign.jpg",
        help="Local image path or URL.",
    )
    parser.add_argument("--question", default="Describe this image in detail.")
    for component in ("tlm", "dlm", "vision"):
        parser.add_argument(f"--{component}-qpc")
        parser.add_argument(f"--{component}-cores", type=int, default=8)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--enable-thinking", action="store_true", default=False)
    thinking.add_argument(
        "--disable-thinking", dest="enable_thinking", action="store_false"
    )
    args = parser.parse_args(argv)
    if min(args.tlm_cores, args.dlm_cores, args.vision_cores) < 1:
        parser.error("Core counts must be positive.")
    tlm_repo = args.model
    dlm_repo = args.draft_model
    question = args.question
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # QPC parameters. Missing QPCs are exported and compiled automatically.
    ctx_len = 4096
    seq_len = 128
    image_size = 448
    tlm_devices = [0, 1, 2, 3]
    dlm_devices = [4, 5, 6, 7]
    vision_devices = [8, 9, 10, 11]
    target_config = AutoConfig.from_pretrained(tlm_repo)
    draft_config = AutoConfig.from_pretrained(dlm_repo)
    block_size = (
        getattr(draft_config, "block_size", None)
        or draft_config.dflash_config["block_size"]
    )

    processor = AutoProcessor.from_pretrained(tlm_repo)
    processor_kwargs = {"min_pixels": image_size**2, "max_pixels": image_size**2}
    if Path(args.image).is_file():
        with Image.open(args.image) as local_image:
            image = local_image.convert("RGB")
    else:
        image = fetch_image(args.image).convert("RGB")
    image = image.resize((image_size, image_size))
    prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    grid = processor(
        text=prompt, images=image, return_tensors="pt", **processor_kwargs
    )["image_grid_thw"]

    common = dict(
        model=tlm_repo,
        max_num_seqs=1,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        async_scheduling=False,
        enable_prefix_caching=False,
        mm_processor_kwargs=processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 0},
        seed=0,
    )
    compile_config = {
        "prefill_seq_len": seq_len,
        "height": image_size,
        "width": image_size,
        "aic_enable_depth_first": False,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
    }
    target_compile = {
        **compile_config,
        "device_group": tlm_devices,
        "num_cores": args.tlm_cores,
    }
    if target_config.model_type in {"qwen3_5", "qwen3_5_moe"}:
        # One target slot does not need continuous-batching cache indexing.
        target_compile["pretrained_extra_args"] = {"continuous_batching": False}
    vision_compile = {
        **compile_config,
        "device_group": vision_devices,
        "num_cores": args.vision_cores,
    }
    draft_compile = {
        "device_group": dlm_devices,
        "num_cores": args.dlm_cores,
        "prefill_seq_len": block_size,
        "aic_enable_depth_first": False,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
    }
    for options, qpc_path in (
        (target_compile, args.tlm_qpc),
        (draft_compile, args.dlm_qpc),
        (vision_compile, args.vision_qpc),
    ):
        if qpc_path:
            options["qpc_path"] = qpc_path

    with ExitStack() as cleanup:
        vision = LLM(
            **common,
            runner="pooling",
            additional_config={"override_qaic_config": vision_compile},
        )
        cleanup.callback(vision.llm_engine.engine_core.shutdown)
        embeddings = vision.encode(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            pooling_task="embed",
        )[0].outputs.data

        llm = LLM(
            **common,
            quantization="mxfp6",
            kv_cache_dtype="auto",
            enable_mm_embeds=True,
            disable_log_stats=False,
            additional_config={
                "override_qaic_config": target_compile,
                "draft_override_qaic_config": draft_compile,
            },
            speculative_config={
                "method": "dflash",
                "model": dlm_repo,
                "num_speculative_tokens": block_size - 1,
            },
        )
        cleanup.callback(llm.llm_engine.engine_core.shutdown)
        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": {
                            "image_embeds": embeddings,
                            "image_grid_thw": grid,
                        }
                    },
                }
            ],
            sampling_params,
        )
        for output in outputs:
            generated = output.outputs[0]
            print(
                f"Prompt: {question!r}, Generated text: {generated.text!r}, "
                f"Num generated tokens: {len(generated.token_ids)}"
            )


if __name__ == "__main__":
    main()
