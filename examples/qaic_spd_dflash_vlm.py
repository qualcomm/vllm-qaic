# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""DFlash VLM inference with split QAIC vision and language engines (BS1)."""

import argparse
import os
import time
from contextlib import ExitStack
from pathlib import Path


MODEL_MAP = {
    "Qwen/Qwen3.5-4B": "z-lab/Qwen3.5-4B-DFlash",
    "Qwen/Qwen3.6-27B": "z-lab/Qwen3.6-27B-DFlash",
    "Qwen/Qwen3.6-35B-A3B": "z-lab/Qwen3.6-35B-A3B-DFlash",
    "google/gemma-4-31B-it": "z-lab/gemma-4-31B-it-DFlash",
}


def device_group(value):
    return [int(device) for device in value.split(",")]


def run(cleanup):
    from PIL import Image, ImageOps
    from transformers import AutoConfig, AutoProcessor
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--draft-model", help="Draft repo or local checkpoint directory."
    )
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--image", help="Local image; omit for a text-only request.")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--tlm-devices", type=device_group, default=[0, 1, 2, 3])
    parser.add_argument("--dlm-devices", type=device_group, default=[4, 5, 6, 7])
    parser.add_argument("--vision-devices", type=device_group, default=[8, 9, 10, 11])
    for component in ("tlm", "dlm", "vision"):
        parser.add_argument(f"--{component}-qpc", help="Precompiled QPC directory.")
    args = parser.parse_args()
    for component in ("tlm", "dlm", "vision"):
        path = getattr(args, f"{component}_qpc")
        if path and not (Path(path) / "programqpc.bin").is_file():
            parser.error(f"--{component}-qpc has no programqpc.bin: {path}")

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if config.model_type not in {"qwen3_5", "qwen3_5_moe", "qwen3_vl", "gemma4"}:
        parser.error("Supported architectures: Qwen3.5, Qwen3.5-MoE, Qwen3-VL, Gemma4.")
    args.draft_model = args.draft_model or MODEL_MAP.get(args.model)
    if not args.draft_model:
        parser.error("Pass --draft-model for this target, including Qwen3-VL.")
    draft = AutoConfig.from_pretrained(args.draft_model, trust_remote_code=True)
    block_size = getattr(draft, "block_size", None)
    if block_size is None:
        block_size = (getattr(draft, "dflash_config", None) or {}).get("block_size")
    if type(block_size) is not int or block_size < 2:
        parser.error("Draft config must provide an integer block_size >= 2.")
    if args.prefill_seq_len < block_size or args.prefill_seq_len % block_size:
        parser.error("--prefill-seq-len must be a positive multiple of block_size.")
    if config.model_type in {"qwen3_5", "qwen3_5_moe"} and block_size & (
        block_size - 1
    ):
        parser.error("Qwen3.5 DFlash requires a power-of-two block_size.")
    if not 0 < args.max_tokens < args.max_model_len:
        parser.error("--max-tokens must be positive and below --max-model-len.")
    if args.prefill_seq_len > args.max_model_len or args.image_size <= 0:
        parser.error("Invalid prefill length or image size.")
    if config.model_type != "gemma4":
        alignment = (
            config.vision_config.patch_size * config.vision_config.spatial_merge_size
        )
        if args.image_size % alignment:
            parser.error(f"--image-size must be a multiple of {alignment}.")

    processor_kwargs = (
        {}
        if config.model_type == "gemma4"
        else {"min_pixels": args.image_size**2, "max_pixels": args.image_size**2}
    )
    common = dict(
        model=args.model,
        trust_remote_code=True,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        long_prefill_token_threshold=args.prefill_seq_len,
        async_scheduling=False,
        enable_prefix_caching=False,
        mm_processor_kwargs=processor_kwargs,
        limit_mm_per_prompt={"image": 1, "video": 0},
        seed=0,
    )

    def compile_config(devices, qpc):
        settings = dict(
            device_group=devices,
            num_cores=8,
            prefill_seq_len=args.prefill_seq_len,
            mxfp6_matmul=True,
            mxint8_kv_cache=False,
            use_onnx_subfunctions=False,
            split_model_io=True,
            mos=1,
        )
        if config.model_type != "gemma4":
            settings.update(height=args.image_size, width=args.image_size)
        if qpc:
            settings["qpc_path"] = qpc
        return settings

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    content = ([{"type": "image"}] if args.image else []) + [
        {"type": "text", "text": args.prompt}
    ]
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    request = {"prompt": prompt}
    vision_time = 0.0
    if args.image:
        with Image.open(args.image) as source:
            image = ImageOps.pad(
                source.convert("RGB"),
                (args.image_size, args.image_size),
                method=Image.Resampling.LANCZOS,
                color="white",
            )
        grid = None
        if config.model_type != "gemma4":
            grid = processor(
                text=prompt, images=image, return_tensors="pt", **processor_kwargs
            )["image_grid_thw"]
        vision = LLM(
            **common,
            runner="pooling",
            additional_config={
                "override_qaic_config": compile_config(
                    args.vision_devices, args.vision_qpc
                )
            },
        )
        cleanup.callback(vision.llm_engine.engine_core.shutdown)
        started = time.perf_counter()
        embeddings = vision.encode(
            [{"prompt": prompt, "multi_modal_data": {"image": image}}],
            pooling_task="embed",
        )[0].outputs.data
        vision_time = time.perf_counter() - started
        request["multi_modal_data"] = {
            "image": embeddings
            if grid is None
            else {"image_embeds": embeddings, "image_grid_thw": grid}
        }

    target_config = compile_config(args.tlm_devices, args.tlm_qpc)
    target_config["pretrained_extra_args"] = {"dtype": "float32"}
    draft_config = dict(
        device_group=args.dlm_devices,
        num_cores=8,
        prefill_seq_len=block_size,
        mxfp6_matmul=True,
        mxint8_kv_cache=False,
        mos=1,
    )
    if args.dlm_qpc:
        draft_config["qpc_path"] = args.dlm_qpc
    target = LLM(
        **common,
        quantization="mxfp6",
        kv_cache_dtype="auto",
        enable_mm_embeds=True,
        additional_config={
            "override_qaic_config": target_config,
            "draft_override_qaic_config": draft_config,
        },
        speculative_config={
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": block_size - 1,
        },
    )
    cleanup.callback(target.llm_engine.engine_core.shutdown)
    started = time.perf_counter()
    result = target.generate(
        [request], SamplingParams(temperature=0, max_tokens=args.max_tokens)
    )[0]
    elapsed = time.perf_counter() - started
    print(result.outputs[0].text)
    print(f"Generated tokens: {len(result.outputs[0].token_ids)}")
    print(f"Vision encode time: {vision_time:.3f} s")
    print(f"Language generation time (includes prefill): {elapsed:.3f} s")


def main():
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with ExitStack() as cleanup:
        run(cleanup)


if __name__ == "__main__":
    main()
