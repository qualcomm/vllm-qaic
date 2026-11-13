# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from contextlib import ExitStack

from transformers import AutoConfig, AutoProcessor
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image


def main() -> None:
    tlm_repo = "Qwen/Qwen3.5-4B"
    dlm_repo = "z-lab/Qwen3.5-4B-DFlash"
    image_path = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/stop_sign.jpg"
    question = "Describe this image in detail."
    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # QPC parameters. Missing QPCs are exported and compiled automatically.
    ctx_len = 4096
    seq_len = 128
    image_size = 448
    tlm_devices = [0, 1, 2, 3]
    dlm_devices = [4, 5, 6, 7]
    vision_devices = [8, 9, 10, 11]
    draft_config = AutoConfig.from_pretrained(dlm_repo)
    block_size = (
        getattr(draft_config, "block_size", None)
        or draft_config.dflash_config["block_size"]
    )

    processor = AutoProcessor.from_pretrained(tlm_repo)
    processor_kwargs = {"min_pixels": image_size**2, "max_pixels": image_size**2}
    image = fetch_image(image_path).convert("RGB").resize((image_size, image_size))
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
        "num_cores": 8,
        "prefill_seq_len": seq_len,
        "height": image_size,
        "width": image_size,
        "aic_enable_depth_first": False,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "mos": 1,
    }

    with ExitStack() as cleanup:
        vision = LLM(
            **common,
            runner="pooling",
            additional_config={
                "override_qaic_config": {
                    **compile_config,
                    "device_group": vision_devices,
                }
            },
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
                "override_qaic_config": {
                    **compile_config,
                    "device_group": tlm_devices,
                },
                "draft_override_qaic_config": {
                    "device_group": dlm_devices,
                    "num_cores": 8,
                    "prefill_seq_len": block_size,
                    "aic_enable_depth_first": False,
                    "mxfp6_matmul": True,
                    "mxint8_kv_cache": True,
                    "mos": 1,
                },
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
