# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import argparse
import contextlib
import gc
import os

import model_configs_vlm
import torch
from PIL import Image

from vllm import LLM, SamplingParams, platforms
from vllm.distributed import destroy_model_parallel

assert platforms.current_platform.device_type == "qaic", (
    "vLLM could not detect qaic plugin"
)


def delete_hf_checkpoint(model_name: str) -> None:
    """Delete the cached HF checkpoint for model_name to free disk between runs.

    Uses the cache configured by HF_HOME / HF_HUB_CACHE. Never raises, since it
    runs during teardown and must not mask a failure from the run itself.

    Skips a model_name that is a local path. Otherwise deletes unconditionally --
    the flag is opt-in, so passing it is the authorisation to evict, and the sweep
    has to bound disk use whether or not the model ran successfully. Ownership is
    deliberately not consulted: a shared cache is group-writable, so the owner of
    a repo dir is just whoever downloaded it first and says nothing about who
    needs it.
    """
    if os.path.isdir(model_name):
        print(f"[cleanup] '{model_name}' is a local path - not deleting")
        return

    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        revisions = []
        for repo in cache_info.repos:
            if repo.repo_type != "model" or repo.repo_id != model_name:
                continue
            revisions.extend(rev.commit_hash for rev in repo.revisions)

        if not revisions:
            print(f"[cleanup] no deletable cache entry for '{model_name}'")
            return

        strategy = cache_info.delete_revisions(*revisions)
        freed = strategy.expected_freed_size_str
        strategy.execute()
        print(f"[cleanup] deleted checkpoint '{model_name}' (freed {freed})")

    except Exception as exc:  # noqa: BLE001 - teardown must not mask the run
        print(f"[cleanup] could not delete '{model_name}': {exc}")


def cleanup(model_name=None, delete_checkpoint=False):
    try:
        destroy_model_parallel()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.qaic.empty_cache()
    except Exception as exc:  # noqa: BLE001 - teardown is not the priority here
        print(f"[cleanup] distributed teardown skipped: {exc}")
    if delete_checkpoint and model_name:
        delete_hf_checkpoint(model_name)


def test_vlm_vllm(
    model_name: str,
    tp_size: int,
    gen_len: int,
    model_impl="vllm",
    delete_hf_checkpoint=False,
):
    """Run one VLM, then tear down and optionally evict its checkpoint.

    The teardown wraps the whole body rather than LLM() onwards, because the hub
    is already touched above it (MistralTokenizer.from_pretrained), and a failure
    there still has to evict the checkpoint that was just downloaded.
    """
    # Un-escape '--' -> '/' as run_llms.py does: a no-op for scraped names, and
    # only matters for a hand-typed --model.
    model_name = model_name.replace("--", "/")
    try:
        _run_vlm(model_name, tp_size, gen_len, model_impl)
    finally:
        cleanup(model_name=model_name, delete_checkpoint=delete_hf_checkpoint)


def _run_vlm(model_name: str, tp_size: int, gen_len: int, model_impl="vllm"):
    cfg = model_configs_vlm.get_config(model_name)
    print(f"[DEBUG] Loaded config: {cfg}")
    qcclEnabled = os.getenv("QAIC_FORCE_PLATFORM_QCCL", 0)
    # Garbage collect so that RAM is freed for RAM limited host.
    MAX_MODEL_LEN = cfg.get("max_model_len", 4096)
    # KV_CACHE_SIZE = 1024 * 1024 * 1024 * 2  # 2GB KV cache memory
    TRUST_REMOTE_CODE = True
    hf_overrides = cfg.get("hf_overrides", None)
    mm_limit = cfg.get(
        "limit_mm_per_prompt",
        {
            # "image": {"count": 16, "width": 512, "height": 512},
            "image": {"count": 1, "width": 160, "height": 160},
            "video": {"count": 0, "num_frames": 32, "width": 640, "height": 640},
        },
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(current_dir, "Cloud_AI_100.jpeg")
    img = Image.open(img_path)

    prompt_token_ids = None
    if "Qwen" in model_name:
        prompt = (
            "USER: <|vision_start|><|image_pad|><|vision_end|>\n"
            "Describe the image in detail.\n"
            "ASSISTANT:"
        )
        # mm_processor_args = {
        #     "max_pixels": 100
        #     * 28
        #     * 28,  # max visual tokens per image each visual token is 28*28.,
        # }
    elif "InternVL" in model_name:
        prompt = "USER: <image>\nDescribe the image in detail.\nASSISTANT:"
        TRUST_REMOTE_CODE = True
    elif cfg.get("mistral_tokenize", False):
        from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from vllm.tokenizers.mistral import MistralTokenizer

        tokenizer = MistralTokenizer.from_pretrained(model_name)
        request = ChatCompletionRequest(
            messages=[
                UserMessage(
                    content=[
                        ImageChunk(image=img),
                        TextChunk(text="Describe the image in detail."),
                    ]
                )
            ]
        )
        result = tokenizer.mistral.encode_chat_completion(request)
        prompt_token_ids = result.tokens
        prompt = None
    else:
        prompt = "<image>\nDescribe the image in detail."
    prompt = cfg.get("prompt", prompt)
    effective_tp = model_configs_vlm.get_tp_size(model_name, tp_size)
    print(
        f"Model:{model_name}, TP_SIZE:{effective_tp} "
        # f"KV_CACHE_SIZE (MB):{KV_CACHE_SIZE / (1024 * 1024)} "
        f"MAX_MODEL_LEN:{MAX_MODEL_LEN} "
        f"QCCL:{qcclEnabled} "
        f"Prompt: {prompt}\n"
    )
    llm = None
    results = []
    try:
        llm = LLM(
            model=model_name,
            dtype=cfg.get("dtype", "float16"),
            skip_mm_profiling=cfg.get("skip_mm_profiling", False),
            tensor_parallel_size=effective_tp,
            max_model_len=MAX_MODEL_LEN,
            # quantization="fp8",
            enforce_eager=True,
            enable_flashinfer_autotune=False,
            # kv_cache_memory_bytes=KV_CACHE_SIZE,
            enable_prefix_caching=False,
            # pipeline_parallel_size=2,
            trust_remote_code=TRUST_REMOTE_CODE,
            # mm_processor_args = cfg.get("mm_processor_kwargs", mm_processor_args),
            hf_overrides=hf_overrides,
            limit_mm_per_prompt=mm_limit,
            async_scheduling=False,
            model_impl=model_impl,
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.98),
            **({"max_num_seqs": cfg["max_num_seqs"]} if "max_num_seqs" in cfg else {}),
        )

        samplingParam = SamplingParams(
            temperature=0.0,
            # min_tokens=gen_len,
            max_tokens=gen_len,
        )
        results = llm.generate(
            {
                "prompt_token_ids": prompt_token_ids,
                "multi_modal_data": {"image": [img]},
            }
            if prompt_token_ids is not None
            else {
                "prompt": prompt,
                "multi_modal_data": {"image": [img]},
            },
            sampling_params=samplingParam,
        )
    finally:
        # Free device memory before the wrapper's finally tears down and evicts.
        del llm
        gc.collect()

    for r in results:
        print(f"Prompt Token Id Shape: {len(r.prompt_token_ids)}")
        print(r.outputs[0].text)
        print(f"Output Token shape {len(r.outputs[0].token_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--gen-len", type=int, default=20)
    parser.add_argument("--model-impl", type=str, required=True)
    parser.add_argument(
        "--delete-hf-checkpoint",
        action="store_true",
        help="Delete the model's cached HF checkpoint once the run is done",
    )

    args = parser.parse_args()
    test_vlm_vllm(**(args.__dict__))
