# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/qaic_spd.py — DFlash (Diffusion-LM draft) SpD.

import argparse
import gc
import json
import random

from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams


def _dflash_block_size(dlm_repo: str) -> int:
    """DLM block_size (num_speculative_tokens + 1) read from the checkpoint config."""
    cfg_path = hf_hub_download(repo_id=dlm_repo, filename="config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    bs = cfg.get("block_size") or cfg.get("dflash_config", {}).get("block_size")
    if bs is None:
        raise ValueError(f"no block_size in {dlm_repo}/config.json")
    return int(bs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the QAIC DFlash SpD example.")
    parser.add_argument(
        "--aic-enable-depth-first",
        "--aic_enable_depth_first",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable depth-first compilation (default: disabled for DFlash).",
    )
    parser.add_argument("--model", "--model_name", default="Qwen/Qwen3-4B")
    parser.add_argument("--draft-model", "--draft_model")
    parser.add_argument("--prompt")
    parser.add_argument("--ctx-len", "--ctx_len", type=int, default=4096)
    parser.add_argument("--prefill-seq-len", "--prefill_seq_len", type=int, default=128)
    parser.add_argument("--max-num-seqs", "--max_num_seqs", type=int, default=4)
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=200)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--tlm-qpc", "--tlm_qpc")
    parser.add_argument("--dlm-qpc", "--dlm_qpc")
    parser.add_argument(
        "--tlm-devices",
        "--tlm_devices",
        type=lambda value: [int(device) for device in value.split(",")],
        default=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--dlm-devices",
        "--dlm_devices",
        type=lambda value: [int(device) for device in value.split(",")],
        default=[0, 1, 2, 3],
    )
    args = parser.parse_args()

    prompts = [
        "The history of artificial intelligence dates back",
        "Speculative decoding accelerates inference by",
        "Large language models are trained on",
        "The key difference between supervised and unsupervised learning is",
    ] * 5
    if args.prompt is not None:
        prompts = [args.prompt]
    random.shuffle(prompts)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    # QPC parameters
    ctx_len = args.ctx_len
    seq_len = args.prefill_seq_len
    decode_bsz = args.max_num_seqs

    tlm_repo = args.model
    dlm_repo = args.draft_model or {
        "Qwen/Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
        "openai/gpt-oss-20b": "z-lab/gpt-oss-20b-DFlash",
        "openai/gpt-oss-120b": "z-lab/gpt-oss-120b-DFlash",
    }.get(tlm_repo)
    if dlm_repo is None:
        parser.error("--draft-model is required for this target model")
    # block_size is a property of the DLM checkpoint; the public API takes
    # num_speculative_tokens = block_size - 1 (bonus token lives in slot 0),
    # matching every other vLLM spec-decode backend.
    block_size = _dflash_block_size(dlm_repo)

    print(
        f"DFlash SpD run (TLM={tlm_repo} -> DLM={dlm_repo}, "
        f"block_size={block_size})...\n"
    )

    llm = LLM(
        model=tlm_repo,
        max_num_seqs=decode_bsz,
        max_model_len=ctx_len,
        long_prefill_token_threshold=seq_len,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        disable_log_stats=False,  # native spec-decode acceptance metrics
        gpu_memory_utilization=1.0,
        async_scheduling=False,
        trust_remote_code=True,
        additional_config={
            "override_qaic_config": {
                "device_group": args.tlm_devices,
                "num_cores": args.cores,
                "qpc_path": args.tlm_qpc,
                "prefill_seq_len": seq_len,
                "aic_enable_depth_first": args.aic_enable_depth_first,
                "mxfp6_matmul": True,
                "mxint8_kv_cache": True,
                "mos": 1,
            },
            "draft_override_qaic_config": {
                "device_group": args.dlm_devices,
                "num_cores": args.cores,
                "qpc_path": args.dlm_qpc,
                "prefill_seq_len": block_size,  # DLM prefills block_size at a time
                "aic_enable_depth_first": args.aic_enable_depth_first,
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

    if tlm_repo in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_generated_tokens = len(output.outputs[0].token_ids)
        print(
            f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
            f"Num generated tokens: {num_generated_tokens!r}"
        )

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
