# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/qaic_spd.py — DFlash (Diffusion-LM draft) SpD.

import gc
import json
import random

from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams


def _dflash_block_size(dlm_repo: str) -> int:
    """DLM block_size (== num_speculative_tokens) read from the checkpoint config."""
    cfg_path = hf_hub_download(repo_id=dlm_repo, filename="config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    bs = cfg.get("block_size") or cfg.get("dflash_config", {}).get("block_size")
    if bs is None:
        raise ValueError(f"no block_size in {dlm_repo}/config.json")
    return int(bs)


def main() -> None:
    prompts = [
        "The history of artificial intelligence dates back",
        "Speculative decoding accelerates inference by",
        "Large language models are trained on",
        "The key difference between supervised and unsupervised learning is",
    ] * 5
    random.shuffle(prompts)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

    # QPC parameters
    ctx_len = 4096
    seq_len = 128  # TLM prefill_seq_len (multiple of block_size)
    decode_bsz = 4

    device_group = [0, 1, 2, 3]

    tlm_repo = "Qwen/Qwen3-4B"
    dlm_repo = "z-lab/Qwen3-4B-DFlash-b16"
    # block_size (== num_speculative_tokens) is a property of the DLM checkpoint.
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
                "device_group": device_group,
                "num_cores": 8,
                "prefill_seq_len": seq_len,
                "mxfp6_matmul": True,
                "mxint8_kv_cache": True,
                "mos": 1,
            },
            "draft_override_qaic_config": {
                "device_group": device_group,
                "num_cores": 8,
                "prefill_seq_len": block_size,  # DLM prefills block_size at a time
                "mxfp6_matmul": True,
                "mxint8_kv_cache": True,
                "mos": 1,
            },
        },
        speculative_config={
            "method": "dflash",
            "model": dlm_repo,
            "num_speculative_tokens": block_size,
        },
    )

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
