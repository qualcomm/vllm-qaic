# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Persistent Qwen3-ASR benchmark for the audio_samples tree.

The LLM/QPC is created once. One warmup request is excluded from all reported
measurements; each remaining file is sent exactly once.
"""

import argparse
import json
import re
import time
from pathlib import Path

import librosa
from vllm import LLM, SamplingParams


def duration_group(path: Path) -> str:
    match = re.search(r"-(40|50|60)s-", path.name)
    return f"{match.group(1)}s" if match else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("qwen3_asr_vllm_benchmark.json"))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    files = sorted(args.audio_root.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"No WAV files found under {args.audio_root}")

    llm = LLM(
        model="Qwen/Qwen3-ASR-0.6B-hf",
        max_num_seqs=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
        additional_config={
            "device_group": [args.device],
            "override_qaic_config": {"prefill_seq_len": 512, "encoder_ctx_len": 3000},
        },
    )
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    warmup_audio = librosa.load(str(files[0]), sr=16000)
    warmup_start = time.perf_counter()
    llm.generate(
        {"prompt": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n", "multi_modal_data": {"audio": warmup_audio}},
        sampling,
    )
    warmup_seconds = time.perf_counter() - warmup_start

    rows = []
    total_start = time.perf_counter()
    for path in files:
        audio = librosa.load(str(path), sr=16000)
        started = time.perf_counter()
        result = llm.generate(
            {"prompt": "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n", "multi_modal_data": {"audio": audio}},
            sampling,
        )[0]
        elapsed = time.perf_counter() - started
        language = path.parent.name
        rows.append({
            "file": str(path),
            "language": language,
            "duration_group": duration_group(path),
            "elapsed_seconds": elapsed,
            "text": result.outputs[0].text,
        })
        print(f"{language} {duration_group(path)} {path.name}: {elapsed:.4f}s")

    report = {
        "model": "Qwen/Qwen3-ASR-0.6B-hf",
        "device_ids": [args.device],
        "warmup_seconds_excluded": warmup_seconds,
        "inference_wall_seconds": time.perf_counter() - total_start,
        "files_inferred_once": len(rows),
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps({"output": str(args.output), "warmup_seconds_excluded": warmup_seconds, "files": len(rows)}))


if __name__ == "__main__":
    main()
