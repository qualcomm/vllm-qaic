# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""One-shot Qwen3-ASR inference on QAIC."""

import argparse
import os
import time

import librosa


PROMPT = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"


def parse_device_ids(value: str) -> list[int]:
    """Parse comma-separated QAIC device IDs."""
    try:
        device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("device IDs must be integers") from exc
    if not device_ids:
        raise argparse.ArgumentTypeError("at least one device ID is required")
    return device_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=parse_device_ids(os.environ.get("QAIC_VISIBLE_DEVICES", "0")),
        help="Comma-separated QAIC device IDs (default: QAIC_VISIBLE_DEVICES or 0).",
    )
    parser.add_argument(
        "--qpc-path",
        default=os.environ.get("VLLM_QAIC_QPC_PATH"),
        help="Compiled QPC directory; defaults to VLLM_QAIC_QPC_PATH.",
    )
    parser.add_argument(
        "--efficient-transformers",
        default=os.environ.get("VLLM_QAIC_EFFICIENT_TRANSFORMERS"),
        help="Optional local QEfficient checkout.",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=512)
    parser.add_argument("--encoder-ctx-len", type=int, default=3000)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()

    if args.qpc_path:
        os.environ["VLLM_QAIC_QPC_PATH"] = args.qpc_path
    if args.efficient_transformers:
        os.environ["VLLM_QAIC_EFFICIENT_TRANSFORMERS"] = args.efficient_transformers

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_num_seqs=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.prefill_seq_len,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
        additional_config={
            "device_group": args.device_ids,
            "override_qaic_config": {
                "prefill_seq_len": args.prefill_seq_len,
                "encoder_ctx_len": args.encoder_ctx_len,
            },
        },
    )
    audio = librosa.load(args.audio_file, sr=16000)
    started = time.perf_counter()
    output = llm.generate(
        {"prompt": PROMPT, "multi_modal_data": {"audio": audio}},
        SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )[0]
    completion = output.outputs[0]
    print(f"text={completion.text!r}")
    print(f"token_ids={list(completion.token_ids)!r}")
    print(f"finish_reason={completion.finish_reason!r}")
    print(f"elapsed_seconds={time.perf_counter() - started:.6f}")


if __name__ == "__main__":
    main()
