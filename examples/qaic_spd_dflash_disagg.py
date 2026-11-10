# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Non-streaming DFlash disaggregation for Qwen3 and Llama text models.

Requires qaic_disagg with speculative decoding support. Prompt features travel
with the completed target KV handoff; DLM prefill runs on the decode side.
Omit QPC paths to use QEfficient's compile cache or compile missing configurations.
Additional arguments are forwarded to the existing qaic_disagg launcher.
"""

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="z-lab/Qwen3-8B-DFlash-b16")
    parser.add_argument("--prefill-devices", default="0,1,2,3")
    parser.add_argument("--decode-devices", default="4,5,6,7")
    parser.add_argument("--dlm-devices", default="8,9,10,11")
    parser.add_argument("--prefill-qpc")
    parser.add_argument("--decode-qpc")
    parser.add_argument("--dlm-qpc")
    parser.add_argument("--prefill-port", type=int, default=8001)
    parser.add_argument("--decode-port", type=int, default=8002)
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--prefill-max-num-seqs", type=int, default=1)
    parser.add_argument("--decode-max-num-seqs", type=int, default=1)
    parser.add_argument("--prefill-pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument(
        "--mxfp6-matmul", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--mxint8-kv-cache", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    args, launcher_args = parser.parse_known_args()

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.draft_model, trust_remote_code=True)
    block_size = (
        getattr(config, "block_size", None) or config.dflash_config["block_size"]
    )
    if args.prefill_seq_len < block_size or args.prefill_seq_len % block_size:
        parser.error(
            "--prefill-seq-len must be a positive multiple of the draft block size."
        )
    if args.decode_qpc and not args.dlm_qpc:
        parser.error("Pass --dlm-qpc with --decode-qpc, or omit both to compile.")
    groups = [
        [int(d) for d in group.split(",")]
        for group in (args.prefill_devices, args.decode_devices, args.dlm_devices)
    ]
    devices = [device for group in groups for device in group]
    if len(set(devices)) != len(devices):
        parser.error("Prefill, decode, and DLM device groups must not overlap.")

    common = {
        "num_cores": args.cores,
        "mxfp6_matmul": args.mxfp6_matmul,
        "mxint8_kv_cache": args.mxint8_kv_cache,
        "aic_enable_depth_first": args.aic_enable_depth_first,
        "mos": 1,
    }
    prefill, decode, draft = (dict(common) for _ in range(3))
    for cfg in (prefill, decode):
        cfg["prefill_seq_len"] = args.prefill_seq_len
    if args.prefill_pipeline_parallel_size > 1:
        prefill["mdp_strategy"] = "intersection"
    for cfg, path in (
        (prefill, args.prefill_qpc),
        (decode, args.decode_qpc),
        (draft, args.dlm_qpc),
    ):
        if path:
            cfg["qpc_path"] = path
    draft.update(device_group=groups[2], prefill_seq_len=block_size)
    speculative = {
        "method": "dflash",
        "model": args.draft_model,
        "num_speculative_tokens": block_size - 1,
    }
    command = [
        sys.executable,
        "-m",
        "qaic_disagg",
        "--model",
        args.model,
        "--prefill-port",
        str(args.prefill_port),
        "--decode-port",
        str(args.decode_port),
        "--prefill-device-group",
        args.prefill_devices,
        "--decode-device-group",
        args.decode_devices,
        "--prefill-max-num-seqs",
        str(args.prefill_max_num_seqs),
        "--decode-max-num-seqs",
        str(args.decode_max_num_seqs),
        "--prefill-pipeline-parallel-size",
        str(args.prefill_pipeline_parallel_size),
        "--max-model-len",
        str(args.ctx_len),
        "--prefill-max-num-batched-tokens",
        str(args.prefill_max_num_seqs * (args.prefill_seq_len + block_size)),
        "--max-num-batched-tokens",
        str(
            max(args.prefill_max_num_seqs, args.decode_max_num_seqs)
            * (args.prefill_seq_len + block_size)
        ),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        *(["--quantization", "mxfp6"] if args.mxfp6_matmul else []),
        "--kv-cache-dtype",
        "mxint8" if args.mxint8_kv_cache else "auto",
        "--prefill-override-qaic-config",
        json.dumps(prefill),
        "--decode-override-qaic-config",
        json.dumps(decode),
        # Prefill exports features; only decode loads the draft on its own devices.
        "--speculative-config",
        json.dumps(speculative),
        "--decode-speculative-config",
        json.dumps({**speculative, "draft_override_qaic_config": draft}),
        *launcher_args,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
