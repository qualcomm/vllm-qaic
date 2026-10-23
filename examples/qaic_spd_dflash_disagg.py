# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Launch DFlash using qaic_disagg's existing proxy, workers, and connector.

Requires an installed qaic_disagg package compatible with this vLLM version.
Supports GPT-OSS, Qwen3, and Llama text models with a matching DFlash draft.
Use --prefill-pipeline-parallel-size to enable multi-stage prefill.
Omit QPC paths to export and compile through the native launcher.
Send requests to /v1/chat/completions on the proxy port.
"""

import argparse
import importlib.util
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--draft-model")
    parser.add_argument("--prefill-devices", default="0,1,2,3")
    parser.add_argument("--decode-devices", default="4,5,6,7")
    parser.add_argument("--dlm-devices", default="8,9,10,11")
    parser.add_argument("--prefill-qpc")
    parser.add_argument("--decode-qpc")
    parser.add_argument("--dlm-qpc")
    parser.add_argument("--ctx-len", type=int, default=4096)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--prefill-max-num-seqs", type=int, default=1)
    parser.add_argument("--decode-max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--prefill-pipeline-parallel-size",
        type=int,
        default=1,
        help="Prefill pipeline stages; 1 disables pipelining (default: %(default)s).",
    )
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--aic-enable-depth-first", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-port", type=int, default=8001)
    parser.add_argument("--decode-port", type=int, default=8002)
    args, launcher_args = parser.parse_known_args()
    if args.prefill_max_num_seqs < 1 or args.decode_max_num_seqs < 1:
        parser.error("Prefill and decode --max-num-seqs must be positive.")
    if importlib.util.find_spec("qaic_disagg") is None:
        parser.error(
            "Install a compatible qaic_disagg package in this environment first."
        )

    os.environ.setdefault("TRITON_CPU_BACKEND", "1")
    from qaic_spd_dflash import _dflash_block_size

    draft_model = args.draft_model or {
        "openai/gpt-oss-20b": "z-lab/gpt-oss-20b-DFlash",
        "openai/gpt-oss-120b": "z-lab/gpt-oss-120b-DFlash",
        "Qwen/Qwen3-4B": "z-lab/Qwen3-4B-DFlash-b16",
        "Qwen/Qwen3-8B": "z-lab/Qwen3-8B-DFlash-b16",
        "meta-llama/Llama-3.1-8B-Instruct": (
            "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"
        ),
    }.get(args.model)
    if draft_model is None:
        parser.error("--draft-model is required for this target model.")
    block_size = _dflash_block_size(draft_model)
    if args.prefill_seq_len < block_size or args.prefill_seq_len % block_size:
        parser.error(
            "--prefill-seq-len must be a positive multiple of "
            f"the DFlash block size ({block_size})."
        )
    groups = [
        [int(device) for device in group.split(",")]
        for group in (args.prefill_devices, args.decode_devices, args.dlm_devices)
    ]
    devices = [device for group in groups for device in group]
    if len(set(devices)) != len(devices):
        parser.error("Use separate device groups for prefill, decode, and DLM.")
    if (
        args.prefill_pipeline_parallel_size < 1
        or len(groups[0]) % args.prefill_pipeline_parallel_size
    ):
        parser.error(
            "--prefill-pipeline-parallel-size must be positive and divide "
            "the number of prefill devices."
        )

    common = {
        "num_cores": args.cores,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "aic_enable_depth_first": args.aic_enable_depth_first,
        "mos": 1,
    }
    prefill, decode, draft = (dict(common) for _ in range(3))
    if args.prefill_pipeline_parallel_size > 1:
        prefill["mdp_strategy"] = "intersection"
    for config, qpc in (
        (prefill, args.prefill_qpc),
        (decode, args.decode_qpc),
        (draft, args.dlm_qpc),
    ):
        if qpc:
            config["qpc_path"] = qpc
    draft.update(device_group=groups[2], prefill_seq_len=block_size)
    command = [
        sys.executable,
        "-m",
        "qaic_disagg",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
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
        "--prefill-long-prefill-token-threshold",
        str(args.prefill_seq_len),
        "--max-num-batched-tokens",
        str(
            max(args.prefill_max_num_seqs, args.decode_max_num_seqs)
            * (args.prefill_seq_len + block_size)
        ),
        "--no-async-scheduling",
        "--no-enable-prefix-caching",
        "--quantization",
        "mxfp6",
        "--kv-cache-dtype",
        "mxint8",
        "--prefill-override-qaic-config",
        json.dumps(prefill),
        "--decode-override-qaic-config",
        json.dumps(decode),
        "--additional-config",
        json.dumps({"draft_override_qaic_config": draft}),
        "--decode-speculative-config",
        json.dumps(
            {
                "method": "dflash",
                "model": draft_model,
                "num_speculative_tokens": block_size - 1,
            }
        ),
        "--proxy-workers",
        "1",
        "-vvv",
        *launcher_args,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
