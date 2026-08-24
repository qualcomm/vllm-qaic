# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import json
import os
import subprocess
import sys

import pytest


def _run_streaming(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run cmd, streaming its combined stdout/stderr to this process's
    stdout line-by-line as it's produced (unlike subprocess.run(capture_output=True),
    which buffers everything until the child exits)."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines = []
    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()
        lines.append(line)
    process.wait()
    return subprocess.CompletedProcess(cmd, process.returncode, stdout="".join(lines))


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=2048,
    dtype="auto",
    kv_dtype="mxint8",
    dataset="sharegpt",
)
def test_offline_throughput(
    model_name,
    seq_len,
    ctx_len,
    decode_bsz,
    dtype,
    kv_dtype,
    device_group,
    override_qaic_config,
    sharegpt_dataset_path,
):
    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "throughput",
        "--model",
        model_name,
        "--dataset",
        sharegpt_dataset_path,
        "--input-len",
        str(seq_len),
        "--max-model-len",
        str(ctx_len),
        "--num-prompts",
        "10",
        "--seed",
        "1",
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        kv_dtype,
        "--backend",
        "vllm",
        "--max-num-seqs",
        str(decode_bsz),
        "--quantization",
        "mxfp6",
        "--long-prefill-token-threshold",
        str(seq_len),
        "--no-async-scheduling",
    ]
    if os.environ.get("DISABLE_PREFIX_CACHING") == "1":
        cmd.append("--no-enable-prefix-caching")
    _additional_config: dict = {"device_group": device_group}
    if override_qaic_config is not None:
        _additional_config["override_qaic_config"] = override_qaic_config
    cmd += ["--additional-config", json.dumps(_additional_config)]
    # Run offline throughput
    result = _run_streaming(cmd)

    assert result.returncode == 0
    assert "Throughput" in result.stdout


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="auto",
    kv_dtype="auto",
)
def test_offline_latency(
    model_name,
    seq_len,
    ctx_len,
    decode_bsz,
    dtype,
    kv_dtype,
    device_group,
    override_qaic_config,
):
    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "latency",
        "--model",
        model_name,
        "--input-len",
        str(seq_len),
        "--max-model-len",
        str(ctx_len),
        "--batch-size",
        "1",
        "--num-iters-warmup",
        "1",
        "--num-iters",
        "1",
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        kv_dtype,
        "--max-num-seqs",
        str(decode_bsz),
        "--quantization",
        "mxfp6",
        "--long-prefill-token-threshold",
        str(seq_len),
        "--no-async-scheduling",
    ]
    if os.environ.get("DISABLE_PREFIX_CACHING") == "1":
        cmd.append("--no-enable-prefix-caching")
    _additional_config_lat: dict = {"device_group": device_group}
    if override_qaic_config is not None:
        _additional_config_lat["override_qaic_config"] = override_qaic_config
    cmd += ["--additional-config", json.dumps(_additional_config_lat)]
    # Run offline latency
    result = _run_streaming(cmd)

    assert result.returncode == 0
    assert "percentile latency" in result.stdout
