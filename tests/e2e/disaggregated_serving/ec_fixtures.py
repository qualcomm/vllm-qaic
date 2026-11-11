# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Fixtures for disaggregated multimodal serving with the Encoder Cache (EC)."""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ..conftest import _closest_marker_for_node, _device_pool
from .fixtures import _free_ports
from .utils import check_port_status

_REQUIRED_KEYS = (
    "model_name",
    "ctx_len",
    "num_encode_workers",
    "encode_device_group_size",
    "num_prefill_workers",
    "prefill_device_group_size",
    "num_decode_workers",
    "decode_device_group_size",
    "decode_bsz",
)


def _group_devices(ids, num_workers, group_size):
    groups = []
    for worker in range(num_workers):
        chunk = ids[worker * group_size : (worker + 1) * group_size]
        groups.append(",".join(str(device) for device in chunk))
    return groups


@pytest.fixture(scope="class")
def _ec_disagg_test_config(request, pytestconfig, host, port, device_pool_ids):
    marker = _closest_marker_for_node(request, request.node, "qaic_test_config")
    if marker is None:
        raise KeyError(
            "EC disaggregated tests require @pytest.mark.qaic_test_config(...) with "
            + ", ".join(_REQUIRED_KEYS)
        )
    kwargs = marker.kwargs
    missing = [key for key in _REQUIRED_KEYS if key not in kwargs]
    if missing:
        raise KeyError(f"qaic_test_config is missing: {', '.join(missing)}")

    counts = {
        role: (kwargs[f"num_{role}_workers"], kwargs[f"{role}_device_group_size"])
        for role in ("encode", "prefill", "decode")
    }
    total = sum(n * size for n, size in counts.values())
    ids = _device_pool.acquire(device_pool_ids, total)
    request.addfinalizer(lambda: _device_pool.release(ids))

    devices, offset = {}, 0
    for role, (num_workers, group_size) in counts.items():
        span = num_workers * group_size
        devices[role] = _group_devices(
            ids[offset : offset + span], num_workers, group_size
        )
        offset += span

    kv_port, ec_port = _free_ports(host, 2)
    return {
        "kwargs": kwargs,
        "devices": devices,
        "worker_ports": {
            role: ",".join(str(p) for p in _free_ports(host, counts[role][0]))
            for role in counts
        },
        "kv_handoff_port": kv_port,
        "ec_handoff_port": ec_port,
        "disaggregated_server_port": port,
        "client_request_timeout": pytestconfig.getoption("client_request_timeout"),
        "disaggregated_startup_timeout": kwargs.get(
            "disaggregated_startup_timeout",
            pytestconfig.getoption("disaggregated_startup_timeout"),
        ),
    }


def _ec_disagg_command(test_config, use_ec):
    kwargs = test_config["kwargs"]
    devices = test_config["devices"]
    ports = test_config["worker_ports"]
    commands = [
        sys.executable,
        "-m",
        "qaic_disagg",
        "--seed",
        "0",
        "-vvv",
        f"--model={kwargs['model_name']}",
        f"--port={test_config['disaggregated_server_port']}",
        f"--encode-port={ports['encode']}",
        "--encode-device-group",
        *devices["encode"],
        f"--prefill-port={ports['prefill']}",
        "--prefill-device-group",
        *devices["prefill"],
        f"--decode-port={ports['decode']}",
        "--decode-device-group",
        *devices["decode"],
        f"--max-model-len={kwargs['ctx_len']}",
        "--encode-max-num-seqs=1",
        "--prefill-max-num-seqs=1",
        f"--decode-max-num-seqs={kwargs['decode_bsz']}",
        "--generation-config=vllm",
        "--quantization=mxfp6",
        "--kv-cache-dtype=mxint8",
        "--no-async-scheduling",
        f"--limit-mm-per-prompt={json.dumps({'image': 1})}",
        f"--kv-handOff-port={test_config['kv_handoff_port']}",
        *kwargs.get("extra_args", []),
    ]
    for role in ("encode", "prefill", "decode"):
        override = kwargs.get(f"{role}_override_qaic_config")
        if override is not None:
            commands += [f"--{role}-override-qaic-config", json.dumps(override)]
    if use_ec:
        commands += [
            "--ec-connector=QaicECConnector",
            f"--ec-handOff-port={test_config['ec_handoff_port']}",
            f"--ec-store-size-gb={kwargs.get('ec_store_size_gb', 64)}",
            f"--ec-store-ttl={kwargs.get('ec_store_ttl', 3600)}",
        ]
    return commands


@pytest.fixture(scope="class", params=[True, False], ids=["ec", "legacy"])
def ec_disagg_server(request, _ec_disagg_test_config):
    """Starts one E-P-D disaggregated server per transport (EC, then legacy)
    shared by every test in the class. Server logs are written under
    ``test_logs/`` in the repo."""
    test_config = dict(_ec_disagg_test_config)
    use_ec = request.param
    port = test_config["disaggregated_server_port"]
    commands = _ec_disagg_command(test_config, use_ec)
    log_dir = (
        Path(request.config.rootpath)
        / "test_logs"
        / f"ec_disagg_{'ec' if use_ec else 'legacy'}_{port}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Starting EC disaggregated server: %s", " ".join(commands))
    env = os.environ.copy()
    env.setdefault("VLLM_QAIC_USE_FULL_KV_TRANSFER", "1")
    env.setdefault("QAIC_VC_QUEUE_SIZE_POW2", "15")
    env["QAIC_DISAGG_DEACTIVATION_TIMEOUT"] = "180"
    process = subprocess.Popen(commands, env=env, cwd=log_dir)

    timeout = test_config["disaggregated_startup_timeout"]
    elapsed = 0
    while elapsed < timeout:
        if process.poll() is not None:
            pytest.fail(
                f"EC disaggregated server exited early with code {process.returncode}"
            )
        if check_port_status(port, "0.0.0.0", check_open=True):
            break
        time.sleep(5)
        elapsed += 5
    else:
        process.terminate()
        pytest.fail(f"EC disaggregated server failed to start on port {port}")

    test_config.update(process=process, use_ec=use_ec, log_dir=log_dir, host="0.0.0.0")
    yield test_config

    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        process.kill()
    time.sleep(5)

    logging.info("EC disaggregated server logs: %s", log_dir)
