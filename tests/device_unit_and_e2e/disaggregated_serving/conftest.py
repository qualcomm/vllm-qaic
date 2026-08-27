# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared disaggregated-serving test helpers"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import regex as re

from ..conftest import _closest_marker_for_node, _device_pool, _free_port
from .utils import check_port_status


def pytest_addoption(parser):
    parser.addoption("--client-request-timeout", type=int, default=60)
    parser.addoption("--disaggregated-startup-timeout", type=int, default=1200)


@pytest.fixture(scope="session")
def kv_port(host):
    return _free_port(host)


@pytest.fixture
def kv_handoff_server():
    procs: list[subprocess.Popen] = []

    def _start(port: int, size: int) -> subprocess.Popen:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "qaic_disagg.kv_handoff.server",
                "--port",
                str(port),
                "--size",
                str(size),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(10)
        procs.append(proc)
        return proc

    yield _start
    for proc in procs:
        proc.terminate()


def _free_ports(host: str, count: int) -> list[int]:
    ports = []
    while len(ports) < count:
        candidate = _free_port(host)
        if candidate not in ports:
            ports.append(candidate)
    return ports


@pytest.fixture(scope="class")
def _disagg_test_config(request, pytestconfig, host, port, device_pool_ids):
    marker = _closest_marker_for_node(request, request.node, "qaic_test_config")
    if marker is None:
        raise KeyError(
            "TestDisaggregatedServingBasic requires @pytest.mark.qaic_test_config(...) "
            "with model_name, seq_len, ctx_len, decode_bsz, num_prefill_workers, "
            "num_decode_workers, prefill_device_group_size, decode_device_group_size, "
            "prefill_max_num_seqs"
        )
    kwargs = marker.kwargs
    required = (
        "model_name",
        "seq_len",
        "ctx_len",
        "decode_bsz",
        "num_prefill_workers",
        "num_decode_workers",
        "prefill_device_group_size",
        "decode_device_group_size",
        "prefill_max_num_seqs",
    )
    missing = [key for key in required if key not in kwargs]
    if missing:
        raise KeyError(
            f"qaic_test_config marker on TestDisaggregatedServingBasic is missing "
            f"required field(s): {', '.join(missing)}"
        )

    num_prefill_workers = kwargs["num_prefill_workers"]
    num_decode_workers = kwargs["num_decode_workers"]
    prefill_device_group_size = kwargs["prefill_device_group_size"]
    decode_device_group_size = kwargs["decode_device_group_size"]

    num_prefill_devices = num_prefill_workers * prefill_device_group_size
    num_decode_devices = num_decode_workers * decode_device_group_size
    ids = _device_pool.acquire(
        device_pool_ids, num_prefill_devices + num_decode_devices
    )
    request.addfinalizer(lambda: _device_pool.release(ids))
    prefill_ids, decode_ids = ids[:num_prefill_devices], ids[num_prefill_devices:]

    prefill_devices = " ".join(
        ",".join(
            str(d)
            for d in prefill_ids[
                i * prefill_device_group_size : (i + 1) * prefill_device_group_size
            ]
        )
        for i in range(num_prefill_workers)
    )
    decode_devices = " ".join(
        ",".join(
            str(d)
            for d in decode_ids[
                i * decode_device_group_size : (i + 1) * decode_device_group_size
            ]
        )
        for i in range(num_decode_workers)
    )

    config = {
        "model_name": kwargs["model_name"],
        "seq_len": kwargs["seq_len"],
        "ctx_len": kwargs["ctx_len"],
        "decode_bsz": kwargs["decode_bsz"],
        "prefill_devices": prefill_devices,
        "decode_devices": decode_devices,
        "prefill_max_num_seqs": kwargs["prefill_max_num_seqs"],
        "client_request_timeout": pytestconfig.getoption("client_request_timeout"),
        "disaggregated_startup_timeout": pytestconfig.getoption(
            "disaggregated_startup_timeout"
        ),
        "disaggregated_server_port": port,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
    }

    config["prefill_worker_ports"] = ",".join(
        str(p) for p in _free_ports(host, num_prefill_workers)
    )
    config["decode_worker_ports"] = ",".join(
        str(p) for p in _free_ports(host, num_decode_workers)
    )
    config["kv_handoff_port"] = _free_port(host)
    return config


@pytest.fixture(scope="class")
def disagg_server(request, _disagg_test_config):
    """Starts one plain (no router-policy/API-key/SSL/KV-store overrides)
    disaggregated server shared by every test in the class, and tears it
    down once after the last test."""
    test_config = _disagg_test_config
    model_name = test_config["model_name"]
    port = test_config["disaggregated_server_port"]
    prefill_devices = test_config["prefill_devices"]
    decode_devices = test_config["decode_devices"]
    prefill_max_num_seqs = test_config["prefill_max_num_seqs"]
    decode_bsz = test_config["decode_bsz"]
    ctx_len = test_config["ctx_len"]
    seq_len = test_config["seq_len"]
    disaggregated_startup_timeout = test_config["disaggregated_startup_timeout"]
    kv_handoff_port = test_config["kv_handoff_port"]
    server_bind_host = "0.0.0.0"
    current_dir = Path.cwd()

    commands = [
        sys.executable,
        "-m",
        "qaic_disagg",
        "--seed",
        "0",
        "-vvv",
        f"--model={model_name}",
        f"--port={port}",
        "--prefill-device-group",
        *prefill_devices.split(" "),
        "--decode-device-group",
        *decode_devices.split(" "),
        f"--prefill-max-num-seqs={prefill_max_num_seqs}",
        f"--prefill-port={test_config['prefill_worker_ports']}",
        f"--decode-port={test_config['decode_worker_ports']}",
        f"--decode-max-num-seqs={decode_bsz}",
        f"--max-model-len={ctx_len}",
        f"--prefill-long-prefill-token-threshold={seq_len}",
        f"--kv-handOff-port={kv_handoff_port}",
    ]

    test_config["constant_factor"] = (
        decode_bsz * test_config["num_decode_workers"]
        + prefill_max_num_seqs * test_config["num_prefill_workers"]
    )
    test_config["host"] = server_bind_host

    logging.info(
        "Starting disaggregated server (model: %s, port: %s)...", model_name, port
    )
    logging.info("Command: %s", " ".join(commands))
    env = os.environ.copy()
    env["QAIC_DISAGG_DEACTIVATION_TIMEOUT"] = "180"
    process = subprocess.Popen(commands, env=env)

    interval = 5
    elapsed = 0
    while elapsed < disaggregated_startup_timeout:
        if process.poll() is not None:
            pytest.fail(
                f"ERROR: Disaggregated server process terminated unexpectedly with "
                f"exit code {process.returncode}. Check server logs for details."
            )
        if check_port_status(port, server_bind_host, check_open=True):
            logging.info("Port %s is open and accepting connections.", port)
            break
        time.sleep(interval)
        elapsed += interval
    else:
        process.terminate()
        pytest.fail(f"Server {model_name} failed to start on port {port}.")

    test_config["process"] = process
    yield test_config

    # --- Teardown ---
    logging.info(
        "Shutting down disaggregated server (model: %s, port: %s)...", model_name, port
    )
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        logging.warning(
            "Server %s on port %s did not terminate gracefully, forcing kill.",
            model_name,
            port,
        )
        process.kill()

    class_name = request.node.name
    sanitized_name = re.sub(r'[\\/:*?"<>|\]\[()\s-]+', "_", class_name).strip("_")
    model_fn = model_name.split("/")[-1]
    disagg_log_path = current_dir / f"{sanitized_name}/{model_fn}_{port}"
    os.makedirs(disagg_log_path, exist_ok=True)
    for f in current_dir.glob("qaic_*.log"):
        try:
            f.rename(disagg_log_path / f.name)
        except OSError as e:
            logging.warning(
                "Error moving log file %s to %s: %s", f.name, disagg_log_path, e
            )

    logging.info("Sleeping for 5s to clean up the device.")
    time.sleep(5)
    logging.info("Disaggregated server has been shut down.")
