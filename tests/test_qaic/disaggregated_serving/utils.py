# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Shared helpers for the disagg+SpD hardware E2E suite (test_spd_disagg.py).

Ported from vllm_0_15_0 (commit 2502833607's post-commit state). Hardware
dependent -- not exercised by the non-hardware verification gate; see
docs/qaic/disagg_spd_port.md for run instructions once a device group is
reserved.

NOTE: get_prompts("/v1/completions") reads a pickled dataset from
tests/test_qaic/dataset/, which does not yet exist in vllm-qaic. Copy it from
vllm_0_15_0's tests/test_qaic/dataset/ before running any test that calls
get_prompts with that endpoint (see docs/qaic/disagg_spd_port.md).
"""

import datetime
import ipaddress
import json
import logging
import os
import pickle
import platform
import regex as re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest
import requests
from QEfficient import QEFFAutoModelForCausalLM

from vllm.utils.network_utils import is_valid_ipv6_address

if os.environ.get("HF_TOKEN"):
    hf_token = os.environ.get("HF_TOKEN")
else:
    raise OSError("HF_TOKEN is not set")


# Data Loading Utility
def get_prompts(api_endpoint, truncate_at=100, total_max_chars=250):
    if api_endpoint == "/v1/completions":
        # Dataset is already preprocessed & part of the repo.
        mlperf_dataset_fname = str(
            Path(__file__).parent.parent
            / "dataset"
            / "open_orca_gpt4_tokenized_llama3.sampled_1024new_truncated_prompts_to_128.pkl"  # noqa: E501
        )
        with open(mlperf_dataset_fname, "rb") as f:
            dataset = pickle.load(f)
        prompts = dataset["input"].tolist()
    elif api_endpoint == "/v1/chat/completions":
        prompts = get_sharegpt_conversations(
            truncate_at=truncate_at, total_max_chars=total_max_chars
        )
    return prompts


def get_sharegpt_conversations(truncate_at=100, total_max_chars=250):
    """Downloads and parses ShareGPT conversations."""
    share_gpt_json = "./ShareGPT_V3_unfiltered_cleaned_split.json"
    if not os.path.isfile(share_gpt_json):
        logging.info("Downloading ShareGPT dataset...")
        data_download = subprocess.run(
            [
                "wget",
                "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                "./ShareGPT_V3_unfiltered_cleaned_split.json",
            ],
            capture_output=False,
            text=True,
            check=False,
        )
        if data_download.returncode != 0:
            logging.error("ShareGPT dataset download failed: %s", data_download.stderr)
            raise RuntimeError(
                f"ShareGPT dataset download failed: {data_download.stderr}"
            )
        logging.info("ShareGPT dataset downloaded.")

    role_mapping = {"human": "user", "gpt": "assistant"}

    conversations = []
    ESTIMATED_MESSAGE_OVERHEAD = 50

    with open(share_gpt_json) as f:
        share_gpt_data = json.load(f)

    for data in share_gpt_data:
        current_total_content_length = 0
        conversation = []
        for entry in data["conversations"]:
            original_content = entry["value"]
            processed_content = original_content
            if truncate_at is not None and len(processed_content) > truncate_at:
                processed_content = processed_content[:truncate_at]

            message_content_length = len(processed_content)

            if (
                total_max_chars is not None
                and (
                    current_total_content_length
                    + message_content_length
                    + ESTIMATED_MESSAGE_OVERHEAD
                )
                > total_max_chars
            ):
                break

            new_entry = {
                "role": role_mapping.get(entry["from"], entry["from"]),
                "content": processed_content,
            }
            current_total_content_length += message_content_length
            if current_total_content_length > 0:
                conversation.append(new_entry)
        if conversation:  # Only add non-empty conversations
            conversations.append(conversation)
    return conversations


def parse_stream(stream_text: str) -> list:
    """
    Parse stream text into structured data.

    Args:
        stream_text: Raw text from streaming API response

    Returns:
        list: List of parsed objects where:
              - JSON data is parsed into dictionaries
              - "[DONE]" markers are preserved as strings
              - Invalid packets are kept as-is for debugging

    Example:
        Input: "data: {\"text\": \"Hello\"}\n\ndata: [DONE]"
        Output: [{"text": "Hello"}, "[DONE]"]
    """
    # Extract all data lines using regex to match 'data: ' prefix
    data_packets = re.findall(r"data: (.*)", stream_text)
    parsed_packets = []

    for packet in data_packets:
        if packet == "[DONE]":
            parsed_packets.append(packet)
        else:
            try:
                # Attempt to parse JSON data
                parsed_data = json.loads(packet)
                parsed_packets.append(parsed_data)
            except json.JSONDecodeError:
                # Keep invalid packets as-is for debugging purposes
                parsed_packets.append(packet)

    return parsed_packets


@contextmanager
def start_vllm_openai_server(
    model_name,
    port=8080,
    device_group=(0,),
    ctx_len=12288,
    seq_len=256,
    decode_bsz=2,
    host="0.0.0.0",
    seed=0,
):
    commands = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        f"--model={model_name}",
        f"--port={port}",
        f"--host={host}",
        f"--max-num-seqs={decode_bsz}",
        f"--max-seq-len-to-capture={seq_len}",
        f"--max-model-len={ctx_len}",
        f'--additional-config={{"device_group": {list(device_group)}}}',
        "--device=qaic",
        f"--seed={seed}",
    ]
    print(f"\nStarting vLLM OpenAI server with command: {' '.join(commands)}")
    process = subprocess.Popen(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for port 8080 to be open
    max_wait_time = 10 * 60  # seconds
    interval = 15  # seconds
    elapsed = 0
    try:
        while elapsed < max_wait_time:
            if check_port_status(port, host, check_open=True):
                print(f"Port {port} is open and accepting connections.")
                break
            time.sleep(interval)
            elapsed += interval
        else:
            print(f"Error: Timeout: Port {port} did not open within expected time.")
            process.terminate()
            pytest.exit(f"Server failed to start on port {port}.")

        print("vLLM OpenAI server is now running.")
        yield process
    finally:
        print("Shutting down vLLM OpenAI server...")
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logging.warning(
                "Server %s on port %s did not terminate gracefully, forcing kill.",
                model_name,
                port,
            )
            process.kill()
        print("vLLM OpenAI server has been shut down.")


def query_server(
    user_input: str,
    max_tokens: int = None,
    temperature: float = 0,
    ignore_eos: bool = True,
    host: str = "0.0.0.0",
    port: str = "8081",
    api_endpoint="/v1/completions",
    headers: dict = None,
    stream=False,
    timeout=120,
    verify=None,
    ssl_enabled=False,
) -> requests.Response:
    # TODO: Add support for handling ssl certs
    if is_valid_ipv6_address(host):
        host = f"[{host}]"
    url = f"http://{host}:{port}{api_endpoint}"
    if ssl_enabled:
        url = f"https://{host}:{port}{api_endpoint}"

    data_json = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "stream": stream,
    }
    if api_endpoint == "/v1/chat/completions":
        data_json["messages"] = user_input
    elif api_endpoint == "/v1/completions":
        data_json["prompt"] = user_input
    response = requests.post(
        url,
        json=data_json,
        headers=headers,
        timeout=timeout if timeout else 120,
        verify=verify if ssl_enabled else False,
    )
    return response


def _apply_pipeline_partitioning(
    input_data: dict,
    num_devices: int,
    num_partitions: int,
    layers_per_partition: int,
    num_cores: int,
) -> dict:
    """
    Applies pipeline partitioning logic to the input model graph data.

    Args:
        input_data (dict): The initial MDP configuration loaded from JSON.
        num_devices (int): Total number of devices available.
        num_partitions (int): Number of partitions to create.
        layers_per_partition (int): Number of layers to assign per partition.
        num_cores (int): Cores per device. Must match the -aic-num-cores value
            passed to qaic-compile, or compilation fails with
            "numCores from cmdline [N] conflicts with partition config [M]".

    Returns:
        dict: The updated MDP configuration with pipeline partitions.
    """

    def extract_layer_num(node):
        match = re.search(r"/model/layers\.(\d+)/", node)
        return int(match.group(1)) if match else None

    # Flatten all nodes from all partitions in order
    all_nodes = []
    data = input_data.copy()
    for part in data["partitions"]:
        all_nodes.extend(part["nodeList"])

    # Prepare new partitions list
    partitions = [[] for _ in range(num_partitions)]

    # Prepare partitions
    partitions = [[] for _ in range(num_partitions)]

    # Find max layer
    max_layer = -1
    for node in all_nodes:
        layer_num = extract_layer_num(node)
        if layer_num is not None and layer_num > max_layer:
            max_layer = layer_num

    # Assign nodes to partitions, keeping order and grouping special nodes with
    # their surrounding layer
    last_layer_num = 0
    for node in all_nodes:
        layer_num = extract_layer_num(node)
        if layer_num is not None:
            last_layer_num = layer_num
        partition_idx = min(last_layer_num // layers_per_partition, num_partitions - 1)
        partitions[partition_idx].append(node)

    # Assign devices to partitions
    device_ids = list(range(num_devices))
    devices_per_partition = num_devices // num_partitions
    partition_objs = []
    for i, node_list in enumerate(partitions):
        assigned_devices = device_ids[
            i * devices_per_partition : (i + 1) * devices_per_partition
        ]
        partition_objs.append(
            {
                "name": f"Partition{i}",
                "nodeList": node_list,
                "devices": [
                    {
                        "deviceId": dev_id,
                        "numCores": num_cores,
                    }
                    for dev_id in assigned_devices
                ],
            }
        )

    # Update connections
    data["connections"] = [{"devices": device_ids, "type": "p2p"}]
    data["partitions"] = partition_objs

    return data


def generate_mdp_partition_config(
    model_name: str,
    num_devices: int,
    num_partitions: int,
    prefill_seq_len=32,
    ctx_len=128,
    prefill_max_num_seqs=1,
    num_cores: int = 16,
) -> str:
    assert num_partitions <= num_devices, (
        "num_partitions must be less than or equal to num_devices"
    )
    # The real prefill/decode compile path (vllm_qaic's qaic.py) pins
    # qaic_config["num_replicate_kv_heads"] whenever kv_transfer_config is
    # set (i.e. always, for disagg), which gates QEfficient's
    # ReplicateKVHeadTransform and changes the exported ONNX graph's node
    # names/topology. This MDP template must be dumped from a graph with
    # the same transform applied, or the resulting partition config's node
    # names won't match the real compile's graph and qaic-compile will
    # reject it with "invalid node name".
    from QEfficient.utils.config_utils import calculate_num_replicate_kv_heads

    #  Step 1: Dump the initial partition config JSON
    model = QEFFAutoModelForCausalLM.from_pretrained(
        model_name,
        continuous_batching=True,
        token=hf_token,
        attn_implementation="eager",
    )
    text_config = getattr(
        model.model.config, "get_text_config", lambda: model.model.config
    )()
    qaic_config = {
        "num_replicate_kv_heads": calculate_num_replicate_kv_heads(
            num_devices=1,
            text_model_config=text_config,
        )
    }
    model.model.qaic_config = qaic_config

    model_name_simple = model_name.split("/")[-1]
    layers_per_partition = model.num_layers // num_partitions
    # Encode num_cores in the filename so a config generated for a different
    # core count is never silently reused — a numCores mismatch between this
    # file and the -aic-num-cores compile flag fails qaic-compile.
    output_json_path = (
        f"./{model_name_simple}_TS{num_devices // num_partitions}"
        f"_L{layers_per_partition}_{num_partitions}S_C{num_cores}.json"
    )
    dump_json_path = "./temp_mdp_dump_raw.json"
    if os.path.exists(output_json_path):
        print(f"Using Existing MDP file located at: {output_json_path}")
        return output_json_path
    else:
        print(
            f"Creating MDP file with Parameters: num_devices={num_devices}, "
            f"num_partitions={num_partitions}, "
            f"layers_per_partition={layers_per_partition}"
        )
        model.compile(
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            batch_size=1,
            full_batch_size=prefill_max_num_seqs,
            mdp_dump_partition_config=dump_json_path,
        )

        #  Step 2: Load the dumped JSON and apply pipeline partitioning logic
        initial_mdp_data = {}
        try:
            # Load the raw MDP configuration JSON generated above
            with open(dump_json_path) as f:
                initial_mdp_data = json.load(f)
        except Exception as e:
            print(f"An error occurred while loading the dumped JSON: {e}")
            raise

        try:
            final_mdp_data = _apply_pipeline_partitioning(
                initial_mdp_data,
                num_devices,
                num_partitions,
                layers_per_partition,
                num_cores,
            )
        except Exception as e:
            print(f"An error occurred during pipeline partitioning: {e}")
            raise

        #  Step 3: Save the final partitioned JSON
        try:
            with open(output_json_path, "w") as f:
                json.dump(final_mdp_data, f, indent=4)
        except Exception as e:
            print(f"An error occurred while saving the final output JSON: {e}")
            raise
        finally:
            if os.path.exists(dump_json_path):
                os.remove(dump_json_path)
        return output_json_path


def get_vllm_process_info():
    """
    Fetches the full 'ps -ef' output once, then parses it for vLLM processes
    matching both
    'prefill_only=False' (decode) & 'prefill_only=True' (prefill) modes.
    """
    decode_pattern = "prefill_only=False"
    prefill_pattern = "prefill_only=True"

    decode_results = []
    prefill_results = []

    # Regex patterns for extraction
    port_regex = re.compile(r"--port\s+(\d+)")
    device_group_regex = re.compile(r'"device_group":\s*\[([\d,\s]+)\]')

    try:
        for proc in psutil.process_iter(["pid", "cmdline"]):
            if proc.info["cmdline"] is None:
                continue
            cmdline_str = " ".join(proc.info["cmdline"])
            # Check if the process is a 'vllm serve' process
            if "vllm serve" in cmdline_str:
                info = {
                    "pid": str(proc.info["pid"]),
                    "port": None,
                    "device_group": None,
                }

                port_match = port_regex.search(cmdline_str)
                if port_match:
                    info["port"] = port_match.group(1)

                device_group_match = device_group_regex.search(cmdline_str)
                if device_group_match:
                    info["device_group"] = device_group_match.group(1)

                # Categorize based on prefill_only setting
                if decode_pattern in cmdline_str:
                    decode_results.append(info)
                elif prefill_pattern in cmdline_str:
                    prefill_results.append(info)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return [], []
    return prefill_results, decode_results


def check_device_status(device_ids):
    sys.path.append(f"/opt/qti-aic/dev/lib/{platform.machine()}")
    from qaicrt import Util as qaic_util

    return [
        {
            "status": qaic_util().getDeviceInfo(qid)[1].devStatus,
            "numActiveNWs": qaic_util().getDeviceInfo(qid)[1].devData.numActiveNWs,
            "numLoadedNWs": qaic_util().getDeviceInfo(qid)[1].devData.numLoadedNWs,
        }
        for qid in device_ids
    ]


def check_port_status(
    port: int,
    host: str = "localhost",
    timeout: int = 1,
    check_open: bool = True,
) -> bool:
    """
    Checks if a port is open or free on the given host.

    Args:
        port: The port number to check.
        host: The host address (default: 'localhost').
        timeout: The timeout for the socket operation in seconds.
        check_open: If True, checks if the port is open.
                    If False, checks if the port is free.

    Returns:
        True if the port meets the specified status (open or free),
        False otherwise.
    """
    try:
        for res in socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            with socket.socket(af, socktype, proto) as sock:
                sock.settimeout(timeout)
                try:
                    # Connection successful means the port is in use/open
                    sock.connect(sa)
                    return check_open
                except (TimeoutError, ConnectionRefusedError):
                    pass
            # Connection failed (timeout or refused) means port is likely free
            return not check_open
    except (socket.gaierror, Exception):
        # Catch any other unexpected exceptions (e.g., host not found)
        return False  # Unsuccessful check, assume neither open nor free


def get_available_ports(start_port, num_ports):
    ports = []
    used_ports = {
        conn.laddr.port
        for conn in psutil.net_connections()
        if conn.laddr and hasattr(conn.laddr, "port")
    }
    current_port = start_port
    while len(ports) < num_ports:
        # Skip ports that are already in use or not available (not free).
        if current_port not in used_ports and check_port_status(
            current_port, check_open=False
        ):
            ports.append(current_port)
        current_port += 1
    return ",".join(map(str, ports))


def get_ipv6_address():
    """
    Executes 'ifconfig | grep inet6' and parses the output to find IPv6 addresses.
    """
    try:
        # Execute the command
        command = "ifconfig | grep inet6"
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=True
        )
        output_lines = result.stdout.splitlines()

        # Regex to find an IPv6 address after 'inet6 '
        # It specifically looks for global/ULA addresses, including link-local (fe80::)
        # Excludes loopback ::1
        ipv6_pattern = re.compile(r"inet6\s+((?!::1)[0-9a-fA-F:]+)")

        ipv6_addresses = []
        for line in output_lines:
            match = ipv6_pattern.search(line)
            if match:
                address = match.group(1)
                # Some systems might include a /prefix at the end, remove it if present
                if "/" in address:
                    address = address.split("/")[0]
                ipv6_addresses.append(address)

        if ipv6_addresses:
            logging.info(
                "Found IPv6 addresses (excluding link-local and loopback) "
                "via ifconfig | grep inet6:"
            )
            for addr in ipv6_addresses:
                logging.info("- %s", addr)
            return ipv6_addresses
        else:
            logging.warning(
                "No non-link-local IPv6 addresses found using ifconfig | grep inet6."
            )
            return []  # Return empty list instead of None for easier handling
    except Exception:
        logging.exception("An unexpected error occurred")
        return []


def create_self_signed_cert_and_ca(
    server_cert_path,
    server_key_path,
    ca_cert_path,
    ca_key_path,
    host="localhost",
):
    """
    Generates a self-signed CA certificate and a server certificate signed by it.
    Includes only 'host' and the actual machine hostname (if different) in
    server cert SANs. 'localhost' and '127.0.0.1' are explicitly excluded.
    """
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    # Generate CA private key
    ca_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    # Generate CA certificate
    ca_subject = ca_issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Test CA - Test Org"),
        ]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_subject)
        .issuer_name(ca_issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=3650)
        )  # 10 years
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256(), default_backend())
    )

    # Write CA private key and certificate
    with open(ca_key_path, "wb") as f:
        f.write(
            ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(ca_cert_path, "wb") as f:
        f.write(ca_cert.public_bytes(serialization.Encoding.PEM))

    # Generate server private key
    server_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )

    # Prepare SANs for server certificate
    alt_names = []

    # Add the primary host from the function argument
    if host:
        try:
            # Try to parse as IP address first
            ip_obj = ipaddress.ip_address(host)
            alt_names.append(x509.IPAddress(ip_obj))
        except ValueError:
            # If not an IP, treat as DNS name
            alt_names.append(x509.DNSName(host))

    # Add the current machine's hostname if it's different and not already present
    current_machine_hostname = socket.gethostname()
    if current_machine_hostname and not any(
        isinstance(name, x509.DNSName) and name.value == current_machine_hostname
        for name in alt_names
    ):
        alt_names.append(x509.DNSName(current_machine_hostname))

    # Define the Common Name for the certificate (can be the primary host or
    # current machine hostname)
    cert_common_name = host if host else current_machine_hostname

    # Build server certificate
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name(
                [
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test Org"),
                    x509.NameAttribute(
                        NameOID.COMMON_NAME, cert_common_name
                    ),  # <--- Common Name set here
                ]
            )
        )
        .issuer_name(ca_subject)  # Signed by the CA
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        )  # 1 year
    )

    # Only add SAN extension if there are names to add
    if alt_names:
        server_cert = server_cert.add_extension(
            x509.SubjectAlternativeName(alt_names), critical=False
        )  # <--- SANs defined here

    server_cert = server_cert.sign(ca_key, hashes.SHA256(), default_backend())

    # Write server private key and certificate
    with open(server_key_path, "wb") as f:
        f.write(
            server_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
    with open(server_cert_path, "wb") as f:
        f.write(server_cert.public_bytes(serialization.Encoding.PEM))

    return server_cert_path, server_key_path, ca_cert_path, ca_key_path


def within_tolerance(val1, val2, tolerance=1.0):
    # Handle missing or empty values
    if val1 in ("", None) or val2 in ("", None):
        return val1 == val2

    # Handle list comparison
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        return all(
            within_tolerance(v1, v2, tolerance)
            for v1, v2 in zip(val1, val2, strict=False)
        )

    try:
        val1 = float(val1)
        val2 = float(val2)
        return abs(val1 - val2) <= tolerance * max(abs(val1), abs(val2), 1e-8)
    except (ValueError, TypeError):
        return val1 == val2


def parse_generated_texts_and_metrics(
    result_dir: str,
    result_filename: str,
) -> tuple[str, dict[str, str]]:
    file_path = os.path.join(result_dir, result_filename)
    with open(file_path) as f:
        data = json.load(f)
    os.remove(file_path)

    generated_texts = data.get("generated_texts", "")
    keys = [
        "duration",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "request_goodput",
        "output_throughput",
        "total_token_throughput",
        "input_lens",
        "output_lens",
        "mean_ttft_ms",
        "median_ttft_ms",
        "std_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "std_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "std_itl_ms",
        "p99_itl_ms",
    ]
    metrics = {k: data.get(k, "") for k in keys}
    return generated_texts, metrics


def run_benchmark_serving(
    args: list[str], kwargs_dict, result_filename: str
) -> tuple[str, dict[str, str]]:
    pwd = os.getcwd()
    cmd = [
        sys.executable,
        "../../../benchmarks/benchmark_serving.py",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(pwd),
        "--result-filename",
        result_filename,
    ] + args

    def add_arg(flag, value):
        if value is None:
            return
        cmd.append(f"--{flag}")
        if not isinstance(value, bool):  # Skip value for flags
            cmd.append(str(value))

    # Add arguments conditionally
    add_arg("backend", kwargs_dict.get("backend"))
    add_arg("endpoint", kwargs_dict.get("endpoint"))
    add_arg("model", kwargs_dict.get("model_name"))
    add_arg("tokenizer", kwargs_dict.get("tokenizer"))
    add_arg("tokenizer-mode", kwargs_dict.get("tokenizer_mode"))
    add_arg("dataset-name", kwargs_dict.get("dataset"))
    add_arg("num-prompts", kwargs_dict.get("decode_bsz") * 2)
    add_arg("max-concurrency", kwargs_dict.get("max_concurrency"))
    add_arg("request-rate", kwargs_dict.get("request_rate"))
    add_arg("burstiness", kwargs_dict.get("burstiness"))
    add_arg("trust-remote-code", kwargs_dict.get("trust_remote_code"))
    add_arg("seed", kwargs_dict.get("seed"))
    dataset = kwargs_dict.get("dataset")
    seq_len = kwargs_dict.get("seq_len")
    # Indirectly Limiting the generated output sequence length
    ctx_len = min(kwargs_dict.get("ctx_len"), 1024)
    if dataset == "sonnet":
        cmd += [
            "--sonnet-input-len",
            str(seq_len),
            "--sonnet-output-len",
            str(ctx_len - seq_len),
            "dataset-path",
            "../../benchmarks/sonnet.txt",
        ]
    elif dataset == "sharegpt":
        # Download sharegpt dataset
        if not os.path.isfile(f"{pwd}/ShareGPT_V3_unfiltered_cleaned_split.json"):
            data_download = subprocess.run(
                [
                    "wget",
                    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                    f"{pwd}/ShareGPT_V3_unfiltered_cleaned_split.json",
                ],
                capture_output=False,
                text=False,
            )
            assert data_download.returncode == 0, "ShareGPT dataset download failed..."
        dataset_path = f"{pwd}/ShareGPT_V3_unfiltered_cleaned_split.json"

        cmd += [
            "--sharegpt-max-input-len",
            str(seq_len),
            "--sharegpt-max-model-len",
            str(ctx_len),
            "--sharegpt-output-len",
            str(ctx_len - seq_len),
            "--dataset-path",
            dataset_path,
        ]
    elif dataset == "random":
        cmd += [
            "--random-input-len",
            str(seq_len),
            "--random-output-len",
            str(ctx_len - seq_len),
            "--random-range-ratio",
            str(0.0),
        ]
    print(f"Running Benchmarking Script with command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    return parse_generated_texts_and_metrics(str(pwd), result_filename)
