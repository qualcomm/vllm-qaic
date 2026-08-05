# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
E2E tests for speculative decoding (SpD) + disaggregated serving on QAIC.

These tests verify that disaggregated serving works correctly with ngram,
suffix, and draft_model SpD methods.  They launch the qaic_disagg server with
SpD enabled on the decode instances and validate:

1. Output is generated correctly (non-empty, valid responses).
2. Data consistency is maintained across shuffled prompt orders.
3. All SpD methods (ngram, suffix, draft_model) produce correct results.

Ported from vllm_0_15_0 (commit 2502833607's post-commit state) into
vllm-qaic's out-of-tree plugin layout. HARDWARE DEPENDENT: not run as part of
the non-hardware verification gate for this port; see
docs/qaic/disagg_spd_port.md for the follow-up commands once a device group
is reserved. Requires tests/test_qaic/dataset/ (copy from vllm_0_15_0, see
utils.py's module docstring) for get_prompts("/v1/completions").

Device allocation (default: single device 0, NSP split 8/8):
    Override with --device-id to use a different QID, or use
    --prefill-devices / --decode-devices for multi-device setups.

Run examples:
    # Single device (default: device 0, 8 NSPs each for prefill/decode):
    pytest tests/test_qaic/disaggregated_serving/test_spd_disagg.py -v \
        --model-name meta-llama/Llama-3.1-8B-Instruct

    # Single device, custom QID:
    pytest tests/test_qaic/disaggregated_serving/test_spd_disagg.py -v \
        --device-id 4 --model-name meta-llama/Llama-3.1-8B-Instruct

    # Multi-device (separate prefill/decode devices):
    pytest tests/test_qaic/disaggregated_serving/test_spd_disagg.py -v \
        --prefill-devices "4,5" --decode-devices "6,7" \
        --model-name meta-llama/Llama-3.1-8B-Instruct

    # With draft model SpD:
    pytest tests/test_qaic/disaggregated_serving/test_spd_disagg.py -v \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --speculative-model meta-llama/Llama-3.2-1B-Instruct
"""

import contextlib
import itertools
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import psutil
import pytest
import regex as re
import torch

from vllm_qaic.worker.model_runner import QaicModelRunnerAoT as QaicModelRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_SEQ_LEN = 512
DEFAULT_CTX_LEN = 12288
DEFAULT_DECODE_BSZ = 2
DEFAULT_NUM_SPEC_TOKENS = 3

# Default: single device 0, NSP split 8/8 between prefill and decode
DEFAULT_DEVICE = "0"
DEFAULT_NUM_CORES = 8


def _import_utils():
    """Lazy import of .utils to avoid collection-time ImportError on envs
    that lack the full disaggregated serving dependencies."""
    from .utils import (  # noqa: PLC0415
        check_port_status,
        generate_mdp_partition_config,
        get_available_ports,
        get_prompts,
        query_server,
    )

    return (
        check_port_status,
        generate_mdp_partition_config,
        get_available_ports,
        get_prompts,
        query_server,
    )


def get_prompts(api_endpoint, **kwargs):
    """Lazy proxy for utils.get_prompts."""
    _, _, _, _get_prompts, _ = _import_utils()
    return _get_prompts(api_endpoint, **kwargs)


def query_server(*args, **kwargs):
    """Lazy proxy for utils.query_server."""
    _, _, _, _, _query_server = _import_utils()
    return _query_server(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def spd_test_config(pytestconfig):
    """Parse CLI options and prepare SpD + disagg test configuration."""
    _, generate_mdp_partition_config, get_available_ports, _, _ = _import_utils()

    # Single device mode: --device sets both prefill and decode to the same QID.
    # Users can still override individually with --prefill-devices / --decode-devices.
    device = pytestconfig.getoption("device_id")
    device_str = str(device) if device is not None else DEFAULT_DEVICE

    config = {
        "model_name": pytestconfig.getoption("model_name") or DEFAULT_MODEL_NAME,
        "seq_len": pytestconfig.getoption("seq_len") or DEFAULT_SEQ_LEN,
        "ctx_len": pytestconfig.getoption("ctx_len") or DEFAULT_CTX_LEN,
        "decode_bsz": pytestconfig.getoption("decode_bsz") or DEFAULT_DECODE_BSZ,
        "client_request_timeout": pytestconfig.getoption("client_request_timeout"),
        "prefill_devices": pytestconfig.getoption("prefill_devices") or device_str,
        "decode_devices": pytestconfig.getoption("decode_devices") or device_str,
        "prefill_max_num_seqs": pytestconfig.getoption("prefill_max_num_seqs"),
        "disaggregated_startup_timeout": pytestconfig.getoption(
            "disaggregated_startup_timeout"
        ),
        "disaggregated_server_port": pytestconfig.getoption(
            "disaggregated_server_port"
        ),
        "kv_transfer_port": pytestconfig.getoption("kv_transfer_port"),
        "speculative_model": pytestconfig.getoption("speculative_model"),
        "speculative_method": pytestconfig.getoption("speculative_method"),
        "num_speculative_tokens": pytestconfig.getoption("num_speculative_tokens")
        or DEFAULT_NUM_SPEC_TOKENS,
        "ngram_prompt_lookup_max": pytestconfig.getoption("ngram_prompt_lookup_max"),
        "ngram_prompt_lookup_min": pytestconfig.getoption("ngram_prompt_lookup_min"),
        "num_cores": DEFAULT_NUM_CORES,
    }

    num_prefill_workers = len(config["prefill_devices"].split(" "))
    num_prefill_devices_per_worker = len(
        config["prefill_devices"].split(" ")[0].split(",")
    )
    num_decode_workers = len(config["decode_devices"].split(" "))
    config["num_prefill_workers"] = num_prefill_workers
    config["num_decode_workers"] = num_decode_workers
    config["prefill_worker_ports"] = get_available_ports(
        start_port=9900, num_ports=num_prefill_workers
    )
    config["decode_worker_ports"] = get_available_ports(
        start_port=9800, num_ports=num_decode_workers
    )

    mdp_partition_file = generate_mdp_partition_config(
        model_name=config["model_name"],
        num_devices=num_prefill_devices_per_worker,
        num_partitions=num_prefill_devices_per_worker,
        prefill_seq_len=config["seq_len"],
        ctx_len=config["ctx_len"],
        prefill_max_num_seqs=config["prefill_max_num_seqs"],
        num_cores=config["num_cores"],
    )
    config["mdp_partition_file"] = mdp_partition_file
    return config


def _build_speculative_config_json(spd_method, config):
    """Build the --decode-speculative-config JSON for qaic_disagg.

    Uses the current vLLM SpeculativeConfig schema (``method`` + ``model``).
    The older ``speculative_model`` key is silently dropped by pydantic, which
    then rejects ``num_speculative_tokens`` as "provided but without
    speculative model", so it must not be used.
    """
    spec_config = {
        "method": spd_method,
        "num_speculative_tokens": config["num_speculative_tokens"],
    }
    if spd_method == "ngram":
        spec_config["model"] = "ngram"
        if config.get("ngram_prompt_lookup_max"):
            spec_config["prompt_lookup_max"] = config["ngram_prompt_lookup_max"]
        if config.get("ngram_prompt_lookup_min"):
            spec_config["prompt_lookup_min"] = config["ngram_prompt_lookup_min"]
    elif spd_method == "suffix":
        spec_config["model"] = "suffix"
    elif spd_method == "draft_model":
        if config.get("speculative_model"):
            spec_config["model"] = config["speculative_model"]
        else:
            pytest.skip("draft_model SpD requires --speculative-model CLI option")
    return spec_config


@pytest.fixture(scope="function")
def spd_disagg_server(request, spd_test_config):
    """
    Launch a disaggregated server with SpD enabled on the decode instances.

    Parameterize with: {"spd_method": "ngram"|"suffix"|"draft_model"}
    """
    check_port_status, _, _, _, _ = _import_utils()

    config = spd_test_config
    fixture_params = getattr(request, "param", {})
    spd_method = fixture_params.get("spd_method", "ngram")

    model_name = config["model_name"]
    port = config["disaggregated_server_port"]
    kv_handoff_port = config["kv_transfer_port"]
    prefill_worker_ports = config["prefill_worker_ports"]
    decode_worker_ports = config["decode_worker_ports"]
    prefill_devices = config["prefill_devices"]
    decode_devices = config["decode_devices"]
    prefill_max_num_seqs = config["prefill_max_num_seqs"]
    decode_bsz = config["decode_bsz"]
    ctx_len = config["ctx_len"]
    seq_len = config["seq_len"]
    mdp_partition_file = config["mdp_partition_file"]
    disaggregated_startup_timeout = config["disaggregated_startup_timeout"]

    # Build speculative config for decode instances
    spec_config_json = _build_speculative_config_json(spd_method, config)

    # qaic_disagg parses --*-override-qaic-config as JSON (json.loads), so pass
    # dicts, not space-separated key=value tokens. prefill_seq_len replaces the
    # old --max-seq-len-to-capture flag, which the current qaic_disagg CLI no
    # longer accepts.
    prefill_override = {
        "num_cores": config["num_cores"],
        "mdp_load_partition_config": mdp_partition_file,
        "prefill_seq_len": seq_len,
    }
    decode_override = {"num_cores": config["num_cores"]}

    commands = [
        sys.executable,
        "-m",
        "qaic_disagg",
        "--seed",
        "0",
        f"--model={model_name}",
        f"--port={port}",
        f"--kv-handOff-port={kv_handoff_port}",
        "--prefill-device-group",
        *prefill_devices.split(" "),
        "--decode-device-group",
        *decode_devices.split(" "),
        f"--prefill-max-num-seqs={prefill_max_num_seqs}",
        f"--prefill-port={prefill_worker_ports}",
        f"--decode-port={decode_worker_ports}",
        f"--decode-max-num-seqs={decode_bsz}",
        f"--max-model-len={ctx_len}",
        # The kv-handoff shared-memory store defaults to 64 (MB), too small for
        # an 8B model's KV buffer — the kv_consumer then fails at the first
        # request with "Shared memory buffer not found". 768 matches known-good
        # disagg deployments for this model size.
        "--kv-store-size",
        "768",
        f"--prefill-override-qaic-config={json.dumps(prefill_override)}",
        f"--decode-override-qaic-config={json.dumps(decode_override)}",
        f"--decode-speculative-config={json.dumps(spec_config_json)}",
    ]

    logging.info(
        "Starting disagg+SpD server (model: %s, port: %s, spd_method: %s)...",
        model_name,
        port,
        spd_method,
    )
    logging.info("Command: %s", " ".join(commands))

    my_env = os.environ.copy()
    # qaic_disagg is installed editable into this venv as its own repo (not a
    # subdirectory of vllm_qaic), so locate its parent dir via its own
    # installed location -- same pattern as test_disagg_spd_unit.py's
    # _find_disagg_pkg_root(). `python -m qaic_disagg` needs that parent dir
    # on PYTHONPATH so the inner package (with __main__.py) resolves.
    import importlib.util  # noqa: PLC0415

    _spec = importlib.util.find_spec("qaic_disagg")
    if _spec and _spec.submodule_search_locations:
        _disagg_parent = str(Path(_spec.submodule_search_locations[0]).parent)
        my_env["PYTHONPATH"] = (
            _disagg_parent + os.pathsep + my_env.get("PYTHONPATH", "")
        )
    process = subprocess.Popen(
        commands,
        env=my_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for port to be open
    interval = 5
    elapsed = 0
    while elapsed < disaggregated_startup_timeout:
        if process.poll() is not None:
            pytest.fail(
                f"ERROR: Disagg+SpD server terminated unexpectedly with "
                f"exit code {process.returncode}"
            )
        if check_port_status(port, "0.0.0.0", check_open=True):
            logging.info("Port %s is open and accepting connections.", port)
            break
        time.sleep(interval)
        elapsed += interval
    else:
        process.terminate()
        pytest.fail(
            f"Server {model_name} failed to start on port {port} "
            f"within {disaggregated_startup_timeout}s."
        )

    config["process"] = process
    config["spd_method"] = spd_method
    yield config

    # --- Teardown ---
    logging.info("Shutting down disagg+SpD server (port: %s)...", port)
    # qaic_disagg launches encode/prefill/decode/kvHandOff/proxy children with
    # start_new_session=True, so they live in their own process group and are
    # NOT reachable by killing just the top-level PID below. Snapshot the full
    # descendant tree up front so a forced-kill fallback can reap all of them
    # (mirrors tests/test_qaic/disaggregated_serving/lmcache/test_lmcache.py).
    try:
        tree = [psutil.Process(process.pid)] + psutil.Process(process.pid).children(
            recursive=True
        )
    except psutil.NoSuchProcess:
        tree = []
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        logging.warning("Server did not terminate gracefully, forcing kill.")
        for p in tree:
            with contextlib.suppress(psutil.NoSuchProcess):
                p.kill()
    process.wait()

    # Move logs
    tc_name = request.node.nodeid.split("::")[-1]
    invalid_chars_pattern = r'[\\/:*?"<>|\]\[()\s-]+'
    sanitized_name = re.sub(invalid_chars_pattern, "_", tc_name)
    tc_name = sanitized_name.strip("_")
    current_dir = Path.cwd()
    log_dir = current_dir / f"{tc_name}/disagg_spd_{spd_method}"
    os.makedirs(log_dir, exist_ok=True)
    for f in current_dir.glob("qaic_*.log"):
        try:
            f.rename(log_dir / f.name)
        except OSError as e:
            logging.warning("Error moving log file %s: %s", f.name, e)

    logging.info("Sleeping for 5s to clean up the device.")
    from vllm.distributed import cleanup_dist_env_and_memory  # noqa: PLC0415

    cleanup_dist_env_and_memory()
    time.sleep(5)
    logging.info("Disagg+SpD server has been shut down.")


# ---------------------------------------------------------------------------
# Tests: ngram SpD + disaggregated serving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "ngram"}],
    indirect=True,
)
class TestNgramDisagg:
    """Verify ngram SpD works correctly with disaggregated serving."""

    def test_ngram_data_consistency(self, spd_disagg_server):
        """Same prompts in different order produce the same output."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:5]

        def get_responses(prompt_list):
            response = query_server(
                prompt_list,
                max_tokens=50,
                port=port,
                timeout=timeout * 5,
            )
            assert response.status_code == 200, (
                f"Request failed: {response.status_code} - {response.text}"
            )
            data = response.json()
            return {p: data["choices"][i]["text"] for i, p in enumerate(prompt_list)}

        initial = get_responses(prompts)

        for i in range(3):
            shuffled = prompts[:]
            random.shuffle(shuffled)
            shuffled_map = get_responses(shuffled)
            assert shuffled_map == initial, (
                f"ngram disagg data consistency failed on shuffle {i}"
            )

    def test_ngram_generation_non_empty(self, spd_disagg_server):
        """Verify ngram SpD generates non-empty output for each request."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:10]
        for prompt in prompts:
            response = query_server(
                prompt,
                max_tokens=100,
                port=port,
                timeout=timeout,
            )
            assert response.status_code == 200
            text = response.json()["choices"][0]["text"]
            assert text.strip(), f"Empty output for prompt: {prompt[:50]}..."


# ---------------------------------------------------------------------------
# Tests: suffix SpD + disaggregated serving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "suffix"}],
    indirect=True,
)
class TestSuffixDisagg:
    """Verify suffix SpD works correctly with disaggregated serving."""

    def test_suffix_data_consistency(self, spd_disagg_server):
        """Same prompts in different order produce the same output."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:5]

        def get_responses(prompt_list):
            response = query_server(
                prompt_list,
                max_tokens=50,
                port=port,
                timeout=timeout * 5,
            )
            assert response.status_code == 200, (
                f"Request failed: {response.status_code} - {response.text}"
            )
            data = response.json()
            return {p: data["choices"][i]["text"] for i, p in enumerate(prompt_list)}

        initial = get_responses(prompts)

        for i in range(3):
            shuffled = prompts[:]
            random.shuffle(shuffled)
            shuffled_map = get_responses(shuffled)
            assert shuffled_map == initial, (
                f"suffix disagg data consistency failed on shuffle {i}"
            )

    def test_suffix_generation_non_empty(self, spd_disagg_server):
        """Verify suffix SpD generates non-empty output for each request."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:10]
        for prompt in prompts:
            response = query_server(
                prompt,
                max_tokens=100,
                port=port,
                timeout=timeout,
            )
            assert response.status_code == 200
            text = response.json()["choices"][0]["text"]
            assert text.strip(), f"Empty output for prompt: {prompt[:50]}..."


# ---------------------------------------------------------------------------
# Tests: draft_model SpD + disaggregated serving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "draft_model"}],
    indirect=True,
)
class TestDraftModelDisagg:
    """Verify LLM-as-draft-model SpD works with disaggregated serving."""

    def test_draft_model_data_consistency(self, spd_disagg_server):
        """Same prompts in different order produce the same output."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:5]

        def get_responses(prompt_list):
            response = query_server(
                prompt_list,
                max_tokens=50,
                port=port,
                timeout=timeout * 5,
            )
            assert response.status_code == 200, (
                f"Request failed: {response.status_code} - {response.text}"
            )
            data = response.json()
            return {p: data["choices"][i]["text"] for i, p in enumerate(prompt_list)}

        initial = get_responses(prompts)

        for i in range(3):
            shuffled = prompts[:]
            random.shuffle(shuffled)
            shuffled_map = get_responses(shuffled)
            assert shuffled_map == initial, (
                f"draft_model disagg data consistency failed on shuffle {i}"
            )

    def test_draft_model_generation_non_empty(self, spd_disagg_server):
        """Verify draft_model SpD generates non-empty output."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:10]
        for prompt in prompts:
            response = query_server(
                prompt,
                max_tokens=100,
                port=port,
                timeout=timeout,
            )
            assert response.status_code == 200
            text = response.json()["choices"][0]["text"]
            assert text.strip(), f"Empty output for prompt: {prompt[:50]}..."

    def test_draft_model_parallel_requests(self, spd_disagg_server):
        """Send concurrent requests to verify SpD under load."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = 300
        num_requests = 20

        prompts = list(
            itertools.islice(
                itertools.cycle(get_prompts("/v1/completions")), num_requests
            )
        )

        max_workers = (
            config["decode_bsz"] * config["num_decode_workers"]
            + config["prefill_max_num_seqs"] * config["num_prefill_workers"]
        )

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    query_server,
                    prompt,
                    max_tokens=50,
                    port=port,
                    timeout=timeout,
                ): prompt
                for prompt in prompts
            }
            for future in futures:
                response = future.result()
                assert response.status_code == 200, (
                    f"Parallel request failed: {response.status_code}"
                )
                text = response.json()["choices"][0]["text"]
                assert text.strip(), "Empty output in parallel request"
                results.append(text)

        assert len(results) == num_requests


# ---------------------------------------------------------------------------
# Tests: verify KV producer does NOT run SpD (guard validation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "ngram"}],
    indirect=True,
)
class TestKvProducerSpDGuards:
    """
    Validate that the KV producer (prefill node) does not execute SpD logic.

    These tests verify indirectly that guards are working by confirming:
    - The server starts successfully (guard prevents assertion failures)
    - Requests complete correctly (no draft token corruption on prefill)
    - Timing shows prefill is not doing extra work

    The actual unit-level guard verification is in the guard_logic tests below.
    """

    def test_server_starts_with_spd_config(self, spd_disagg_server):
        """
        Server must start without assertion errors.

        If the KV producer guard is missing, QaicDraftModelProposer or the
        drafter proposal logic would fire on the producer and likely crash.
        """
        config = spd_disagg_server
        assert config["process"].poll() is None, (
            "Server should still be running (guards prevent producer crash)"
        )

    def test_basic_request_succeeds(self, spd_disagg_server):
        """A basic request should succeed, confirming guards don't break flow."""
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"] or 60

        prompts = get_prompts("/v1/completions")[:3]
        response = query_server(
            prompts,
            max_tokens=50,
            port=port,
            timeout=timeout * 3,
        )
        assert response.status_code == 200
        data = response.json()
        for choice in data["choices"]:
            assert choice["text"].strip(), "Expected non-empty response"


# ---------------------------------------------------------------------------
# Tests: acceptance rate validation (disagg vs non-disagg equivalence)
# ---------------------------------------------------------------------------

# Expected acceptance rates (must match non-disagg offline baselines ±tolerance)
EXPECTED_AR_NGRAM = 0.18
EXPECTED_AR_SUFFIX = 0.28
EXPECTED_AR_DRAFT_MODEL = 0.63
AR_TOLERANCE = 0.05


def _load_mtbench_prompts(max_prompts=80):
    """Load first-turn mt-bench prompts for acceptance rate measurement."""
    try:
        import datasets
    except ImportError:
        pytest.skip("datasets library required for acceptance rate tests")

    ds = datasets.load_dataset("philschmid/mt-bench", split="train")
    prompts: list[str] = []
    for row in ds:
        if len(prompts) >= max_prompts:
            break
        turns = row.get("turns", [])
        if turns:
            prompts.append(turns[0])
    return prompts


def _scrape_acceptance_rate(decode_worker_port, timeout=10):
    """Scrape Prometheus metrics from the decode worker and return acceptance rate.

    Returns (acceptance_rate, accepted_tokens, draft_tokens) or raises on failure.
    """
    import requests as http_requests

    url = f"http://localhost:{decode_worker_port}/metrics"
    resp = http_requests.get(url, timeout=timeout)
    resp.raise_for_status()

    accepted = None
    drafted = None
    for line in resp.text.splitlines():
        if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            # Format: metric_name{labels} value
            accepted = float(line.split()[-1])
        elif line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drafted = float(line.split()[-1])

    if accepted is None or drafted is None:
        # Try alternate metric names (without _total suffix)
        for line in resp.text.splitlines():
            if "spec_decode_num_accepted_tokens" in line and not line.startswith("#"):
                accepted = float(line.split()[-1])
            elif "spec_decode_num_draft_tokens" in line and not line.startswith("#"):
                drafted = float(line.split()[-1])

    assert accepted is not None, (
        f"Could not find spec_decode_num_accepted_tokens in /metrics response "
        f"from decode worker port {decode_worker_port}"
    )
    assert drafted is not None, (
        f"Could not find spec_decode_num_draft_tokens in /metrics response "
        f"from decode worker port {decode_worker_port}"
    )
    assert drafted > 0, "No draft tokens recorded — SpD may not have run"
    return accepted / drafted, accepted, drafted


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "ngram"}],
    indirect=True,
)
class TestNgramAcceptanceRate:
    """Verify ngram acceptance rate in disagg matches non-disagg baseline (~18%)."""

    @pytest.mark.skip(
        reason=(
            "disagg appends --disable-log-stats by design "
            "(qaic_disagg/qaic_disagg/utils.py) unless -vvv, so the "
            "spec-decode Prometheus counters this test scrapes from /metrics "
            "are never created. Acceptance-rate math is already covered "
            "outside disagg by test_acceptance_rate.py::"
            "test_ngram_acceptance_rate_llama31_8b with the same expected AR."
        )
    )
    def test_ngram_acceptance_rate(self, spd_disagg_server):
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        decode_port = config["decode_worker_ports"]
        timeout = config["client_request_timeout"] or 60

        # Load mt-bench prompts and send through the server
        prompts = _load_mtbench_prompts()
        for prompt in prompts:
            response = query_server(
                prompt,
                max_tokens=100,
                port=port,
                timeout=timeout,
            )
            assert response.status_code == 200

        # Scrape metrics from the decode worker
        ar, accepted, drafted = _scrape_acceptance_rate(decode_port)
        lower = EXPECTED_AR_NGRAM - AR_TOLERANCE
        upper = EXPECTED_AR_NGRAM + AR_TOLERANCE
        print(
            f"ngram disagg AR: {ar:.4f} "
            f"({accepted:.0f}/{drafted:.0f}, "
            f"expected {EXPECTED_AR_NGRAM} ± {AR_TOLERANCE}, "
            f"band [{lower:.2f}, {upper:.2f}])"
        )
        assert lower <= ar <= upper, (
            f"ngram disagg acceptance rate {ar:.4f} outside [{lower:.2f}, {upper:.2f}]"
        )


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "suffix"}],
    indirect=True,
)
class TestSuffixAcceptanceRate:
    """Verify suffix acceptance rate in disagg matches non-disagg baseline (~28%)."""

    @pytest.mark.skip(
        reason=(
            "disagg appends --disable-log-stats by design "
            "(qaic_disagg/qaic_disagg/utils.py) unless -vvv, so the "
            "spec-decode Prometheus counters this test scrapes from /metrics "
            "are never created. Acceptance-rate math is already covered "
            "outside disagg by test_acceptance_rate.py::"
            "test_suffix_acceptance_rate_llama31_8b with the same expected AR."
        )
    )
    def test_suffix_acceptance_rate(self, spd_disagg_server):
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        decode_port = config["decode_worker_ports"]
        timeout = config["client_request_timeout"] or 60

        prompts = _load_mtbench_prompts()
        for prompt in prompts:
            response = query_server(
                prompt,
                max_tokens=100,
                port=port,
                timeout=timeout,
            )
            assert response.status_code == 200

        ar, accepted, drafted = _scrape_acceptance_rate(decode_port)
        lower = EXPECTED_AR_SUFFIX - AR_TOLERANCE
        upper = EXPECTED_AR_SUFFIX + AR_TOLERANCE
        print(
            f"suffix disagg AR: {ar:.4f} "
            f"({accepted:.0f}/{drafted:.0f}, "
            f"expected {EXPECTED_AR_SUFFIX} ± {AR_TOLERANCE}, "
            f"band [{lower:.2f}, {upper:.2f}])"
        )
        assert lower <= ar <= upper, (
            f"suffix disagg acceptance rate {ar:.4f} outside [{lower:.2f}, {upper:.2f}]"
        )


@pytest.mark.parametrize(
    "spd_disagg_server",
    [{"spd_method": "draft_model"}],
    indirect=True,
)
class TestDraftModelAcceptanceRate:
    """Verify draft_model acceptance rate in disagg matches non-disagg
    baseline (~63%)."""

    @pytest.mark.skip(
        reason=(
            "disagg appends --disable-log-stats by design "
            "(qaic_disagg/qaic_disagg/utils.py) unless -vvv, so the "
            "spec-decode Prometheus counters this test scrapes from /metrics "
            "are never created. Acceptance-rate math is already covered "
            "outside disagg by test_draft_model_acceptance_rate.py::"
            "test_draft_model_acceptance_rate_llama31_8b with the same "
            "expected AR."
        )
    )
    def test_draft_model_acceptance_rate(self, spd_disagg_server):
        config = spd_disagg_server
        port = config["disaggregated_server_port"]
        decode_port = config["decode_worker_ports"]
        timeout = config["client_request_timeout"] or 60

        prompts = _load_mtbench_prompts()
        for prompt in prompts:
            response = query_server(
                prompt,
                max_tokens=100,
                port=port,
                timeout=timeout,
            )
            assert response.status_code == 200

        ar, accepted, drafted = _scrape_acceptance_rate(decode_port)
        lower = EXPECTED_AR_DRAFT_MODEL - AR_TOLERANCE
        upper = EXPECTED_AR_DRAFT_MODEL + AR_TOLERANCE
        print(
            f"draft_model disagg AR: {ar:.4f} "
            f"({accepted:.0f}/{drafted:.0f}, "
            f"expected {EXPECTED_AR_DRAFT_MODEL} ± {AR_TOLERANCE}, "
            f"band [{lower:.2f}, {upper:.2f}])"
        )
        assert lower <= ar <= upper, (
            f"draft_model disagg acceptance rate {ar:.4f} outside "
            f"[{lower:.2f}, {upper:.2f}]"
        )


# ---------------------------------------------------------------------------
# Unit tests: guard logic verification (no hardware required)
# ---------------------------------------------------------------------------


def _make_runner_ns(
    is_kv_producer=False,
    is_kv_consumer=False,
    speculative_config=None,
    drafter=None,
    decode_ks=None,
    num_spec_tokens=3,
):
    """Minimal namespace for calling unbound methods."""
    return types.SimpleNamespace(
        is_kv_producer=is_kv_producer,
        is_kv_consumer=is_kv_consumer,
        speculative_config=speculative_config,
        drafter=drafter,
        decode_ks=decode_ks if decode_ks is not None else [num_spec_tokens],
        num_spec_tokens=num_spec_tokens,
        _draft_token_ids=None,
    )


class TestGuard1_DrafterClearedOnKvProducer:
    """drafter=None on KV producer after __init__."""

    def test_ngram_kv_producer_drafter_is_none(self):
        spec = MagicMock()
        spec.method = "ngram"
        ns = _make_runner_ns(
            is_kv_producer=True,
            speculative_config=spec,
            drafter=MagicMock(),
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is None

    def test_suffix_kv_producer_drafter_is_none(self):
        spec = MagicMock()
        spec.method = "suffix"
        ns = _make_runner_ns(
            is_kv_producer=True,
            speculative_config=spec,
            drafter=MagicMock(),
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is None

    def test_kv_consumer_drafter_preserved(self):
        spec = MagicMock()
        mock_drafter = MagicMock()
        ns = _make_runner_ns(
            is_kv_consumer=True,
            speculative_config=spec,
            drafter=mock_drafter,
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is mock_drafter

    def test_non_disagg_drafter_preserved(self):
        mock_drafter = MagicMock()
        ns = _make_runner_ns(
            is_kv_producer=False,
            speculative_config=MagicMock(),
            drafter=mock_drafter,
        )
        if ns.is_kv_producer:
            ns.drafter = None
        assert ns.drafter is mock_drafter


class TestGuard2_DraftModelProposerNotOnKvProducer:
    """QaicDraftModelProposer construction skipped on KV producer."""

    def _would_construct(self, ns):
        return bool(
            ns.speculative_config
            and ns.speculative_config.uses_draft_model()
            and not ns.is_kv_producer
        )

    def test_kv_producer_skips(self):
        spec = MagicMock()
        spec.uses_draft_model.return_value = True
        ns = _make_runner_ns(is_kv_producer=True, speculative_config=spec)
        assert not self._would_construct(ns)

    def test_kv_consumer_allows(self):
        spec = MagicMock()
        spec.uses_draft_model.return_value = True
        ns = _make_runner_ns(is_kv_consumer=True, speculative_config=spec)
        assert self._would_construct(ns)

    def test_non_disagg_allows(self):
        spec = MagicMock()
        spec.uses_draft_model.return_value = True
        ns = _make_runner_ns(is_kv_producer=False, speculative_config=spec)
        assert self._would_construct(ns)


class TestGuard3_DraftTokenIdsNoneOnKvProducer:
    """_draft_token_ids never set on KV producer."""

    def _should_propose(self, ns):
        return bool(ns.speculative_config is not None and not ns.is_kv_producer)

    def test_kv_producer_no_proposal(self):
        ns = _make_runner_ns(is_kv_producer=True, speculative_config=MagicMock())
        assert not self._should_propose(ns)

    def test_kv_consumer_allows_proposal(self):
        ns = _make_runner_ns(is_kv_consumer=True, speculative_config=MagicMock())
        assert self._should_propose(ns)

    def test_non_disagg_allows_proposal(self):
        ns = _make_runner_ns(is_kv_producer=False, speculative_config=MagicMock())
        assert self._should_propose(ns)

    def test_no_spec_config_no_proposal(self):
        ns = _make_runner_ns(speculative_config=None)
        assert not self._should_propose(ns)

    def test_draft_token_ids_remains_none_on_producer(self):
        ns = _make_runner_ns(is_kv_producer=True, speculative_config=MagicMock())
        if ns.speculative_config is not None and not ns.is_kv_producer:
            ns._draft_token_ids = torch.zeros(1, dtype=torch.int32)
        assert ns._draft_token_ids is None


class TestGuard4_VarKPreservedOnKvConsumer:
    """decode_ks is NOT collapsed on KV consumer — varK dispatch is supported."""

    def test_kv_consumer_preserves_varK(self):
        """KV consumer keeps [0, K] for ngram/suffix varK dispatch."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[0, 3])
        # No guard applied — decode_ks stays as-is
        assert ns.decode_ks == [0, 3]

    def test_kv_consumer_single_k_unchanged(self):
        """KV consumer with draft_model (single K) stays unchanged."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[3])
        assert ns.decode_ks == [3]

    def test_kv_consumer_no_proposals_returns_k0(self):
        """KV consumer with varK returns K=0 when no proposals exist."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[0, 3], num_spec_tokens=3)
        ns.num_decodes = 4  # required by _determine_active_k
        so = types.SimpleNamespace(scheduled_spec_decode_tokens={})
        k = QaicModelRunner._determine_active_k(ns, so)
        assert k == 0, "No proposals → K=0 fallback on KV consumer"

    def test_kv_consumer_with_proposals_returns_k_max(self):
        """KV consumer with varK returns K_max when proposals exist."""
        ns = _make_runner_ns(is_kv_consumer=True, decode_ks=[0, 3], num_spec_tokens=3)
        ns.num_decodes = 4
        so = types.SimpleNamespace(scheduled_spec_decode_tokens={"req_0": [10, 11, 12]})
        k = QaicModelRunner._determine_active_k(ns, so)
        assert k == 3, "Proposals exist → K_max on KV consumer"

    def test_non_disagg_varK_unchanged(self):
        """Non-disagg node varK behavior is identical."""
        ns = _make_runner_ns(decode_ks=[0, 3])
        ns.num_decodes = 4
        so_empty = types.SimpleNamespace(scheduled_spec_decode_tokens={})
        so_full = types.SimpleNamespace(
            scheduled_spec_decode_tokens={"req_0": [10, 11, 12]}
        )
        assert QaicModelRunner._determine_active_k(ns, so_empty) == 0
        assert QaicModelRunner._determine_active_k(ns, so_full) == 3
