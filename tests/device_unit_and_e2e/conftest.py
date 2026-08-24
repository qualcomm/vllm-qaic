# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import enum
import importlib.util
import json
import os
import random
import socket
import subprocess
import time

import psutil
import pytest
import regex as re
import requests
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm import SamplingParams
from vllm.platforms import current_platform


def _parse_override_config(configs: str) -> dict:
    return {
        str(value[0]): value[1] if len(value) > 1 else True
        for value in (
            re.split(r"[:=]", config.strip())
            for config in re.split(r"[ ]+", configs.strip())
        )
    }


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    in_len: int,
    out_len: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int | None = None,
) -> list[tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    with open(dataset_path) as f:
        dataset = json.load(f)
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    random.shuffle(dataset)

    filtered_dataset: list[tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > in_len or prompt_len + output_len > out_len:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def _ensure_sharegpt_dataset() -> str:
    dataset_path = os.path.join(
        os.path.dirname(__file__), "ShareGPT_V3_unfiltered_cleaned_split.json"
    )
    if not os.path.isfile(dataset_path):
        # Download to a PID-unique temp file, then atomically rename into
        # place - guards against multiple concurrent scheduler jobs (each a
        # separate pytest subprocess) calling this at the same time and
        # interleaving writes to the same final path.
        tmp_path = f"{dataset_path}.{os.getpid()}.tmp"
        try:
            data_download = subprocess.run(
                [
                    "wget",
                    "-O",
                    tmp_path,
                    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json",
                ],
                capture_output=False,
                text=False,
            )
            assert data_download.returncode == 0, "ShareGPT dataset download failed..."
            if not os.path.isfile(dataset_path):
                os.replace(tmp_path, dataset_path)
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
    return dataset_path


def get_prompts_sharegpt(
    num_requests: int, in_len: int, out_len: int, model_name: str
) -> list[str]:
    reqs = sample_sharegpt_requests(
        dataset_path=_ensure_sharegpt_dataset(),
        num_requests=num_requests,
        in_len=in_len,
        out_len=out_len,
        tokenizer=AutoTokenizer.from_pretrained(model_name, padding_side="left"),
        fixed_output_len=None,
    )
    return [request[0] for request in reqs]


def check_outputs_equal(outputs_0, outputs_1, name_0, name_1):
    """Compare two sequences of (token_ids, text) outputs, which should be equal."""
    assert len(outputs_0) == len(outputs_1)
    for i, (o0, o1) in enumerate(zip(outputs_0, outputs_1, strict=False)):
        ids0, text0 = o0
        ids1, text1 = o1
        fail_msg = f"Test{i}:\n{name_0}:\t{text0!r}\n{name_1}:\t{text1!r}"
        assert text0 == text1, fail_msg
        assert ids0 == ids1, fail_msg


def _parse_device_id_pool(value: str) -> list[int]:
    return [int(v) for v in value.split(",") if v.strip()]


def pytest_addoption(parser):
    parser.addoption("--seq-len", type=int, default=None)
    parser.addoption("--ctx-len", type=int, default=None)
    parser.addoption("--decode-bsz", type=int, default=4)
    parser.addoption("--dtype", type=str, default=None)
    parser.addoption("--kv-dtype", type=str, default="auto")
    parser.addoption("--dataset", type=str, default=None)
    parser.addoption(
        "--device-id",
        type=_parse_device_id_pool,
        default=[0],
        help="Comma-separated pool of QAIC device ids available to this run, "
        "e.g. '0,1,2,3,4'.",
    )
    parser.addoption(
        "--device-pool-size",
        type=int,
        default=None,
        help="Size of the real device pool, used only for the collection-time "
        "device-count skip check. Takes precedence over len(--device-id) - "
        "for use by collect_jobs.py, which never receives real device IDs "
        "(the scheduler assigns those per job) but still needs an accurate "
        "pool size to decide which tests to skip.",
    )
    parser.addoption("--host", type=str, default="127.0.0.1")
    parser.addoption("--port", type=int, default=None)
    parser.addoption(
        "--override-qaic-config",
        type=_parse_override_config,
        default=None,
    )


def _closest_marker_for_node(request, node, name):
    """Like `node.get_closest_marker(name)`, but also works when `node` is a
    `Class` or `Module` collector (e.g. from a class- or module-scoped
    fixture's `request.node`). Neither has markers of its own once markers
    are applied only per-test (no class-/module-level `pytestmark`) — so
    fall back to the first collected test item belonging to that collector
    and read its marker."""
    marker = node.get_closest_marker(name)
    if marker is not None:
        return marker
    for collector_type in (pytest.Class, pytest.Module):
        if isinstance(node, collector_type):
            for item in request.session.items:
                if item.getparent(collector_type) is node:
                    return item.get_closest_marker(name)
    return None


def _resolve_qaic_test_config_value(request, pytestconfig, key, default=None):
    """Resolve a QAIC run-config value: closest `qaic_test_config` marker kwarg
    first (falling back to the first test in the class/module if resolving
    against a class- or module-scoped `request`), then the CLI option, then
    `default`. Plain function (not a fixture) so it has no fixed scope and is
    callable against function-, class-, and module-scoped `request` objects."""
    marker = _closest_marker_for_node(request, request.node, "qaic_test_config")
    if marker is not None and key in marker.kwargs:
        return marker.kwargs[key]
    cli_value = pytestconfig.getoption(key, default=None)
    if cli_value is not None:
        return cli_value
    return default


class DevicePoolError(Exception):
    """Raised when a device request can't be satisfied by the pool."""


class _DevicePool:
    """Tracks device ids checked out within one pytest subprocess, so
    overlapping fixtures (e.g. `qaic_model` + `device_group`) never get the
    same id. Cross-process disjointness is `scheduler.py`'s job; no lock
    needed since fixture setup/teardown here is sequential.
    """

    def __init__(self):
        self._held: set[int] = set()

    def acquire(self, pool_ids: list[int], count: int) -> list[int]:
        if count > len(pool_ids):
            raise DevicePoolError(
                f"requested {count} device(s) but pool only has {len(pool_ids)}"
            )
        free = [d for d in pool_ids if d not in self._held]
        if len(free) < count:
            raise DevicePoolError(
                f"requested {count} device(s) but only {len(free)} of pool "
                f"{pool_ids} are currently free (held: {sorted(self._held)})"
            )
        chosen = free[:count]
        self._held.update(chosen)
        return chosen

    def release(self, device_ids: list[int]) -> None:
        self._held.difference_update(device_ids)


_device_pool = _DevicePool()


@pytest.fixture(scope="function")
def model_name(request, pytestconfig):
    return _resolve_qaic_test_config_value(request, pytestconfig, "model_name")


@pytest.fixture(scope="function")
def seq_len(request, pytestconfig):
    return _resolve_qaic_test_config_value(
        request, pytestconfig, "seq_len", default=128
    )


@pytest.fixture(scope="function")
def ctx_len(request, pytestconfig):
    return _resolve_qaic_test_config_value(request, pytestconfig, "ctx_len")


@pytest.fixture(scope="function")
def decode_bsz(request, pytestconfig):
    return _resolve_qaic_test_config_value(
        request,
        pytestconfig,
        "decode_bsz",
        default=pytestconfig.getoption("decode_bsz"),
    )


@pytest.fixture(scope="function")
def dtype(request, pytestconfig):
    return _resolve_qaic_test_config_value(request, pytestconfig, "dtype")


@pytest.fixture(scope="function")
def kv_dtype(request, pytestconfig):
    return _resolve_qaic_test_config_value(
        request, pytestconfig, "kv_dtype", default="auto"
    )


@pytest.fixture(scope="function")
def dataset(request, pytestconfig):
    return _resolve_qaic_test_config_value(request, pytestconfig, "dataset")


@pytest.fixture(scope="function")
def num_prompt(request, pytestconfig):
    return _resolve_qaic_test_config_value(
        request, pytestconfig, "num_prompt", default=20
    )


@pytest.fixture(scope="session")
def override_qaic_config(pytestconfig):
    return pytestconfig.getoption("override_qaic_config")


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
        return True


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def host(pytestconfig):
    return pytestconfig.getoption("host")


@pytest.fixture(scope="session")
def port(pytestconfig, host):
    port = pytestconfig.getoption("port")
    if port is not None and _is_port_free(host, port):
        return port
    free_port = _free_port(host)
    if port is not None:
        print(
            f"** Requested port {port} is already in use on {host}; "
            f"using port {free_port} instead."
        )
    return free_port


@pytest.fixture(scope="session")
def device_pool_ids(pytestconfig):
    return pytestconfig.getoption("device_id")


@pytest.fixture
def device_groups(request, pytestconfig, device_pool_ids):
    num_device_groups = _resolve_qaic_test_config_value(
        request, pytestconfig, "num_device_groups", default=1
    )
    device_group_size = _resolve_qaic_test_config_value(
        request, pytestconfig, "device_group_size", default=1
    )
    ids = _device_pool.acquire(device_pool_ids, num_device_groups * device_group_size)
    request.addfinalizer(lambda: _device_pool.release(ids))
    return [
        ids[i * device_group_size : (i + 1) * device_group_size]
        for i in range(num_device_groups)
    ]


@pytest.fixture
def device_group(device_groups):
    return device_groups[0]


@pytest.fixture
def make_runner(
    vllm_runner,
    model_name,
    decode_bsz,
    ctx_len,
    seq_len,
    dtype,
    kv_dtype,
    override_qaic_config,
):
    def _make(
        async_scheduling: bool,
        dg: list,
        enable_chunked_prefill: bool = False,
        block_size: int = 16,
        lora_modules: list | None = None,
        max_num_seqs: int | None = None,
        override_qaic_config: dict | None = override_qaic_config,
        quantization: str | None = dtype,
        kv_cache_dtype: str = kv_dtype,
        long_prefill_token_threshold: int | None = None,
        **kwargs,
    ):
        additional_config = {
            "override_qaic_config": override_qaic_config,
            "device_group": dg,
        }
        if lora_modules is not None:
            additional_config["lora_modules"] = lora_modules
        return vllm_runner(
            model_name,
            max_num_seqs=max_num_seqs if max_num_seqs is not None else decode_bsz,
            max_model_len=ctx_len,
            long_prefill_token_threshold=(
                seq_len
                if long_prefill_token_threshold is None
                else long_prefill_token_threshold
            ),
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            additional_config=additional_config,
            enable_prefix_caching=False,
            async_scheduling=async_scheduling,
            disable_log_stats=False,
            enable_chunked_prefill=enable_chunked_prefill,
            block_size=block_size,
            **kwargs,
        )

    return _make


@pytest.fixture(scope="session")
def sampling_params():
    return SamplingParams(temperature=0.0, max_tokens=None)


@pytest.fixture(scope="session")
def sample_prompts():
    return [
        "My name is",
        "How to eat mangosteen?",
        "How many people died in World War II",
        "Hello ",
        "Who is the president of United States",
        "Who is the president of India",
        "When it snowfalls in San Diego",
        "In which country yamana river flows",
        "How many people died in World War II",
        "Thy youth is proud livery, so gazed on now",
        "Will be a tattered weed, of small worth held:",
        "Then being asked where all thy beauty lies",
        "Where all the treasure of thy lusty days",
        "To say, within thine own deep-sunken eyes",
        "Where is Statue of Liberty located?",
    ]


@pytest.fixture(scope="function")
def sharegpt_prompts(model_name, seq_len, ctx_len):
    def _get(num_requests: int, in_len: int | None = None) -> list[str]:
        return get_prompts_sharegpt(
            num_requests, in_len if in_len is not None else seq_len, ctx_len, model_name
        )

    return _get


@pytest.fixture(scope="function")
def sharegpt_dataset_path(dataset):
    if dataset != "sharegpt":
        raise ValueError("Only sharegpt dataset is currently supported!!!")
    return _ensure_sharegpt_dataset()


@pytest.fixture(scope="class")
def qaic_model(
    request,
    pytestconfig,
    vllm_runner,
    device_pool_ids,
):
    model_name = _resolve_qaic_test_config_value(request, pytestconfig, "model_name")
    seq_len = _resolve_qaic_test_config_value(
        request, pytestconfig, "seq_len", default=128
    )
    ctx_len = _resolve_qaic_test_config_value(request, pytestconfig, "ctx_len")
    dtype = _resolve_qaic_test_config_value(request, pytestconfig, "dtype")
    kv_dtype = _resolve_qaic_test_config_value(
        request, pytestconfig, "kv_dtype", default="auto"
    )
    decode_bsz = _resolve_qaic_test_config_value(
        request,
        pytestconfig,
        "decode_bsz",
        default=pytestconfig.getoption("decode_bsz"),
    )
    override_qaic_config = _resolve_qaic_test_config_value(
        request,
        pytestconfig,
        "override_qaic_config",
        default=pytestconfig.getoption("override_qaic_config"),
    )
    speculative_config = _resolve_qaic_test_config_value(
        request, pytestconfig, "speculative_config", default=None
    )
    draft_override_qaic_config = _resolve_qaic_test_config_value(
        request, pytestconfig, "draft_override_qaic_config", default=None
    )
    enable_lora = _resolve_qaic_test_config_value(
        request, pytestconfig, "enable_lora", default=False
    )
    max_loras = _resolve_qaic_test_config_value(
        request, pytestconfig, "max_loras", default=None
    )
    lora_modules = _resolve_qaic_test_config_value(
        request, pytestconfig, "lora_modules", default=None
    )
    num_device_groups = _resolve_qaic_test_config_value(
        request, pytestconfig, "num_device_groups", default=1
    )
    device_group_size = _resolve_qaic_test_config_value(
        request, pytestconfig, "device_group_size", default=1
    )

    ids = _device_pool.acquire(device_pool_ids, num_device_groups * device_group_size)
    try:
        additional_config = {
            "device_group": ids,
            "override_qaic_config": override_qaic_config,
        }
        if draft_override_qaic_config is not None:
            additional_config["draft_override_qaic_config"] = draft_override_qaic_config
        if lora_modules is not None:
            additional_config["lora_modules"] = lora_modules
        runner_kwargs = {}
        if max_loras is not None:
            runner_kwargs["max_loras"] = max_loras
        runner = vllm_runner(
            model_name,
            max_num_seqs=decode_bsz,
            max_model_len=ctx_len,
            long_prefill_token_threshold=seq_len,
            quantization=dtype,
            kv_cache_dtype=kv_dtype,
            enable_prefix_caching=False,
            async_scheduling=False,
            speculative_config=speculative_config,
            enable_lora=enable_lora,
            additional_config=additional_config,
            **runner_kwargs,
        )
        with runner as model:
            yield model
    finally:
        _device_pool.release(ids)


def _item_scope(item) -> str:
    """Group tests the way xdist's --dist=loadscope used to: class-scoped
    tests share one job; a bare function's scope is its module."""
    return item.nodeid.rsplit("::", 1)[0]


def pytest_collection_modifyitems(session, config, items):
    device_pool_size = config.getoption("device_pool_size")
    pool_size = (
        device_pool_size
        if device_pool_size is not None
        else len(config.getoption("device_id"))
    )
    for item in items:
        marker = item.get_closest_marker("qaic_test_config")
        kwargs = marker.kwargs if marker is not None else {}
        num_device_groups = kwargs.get("num_device_groups", 1)
        device_group_size = kwargs.get("device_group_size", 1)
        required = num_device_groups * device_group_size
        if required > pool_size:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"not enough devices in pool: needs {required}, "
                        f"pool has {pool_size}"
                    )
                )
            )

        if (
            item.get_closest_marker("qaic_aot_mode") is not None
            and not current_platform.is_aot_inference()
        ):
            item.add_marker(pytest.mark.skip(reason="AOT mode is not installed"))

        if (
            item.get_closest_marker("qaic_disagg_installed") is not None
            and importlib.util.find_spec("qaic_disagg") is None
        ):
            item.add_marker(
                pytest.mark.skip(reason="qaic_disagg package is not installed")
            )


class ServerRunner:
    """Launches a vLLM OpenAI-compatible server subprocess (via either the
    `vllm serve` CLI or `python3 -m vllm.entrypoints.openai.api_server`),
    waits for it to become healthy at `http://{host}:{port}/health`, and
    tears down its whole process tree on context-manager exit."""

    class Backend(enum.Enum):
        VLLM_SERVE = enum.auto()
        OPENAI_API_SERVER_MODULE = enum.auto()

    _BACKEND_LAUNCHERS = {
        Backend.VLLM_SERVE: ["vllm", "serve"],
        Backend.OPENAI_API_SERVER_MODULE: [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
        ],
    }

    def __init__(
        self,
        backend: "ServerRunner.Backend",
        model_name: str,
        host: str,
        port: int,
        seq_len: int,
        ctx_len: int,
        decode_bsz: int,
        dtype: str,
        kv_dtype: str,
        additional_config: dict,
        timeout: float = 900,
        enable_lora: bool = False,
        max_loras: int | None = None,
        lora_modules: list[str] | None = None,
        max_num_batched_tokens: int | None = None,
        trust_remote_code: bool = False,
    ):
        self.base_url = f"http://{host}:{port}"

        cmd = [*self._BACKEND_LAUNCHERS[backend], model_name] + [
            "--host",
            host,
            "--port",
            str(port),
            "--max-model-len",
            str(ctx_len),
            "--max-num-seqs",
            str(decode_bsz),
            "--quantization",
            dtype,
            "--kv-cache-dtype",
            kv_dtype,
            "--no-enable-prefix-caching",
            "--no-async-scheduling",
            "--additional-config",
            json.dumps(additional_config),
        ]
        if max_num_batched_tokens is not None:
            cmd += ["--max-num-batched-tokens", str(max_num_batched_tokens)]
        else:
            cmd += ["--long-prefill-token-threshold", str(seq_len)]
        if enable_lora:
            cmd.append("--enable-lora")
        if max_loras is not None:
            cmd += ["--max-loras", str(max_loras)]
        if lora_modules is not None:
            cmd += ["--lora-modules", *lora_modules]
        if trust_remote_code:
            cmd.append("--trust-remote-code")

        self.process = subprocess.Popen(cmd)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Server exited early with code {self.process.returncode}"
                )
            try:
                if (
                    requests.get(f"{self.base_url}/health", timeout=2).status_code
                    == 200
                ):
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError("Server did not become healthy in time")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        parent = psutil.Process(self.process.pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
        parent.wait(timeout=10)


@pytest.fixture
def server_runner():
    return ServerRunner
