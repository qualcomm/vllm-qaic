# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device unit tests for QAIC online serving (the vLLM HTTP server).

The full online suite (test_api_server.py, test_openai_multi_client.py)
exercises 100-way concurrency, mid-stream request cancellation, and the
official OpenAI async client against a real server process — heavier
server-lifecycle scenarios appropriately left to e2e.

This file is the missing on-device-unit layer: confirms the smallest
possible real server (`vllm serve` subprocess, via the same server_runner
fixture the e2e files use) actually starts, serves one request, and serves
a couple of concurrent requests correctly — without the cancellation or
high-concurrency stress scenarios.

Coverage areas
--------------
1. The server starts and serves a single /v1/completions request
2. Two concurrent requests against the same server both succeed
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from .conftest import ServerRunner, _device_pool, _resolve_qaic_test_config_value


@pytest.fixture(scope="class")
def online_server(request, pytestconfig, device_pool_ids, host, port):
    """Class-scoped (unlike server_runner's other e2e consumers) so the
    server subprocess starts once and is shared by every test in
    TestOnlineServingOnDevice, rather than restarting per test — matching
    the qaic_model/disagg_server pattern elsewhere in this suite. Resolves
    config directly via _resolve_qaic_test_config_value/_device_pool
    (as qaic_model does) instead of depending on the function-scoped
    model_name/seq_len/.../device_group fixtures, which would otherwise
    raise a pytest ScopeMismatch against this fixture's class scope."""
    model_name = _resolve_qaic_test_config_value(request, pytestconfig, "model_name")
    seq_len = _resolve_qaic_test_config_value(
        request, pytestconfig, "seq_len", default=128
    )
    ctx_len = _resolve_qaic_test_config_value(request, pytestconfig, "ctx_len")
    decode_bsz = _resolve_qaic_test_config_value(
        request,
        pytestconfig,
        "decode_bsz",
        default=pytestconfig.getoption("decode_bsz"),
    )
    dtype = _resolve_qaic_test_config_value(request, pytestconfig, "dtype")
    kv_dtype = _resolve_qaic_test_config_value(
        request, pytestconfig, "kv_dtype", default="auto"
    )
    override_qaic_config = _resolve_qaic_test_config_value(
        request,
        pytestconfig,
        "override_qaic_config",
        default=pytestconfig.getoption("override_qaic_config"),
    )
    num_device_groups = _resolve_qaic_test_config_value(
        request, pytestconfig, "num_device_groups", default=1
    )
    device_group_size = _resolve_qaic_test_config_value(
        request, pytestconfig, "device_group_size", default=1
    )

    ids = _device_pool.acquire(device_pool_ids, num_device_groups * device_group_size)
    try:
        additional_config = {"device_group": ids}
        if override_qaic_config is not None:
            additional_config["override_qaic_config"] = override_qaic_config

        with ServerRunner(
            ServerRunner.Backend.VLLM_SERVE,
            model_name,
            host,
            port,
            seq_len,
            ctx_len,
            decode_bsz,
            dtype,
            kv_dtype,
            additional_config,
        ) as server:
            yield server.base_url
    finally:
        _device_pool.release(ids)


def _query(base_url: str, model_name: str, prompt: str) -> dict:
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": 5,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    response.raise_for_status()
    return response.json()


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestOnlineServingOnDevice:
    def test_single_request_returns_completion(self, online_server, model_name):
        result = _query(online_server, model_name, "warm up")
        assert result["choices"][0]["text"]

    def test_two_concurrent_requests_both_succeed(self, online_server, model_name):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(_query, online_server, model_name, "test prompt")
                for _ in range(2)
            ]
            for future in futures:
                assert future.result()["choices"][0]["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
