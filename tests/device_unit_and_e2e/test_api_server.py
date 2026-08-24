# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor

import pytest
import requests


@pytest.fixture
def api_server(
    server_runner,
    model_name,
    seq_len,
    ctx_len,
    decode_bsz,
    dtype,
    kv_dtype,
    device_group,
    override_qaic_config,
    host,
    port,
):
    additional_config = {"device_group": device_group}
    if override_qaic_config is not None:
        additional_config["override_qaic_config"] = override_qaic_config

    with server_runner(
        server_runner.Backend.VLLM_SERVE,
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


def _query_server(
    base_url: str, model_name: str, prompt: str, max_tokens: int = 5
) -> dict:
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
    )
    response.raise_for_status()
    return response.json()


def _stream_and_cancel(base_url: str, model_name: str, max_tokens: int) -> None:
    with (
        requests.Session() as session,
        session.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": "canceled requests",
                "max_tokens": max_tokens,
                "ignore_eos": True,
                "stream": True,
            },
            stream=True,
        ) as response,
    ):
        # Read at least one chunk so the request is confirmed in flight
        # before we disconnect - closing before anything is scheduled
        # would abort it without ever exercising the running-request path.
        next(response.iter_content(chunk_size=1), None)


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=512,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_api_server(api_server, model_name, ctx_len):
    """
    Run the API server and test it.

    We test that the server can handle incoming requests, including
    multiple requests at the same time, and that it can handle requests
    being cancelled without crashing.
    """
    base_url = api_server

    # Warm up / single prompt
    result = _query_server(base_url, model_name, "warm up")
    assert result

    # Try with 100 concurrent prompts
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(_query_server, base_url, model_name, "test prompt")
            for _ in range(100)
        ]
        for future in futures:
            assert future.result()

    # Cancel requests: disconnect mid-stream and confirm the server keeps
    # running instead of erroring out (vLLM records client-aborted
    # requests via a scheduler-cleanup path that bypasses the iteration
    # stats/metrics counters, so there's no counter to assert on here).
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(_stream_and_cancel, base_url, model_name, ctx_len - 8)
            for _ in range(20)
        ]
        for future in futures:
            future.result()

    # check that server still runs after cancellations
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                _query_server, base_url, model_name, "test prompt after canceled"
            )
            for _ in range(100)
        ]
        for future in futures:
            assert future.result()
