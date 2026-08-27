# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import asyncio

import openai  # use the official client for correctness check
import pytest


@pytest.fixture
def server(
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
        server_runner.Backend.OPENAI_API_SERVER_MODULE,
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


async def _client(base_url: str, model_name: str, sample_prompts: list[str]):
    client = openai.AsyncOpenAI(base_url=f"{base_url}/v1", api_key="token-abc123")
    prompts = sample_prompts[:2]
    response = await client.completions.create(
        model=model_name, prompt=prompts, max_tokens=5, temperature=0.0
    )
    assert response.id is not None
    assert response.choices is not None and len(response.choices) == len(prompts)


@pytest.mark.asyncio
@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
async def test_multiple_client(server, model_name, sample_prompts):
    await asyncio.gather(
        _client(server, model_name, sample_prompts),
        _client(server, model_name, sample_prompts),
    )
