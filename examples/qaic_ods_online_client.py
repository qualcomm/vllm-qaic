# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from examples/openai_chat_completion_client.py

"""Online QAIC on-device sampling (ODS) example client using vLLM serve.

NOTE: start exactly ONE server configuration at a time on a given port, then
run this client script against that server.

BASELINE:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 4 \
    --long-prefill-token-threshold 128 \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --no-enable-prefix-caching \
    --additional-config '{"override_qaic_config":{"aic_include_sampler":1}}' \
    --no-async-scheduling

MAX_TOP_K_IDS LIMIT:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 4 \
    --long-prefill-token-threshold 128 \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --no-enable-prefix-caching \
    --additional-config '{"override_qaic_config":{"aic_include_sampler":1,"max_top_k_ids":64}}' \
    --no-async-scheduling

DEBUG SUB-MODE:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 4 \
    --long-prefill-token-threshold 128 \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --no-enable-prefix-caching \
    --additional-config '{"override_qaic_config":{"aic_include_sampler":1,"aic_return_pdfs":1}}' \
    --no-async-scheduling

ASYNC SCHEDULING:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 4 \
    --long-prefill-token-threshold 128 \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --no-enable-prefix-caching \
    --additional-config '{"override_qaic_config":{"aic_include_sampler":1}}' \
    --async-scheduling

NOTE: If the server startup log shows "QAIC currently does not support async
scheduling; Falling back to non-async scheduling.", the server silently
downgraded to sync and the async path was NOT actually exercised. Check
server startup logs for this exact message before treating any client-side
PASS/FAIL result below as an async-scheduling validation -- a PASS in that
case only re-confirms the already-validated SYNC path, not async.

Equivalent OpenAI-compatible curl requests:

BASELINE:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize why ODS helps QAIC serving."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 64
    }'

MAX_TOP_K_IDS LIMIT (in-limit top_k):
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "user", "content": "Use an in-limit top_k request."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 32,
        "max_tokens": 64
    }'

MAX_TOP_K_IDS LIMIT (over-limit top_k, expected rejection when max_top_k_ids
is configured):
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "user", "content": "This request should exceed top_k limit."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 999,
        "max_tokens": 64
    }'

DEBUG SUB-MODE:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "user", "content": "Run one request in ODS debug sub-mode."}
        ],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": 64
    }'

ASYNC SCHEDULING (baseline-style request; only meaningful if the server
above logged genuine async-scheduling activation, not a fallback):
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize why ODS helps QAIC serving."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_tokens": 64
    }'
"""

import argparse

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "Explain one practical reason on-device sampling can reduce "
            "host-side overhead in QAIC serving."
        ),
    },
]

over_limit_messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "This request intentionally exceeds top_k limits.",
    }
]

async_scheduling_messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": (
            "Explain one practical reason on-device sampling can reduce "
            "host-side overhead in QAIC serving."
        ),
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    print("=" * 60)
    print("ODS ONLINE EXAMPLE: MODEL DISCOVERY")
    print("=" * 60)
    try:
        model_list = client.models.list()
    except Exception as err:
        print(f"[FAIL] Unable to list models from server: {err}")
        return

    if not model_list.data:
        print("[FAIL] Server returned no models.")
        return

    model = model_list.data[0].id
    print(f"[PASS] Connected to server and discovered model: {model}")

    print("=" * 60)
    print("ODS ONLINE EXAMPLE: BASELINE REQUEST")
    print("=" * 60)
    try:
        baseline_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.7,
            top_p=0.9,
            max_tokens=64,
            stream=args.stream,
            extra_body={"top_k": 40},
        )
        print("[PASS] Baseline chat completion request succeeded.")
        if args.stream:
            for completion_chunk in baseline_completion:
                print(completion_chunk)
        else:
            print(baseline_completion)
    except Exception as err:
        print(f"[FAIL] Baseline chat completion request failed: {err}")

    print("=" * 60)
    print("ODS ONLINE EXAMPLE: max_top_k_ids LIMIT CHECK")
    print("=" * 60)
    print(
        "NOTE: This over-limit top_k check is only expected to be rejected when "
        "the server was started with override_qaic_config.max_top_k_ids (for "
        "example, max_top_k_ids=64)."
    )
    try:
        client.chat.completions.create(
            messages=over_limit_messages,
            model=model,
            temperature=0.7,
            top_p=0.9,
            max_tokens=64,
            stream=False,
            extra_body={"top_k": 999},
        )
        print(
            "[FAIL] Over-limit top_k request was not rejected. "
            "If max_top_k_ids is not configured on the running server, this "
            "result is expected."
        )
    except Exception as err:
        print(f"[PASS] Over-limit top_k request was rejected as expected: {err}")

    print("=" * 60)
    print("ODS ONLINE EXAMPLE: ASYNC SCHEDULING (baseline-style request)")
    print("=" * 60)
    print(
        "NOTE: This result is only a genuine async-scheduling validation if the "
        "server's startup log did NOT show \"QAIC currently does not support "
        "async scheduling; Falling back to non-async scheduling.\" Check server "
        "startup logs before interpreting the result below."
    )
    try:
        async_scheduling_completion = client.chat.completions.create(
            messages=async_scheduling_messages,
            model=model,
            temperature=0.7,
            top_p=0.9,
            max_tokens=64,
            stream=args.stream,
            extra_body={"top_k": 40},
        )
        print("[PASS] Async-scheduling-configuration chat completion request succeeded.")
        if args.stream:
            for completion_chunk in async_scheduling_completion:
                print(completion_chunk)
        else:
            print(async_scheduling_completion)
    except Exception as err:
        print(f"[FAIL] Async-scheduling-configuration chat completion request failed: {err}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
