# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/examples/online_serving/openai_chat_completion_client.py

"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
AOT:
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 16 \
    --long-prefill-token-threshold 128 \
    --quantization mxfp6 --kv-cache-dtype mxint8 \
    --no-enable-prefix-caching \
    --additional-config '{"device_group":[0]}' \
    --no-async-scheduling
eager mode:
export QAIC_VISIBLE_DEVICES=0
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 256 --max-num-seq 1 \
    --long-prefill-token-threshold 128 \
    --no-enable-prefix-caching \
    --no-async-scheduling
"""

import argparse

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020.",
    },
    {"role": "user", "content": "Where was it played?"},
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

    models = client.models.list()
    model = models.data[0].id

    # Chat Completion API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        stream=args.stream,
    )

    print("-" * 50)
    print("Chat completion results:")
    if args.stream:
        for c in chat_completion:
            print(c)
    else:
        print(chat_completion)
    print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
