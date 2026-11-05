# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import json
import socket

import regex as re
import requests

from vllm.utils.network_utils import is_valid_ipv6_address

from ..conftest import _ensure_sharegpt_dataset


def get_prompts(
    api_endpoint,
    ctx_len=None,
    max_tokens=0,
    chars_per_token=3.0,
    truncate_at=100,
    total_max_chars=250,
):
    """Loads the prompt pool for `api_endpoint`. When `ctx_len` is given,
    drops prompts whose estimated token length (a conservative
    `chars_per_token` chars-per-token lower bound, so we never underestimate)
    plus `max_tokens` would exceed `ctx_len` for the model under test."""
    conversations = get_sharegpt_conversations(
        truncate_at=truncate_at, total_max_chars=total_max_chars
    )
    if api_endpoint == "/v1/completions":
        prompts = [conversation[0]["content"] for conversation in conversations]
    elif api_endpoint == "/v1/chat/completions":
        prompts = conversations

    if ctx_len is not None:
        max_chars = (ctx_len - max_tokens) * chars_per_token
        prompts = [p for p in prompts if _prompt_char_len(p, api_endpoint) <= max_chars]
    return prompts


def _prompt_char_len(prompt, api_endpoint) -> int:
    if api_endpoint == "/v1/chat/completions":
        return sum(len(message["content"]) for message in prompt)
    return len(prompt)


def get_sharegpt_conversations(truncate_at=100, total_max_chars=250):
    """Parses ShareGPT conversations, reusing the parent conftest's shared,
    concurrency-safe dataset download."""
    share_gpt_json = _ensure_sharegpt_dataset()

    role_mapping = {"human": "user", "gpt": "assistant"}
    conversations = []
    ESTIMATED_MESSAGE_OVERHEAD = 50

    with open(share_gpt_json) as f:
        share_gpt_data = json.load(f)

    for data in share_gpt_data:
        current_total_content_length = 0
        conversation = []
        for entry in data["conversations"]:
            processed_content = entry["value"]
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
        if conversation:
            conversations.append(conversation)
    return conversations


def check_port_status(
    port: int,
    host: str = "localhost",
    timeout: int = 1,
    check_open: bool = True,
) -> bool:
    """Checks if a port is open or free on the given host."""
    try:
        for res in socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM):
            af, socktype, proto, canonname, sa = res
            with socket.socket(af, socktype, proto) as sock:
                sock.settimeout(timeout)
                try:
                    sock.connect(sa)
                    return check_open
                except (TimeoutError, ConnectionRefusedError):
                    pass
            return not check_open
    except (socket.gaierror, Exception):
        return False


def parse_stream(stream_text: str) -> list:
    """Parse SSE stream text into structured data: JSON packets are parsed
    into dicts, "[DONE]" markers are preserved as strings, and invalid
    packets are kept as-is for debugging."""
    data_packets = re.findall(r"data: (.*)", stream_text)
    parsed_packets = []
    for packet in data_packets:
        if packet == "[DONE]":
            parsed_packets.append(packet)
        else:
            try:
                parsed_packets.append(json.loads(packet))
            except json.JSONDecodeError:
                parsed_packets.append(packet)
    return parsed_packets


def query_server(
    user_input: str,
    max_tokens: int = None,
    temperature: float = 0,
    ignore_eos: bool = True,
    skip_special_tokens: bool = True,
    host: str = "0.0.0.0",
    port: str = "8081",
    api_endpoint="/v1/completions",
    headers: dict = None,
    stream=False,
    timeout=120,
) -> requests.Response:
    if is_valid_ipv6_address(host):
        host = f"[{host}]"
    url = f"http://{host}:{port}{api_endpoint}"

    data_json = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "stream": stream,
    }
    if not skip_special_tokens:
        data_json["skip_special_tokens"] = False
    if api_endpoint == "/v1/chat/completions":
        data_json["messages"] = user_input
    elif api_endpoint == "/v1/completions":
        data_json["prompt"] = user_input
    return requests.post(
        url,
        json=data_json,
        headers=headers,
        timeout=timeout if timeout else 120,
    )
