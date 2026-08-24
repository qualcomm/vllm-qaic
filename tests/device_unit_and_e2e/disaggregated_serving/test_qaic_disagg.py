# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import itertools
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from tqdm import tqdm

from .utils import get_prompts, parse_stream, query_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=2048,
    seq_len=256,
    dtype="mxfp6",
    decode_bsz=8,
    num_prefill_workers=1,
    prefill_device_group_size=2,
    num_decode_workers=1,
    decode_device_group_size=2,
    prefill_max_num_seqs=2,
)
class TestDisaggregatedServingBasic:
    """Basic disaggregated-serving checks. All tests in this class share one
    `qaic_disagg` server (the `disagg_server` fixture, class-scoped) instead
    of each starting/stopping their own — server bring-up is expensive
    (model compile + device warm-up)."""

    def test_data_consistency(self, disagg_server):
        """Ensures the disaggregated server produces the same output for the
        same input, regardless of request order."""
        num_prompts = 50
        api_endpoint = "/v1/completions"
        max_tokens = 50
        test_config = disagg_server
        port = test_config["disaggregated_server_port"]
        timeout = test_config["client_request_timeout"]
        prompts = random.sample(
            get_prompts(
                api_endpoint, ctx_len=test_config["ctx_len"], max_tokens=max_tokens
            ),
            num_prompts,
        )
        print(f"test_data_consistency: using {len(prompts)} prompts")

        def get_responses_map(input_prompts):
            response = query_server(
                input_prompts,
                max_tokens=max_tokens,
                api_endpoint=api_endpoint,
                port=port,
                timeout=timeout * 5,
            )
            assert response.status_code == 200, (
                f"Request Failed with Status code: {response.status_code} - "
                f"{response.text}"
            )
            response_json = response.json()
            assert len(response_json["choices"]) == len(input_prompts), (
                f"Expected {len(input_prompts)} choices, got "
                f"{len(response_json['choices'])}"
            )
            return {
                tuple(p) if isinstance(p, list) else p: response_json["choices"][i][
                    "text"
                ]
                for i, p in enumerate(input_prompts)
            }

        initial_prompt_response_map = get_responses_map(prompts)

        for i in range(3):
            shuffled_prompts = prompts[:]
            random.shuffle(shuffled_prompts)
            shuffled_prompt_response_map = get_responses_map(shuffled_prompts)
            assert shuffled_prompt_response_map == initial_prompt_response_map, (
                f"Data consistency failed on shuffle {i}"
            )

    @pytest.mark.parametrize("num_requests", [50, 100])
    @pytest.mark.parametrize("concurrency_scale_factor", [2, 3])
    def test_parallel_requests(
        self, disagg_server, num_requests, concurrency_scale_factor
    ):
        """Sends multiple parallel requests to the server and verifies that
        all responses are non-empty."""
        test_config = disagg_server
        port = test_config["disaggregated_server_port"]
        timeout = 450  # Each req takes ~9s. 50 Reqs take ~450s
        max_tokens = 300
        candidate_prompts = get_prompts(
            "/v1/completions", ctx_len=test_config["ctx_len"], max_tokens=max_tokens
        )
        prompts = list(
            itertools.islice(itertools.cycle(candidate_prompts), num_requests)
        )
        # Using `itertools.cycle` ensures we don't run out of prompts if
        # `num_requests` is large.
        print(
            f"test_parallel_requests: using {len(prompts)} prompts "
            f"(cycled from {len(candidate_prompts)} candidates)"
        )

        # Note: --max-concurrency = 2 * "# of Decode instances x Decode batch size
        # + # of prefill instances x prefill batch size"
        max_workers = concurrency_scale_factor * test_config["constant_factor"]
        print(
            f"Running parallel requests with max_workers: {max_workers}",
            f"and timeout:{timeout}",
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    query_server,
                    prompt,
                    max_tokens=max_tokens,
                    port=port,
                    timeout=timeout,
                ): prompt
                for prompt in prompts
            }
            for future in tqdm(
                futures,
                desc=f"Processing parallel requests using max_workers={max_workers}",
            ):
                prompt = futures[future]
                try:
                    response = future.result()
                    response.raise_for_status()
                    response_json = response.json()
                    assert response_json["choices"][0].get("text"), (
                        "Expected non-empty output in successful response with API key."
                    )
                    generated_text = response_json["choices"][0].get("text")
                    assert generated_text, (
                        f"Generated text is empty or None for prompt: {prompt}"
                    )
                    assert isinstance(generated_text, str), (
                        f"Generated text is not a string for prompt: {prompt}"
                    )
                except requests.exceptions.HTTPError as e:
                    pytest.fail(f"Failed on request for prompt: {prompt}. Error: {e}")
                except Exception as e:
                    if num_requests == 50:
                        pytest.fail(
                            f"Failed Processing the Requests due to timeout: "
                            f"{timeout}: {e}"
                        )
                    elif num_requests == 100:
                        assert isinstance(e, ConnectionError), (
                            "Expected to Fail processing the requests due to timeout"
                        )

    @pytest.mark.parametrize("stream", [True, False])
    def test_generation_length_one(self, disagg_server, stream):
        """Checks if the model is able to generate a single token at a
        time, i.e. GL=1."""
        test_config = disagg_server
        port = test_config["disaggregated_server_port"]
        timeout = test_config["client_request_timeout"]
        num_prompts = 20
        prompts = random.sample(
            get_prompts(
                "/v1/completions", ctx_len=test_config["ctx_len"], max_tokens=1
            ),
            num_prompts,
        )
        print(f"test_generation_length_one: using {len(prompts)} prompts")
        max_workers = min(num_prompts, os.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    query_server,
                    prompt,
                    max_tokens=1,
                    stream=stream,
                    port=port,
                    timeout=timeout,
                    # Ensure non-empty strings are returned if
                    # generated token is EOS
                    skip_special_tokens=False,
                ): prompt
                for prompt in prompts
            }
            for future in tqdm(futures, desc=f"Testing GL=1 with stream={stream}"):
                prompt = futures[future]
                try:
                    response = future.result()
                    response.raise_for_status()
                    if stream:
                        parsed_packets = parse_stream(response.text)
                        assert len(parsed_packets) == 2, (
                            f"Expected exactly 2 packets (token + DONE), got "
                            f"{len(parsed_packets)}"
                        )
                        assert parsed_packets[-1] == "[DONE]", (
                            "Stream didn't end with [DONE]"
                        )
                        token_packet = parsed_packets[0]
                        assert isinstance(token_packet, dict), (
                            "Token packet not a JSON object"
                        )
                        assert "choices" in token_packet, (
                            "Token packet missing 'choices' field"
                        )
                        assert len(token_packet["choices"]) == 1, (
                            "Expected exactly 1 choice"
                        )

                        choice = token_packet["choices"][0]
                        assert "text" in choice, "Token packet missing 'text' field"
                        assert choice["text"], "Empty token generated"
                        assert choice["finish_reason"] == "length", (
                            f"Unexpected finish_reason {choice['finish_reason']} "
                            f"in token packet"
                        )
                    else:
                        response_json = response.json()
                        generated_text = response_json["choices"][0]["text"]
                        assert generated_text, (
                            f"Generated text is None or empty for prompt: {prompt}"
                        )
                        assert isinstance(generated_text, str), (
                            f"Generated text is not a string for prompt: {prompt}"
                        )
                        assert response_json["usage"]["completion_tokens"] == 1, (
                            f"Expected 1 completion token but got "
                            f"{response_json['usage']['completion_tokens']} for "
                            f"prompt: {prompt}"
                        )
                        assert (
                            response_json["usage"]["prompt_tokens"] + 1
                            == response_json["usage"]["total_tokens"]
                        ), f"Token count mismatch for prompt: {prompt}"
                except Exception as e:
                    pytest.fail(f"Request failed for prompt '{prompt[:50]}...': {e}")
