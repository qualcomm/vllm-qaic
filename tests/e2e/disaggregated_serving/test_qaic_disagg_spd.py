# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
E2E tests for speculative decoding (SpD) + disaggregated serving on QAIC.

These tests verify that disaggregated serving works correctly with ngram,
suffix, and draft_model SpD methods. They launch the qaic_disagg server with
SpD enabled on the decode instances and validate:

1. Output is generated correctly (non-empty, valid responses).
2. Data consistency is maintained across shuffled prompt orders.
3. All SpD methods (ngram, suffix, draft_model) produce correct results.

HARDWARE DEPENDENT: requires live QAIC devices and is not part of the
non-hardware verification gate.

Run examples:
    pytest tests/e2e/disaggregated_serving/test_qaic_disagg_spd.py -v \
        --device-id "0,1,2,3,4,5,6,7"
"""

import itertools
import logging
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from .utils import get_prompts, query_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_FAST_MODEL_NAME = "TinyLlama/TinyLlama_v1.1"
DEFAULT_SEQ_LEN = 512
DEFAULT_CTX_LEN = 12288
DEFAULT_DECODE_BSZ = 2
DEFAULT_NUM_SPEC_TOKENS = 3
# The kv-handoff shared-memory store defaults to 64 (MB), too small for an 8B
# model's KV buffer -- the kv_consumer then fails at the first request with
# "Shared memory buffer not found". 768 matches known-good disagg
# deployments for this model size.
DEFAULT_KV_STORE_SIZE = 768

_BASE_CONFIG = dict(
    model_name=DEFAULT_MODEL_NAME,
    ctx_len=DEFAULT_CTX_LEN,
    seq_len=DEFAULT_SEQ_LEN,
    decode_bsz=DEFAULT_DECODE_BSZ,
    num_prefill_workers=1,
    prefill_device_group_size=1,
    num_decode_workers=1,
    decode_device_group_size=1,
    prefill_max_num_seqs=2,
    num_speculative_tokens=DEFAULT_NUM_SPEC_TOKENS,
    kv_store_size=DEFAULT_KV_STORE_SIZE,
)

_FAST_BASE_CONFIG = {
    **_BASE_CONFIG,
    "model_name": DEFAULT_FAST_MODEL_NAME,
    "ctx_len": 2048,
}


# ---------------------------------------------------------------------------
# Tests: ngram SpD + disaggregated serving
# ---------------------------------------------------------------------------


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(speculative_method="ngram", **_FAST_BASE_CONFIG)
class TestNgramDisagg:
    """Verify ngram SpD works correctly with disaggregated serving."""

    def test_ngram_data_consistency(self, disagg_server):
        """Same prompts in different order produce the same output."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:5]

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

    def test_ngram_generation_non_empty(self, disagg_server):
        """Verify ngram SpD generates non-empty output for each request."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:10]
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


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(speculative_method="suffix", **_FAST_BASE_CONFIG)
class TestSuffixDisagg:
    """Verify suffix SpD works correctly with disaggregated serving."""

    def test_suffix_data_consistency(self, disagg_server):
        """Same prompts in different order produce the same output."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:5]

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

    def test_suffix_generation_non_empty(self, disagg_server):
        """Verify suffix SpD generates non-empty output for each request."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:10]
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


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    speculative_method="draft_model",
    speculative_model=DEFAULT_DRAFT_MODEL_NAME,
    **_BASE_CONFIG,
)
class TestDraftModelDisagg:
    """Verify LLM-as-draft-model SpD works with disaggregated serving."""

    def test_draft_model_data_consistency(self, disagg_server):
        """Same prompts in different order produce the same output."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:5]

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

    def test_draft_model_generation_non_empty(self, disagg_server):
        """Verify draft_model SpD generates non-empty output."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:10]
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

    def test_draft_model_parallel_requests(self, disagg_server):
        """Send concurrent requests to verify SpD under load."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = 300
        num_requests = 20

        prompts = list(
            itertools.islice(
                itertools.cycle(
                    get_prompts("/v1/completions", ctx_len=config["ctx_len"])
                ),
                num_requests,
            )
        )

        max_workers = config["constant_factor"]

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


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(speculative_method="ngram", **_FAST_BASE_CONFIG)
class TestKvProducerSpDGuards:
    """
    Validate that the KV producer (prefill node) does not execute SpD logic.

    This test verifies indirectly that guards are working by confirming the
    server starts successfully without producer-side speculative-decoding
    assertion failures.

    The actual unit-level guard verification is in
    tests/test_qaic/disaggregated_serving/test_disagg_spd_unit.py.
    """

    def test_server_starts_with_spd_config(self, disagg_server):
        """
        Server must start without assertion errors.

        If the KV producer guard is missing, QaicDraftModelProposer or the
        drafter proposal logic would fire on the producer and likely crash.
        """
        config = disagg_server
        assert config["process"].poll() is None, (
            "Server should still be running (guards prevent producer crash)"
        )
