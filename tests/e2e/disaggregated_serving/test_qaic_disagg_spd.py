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

Migrated from the pre-4a18ee6, CLI-option-based test tree
(tests/test_qaic/disaggregated_serving/test_spd_disagg.py) onto this repo's
tests/e2e/ infra: qaic_test_config marker + device pool + the class-scoped
disagg_server fixture (tests/e2e/disaggregated_serving/conftest.py), which
now supports speculative_method/speculative_model/num_speculative_tokens/
kv_store_size marker kwargs. Also switched from the old pickled-OpenOrca
dataset (required HF_TOKEN) to the ShareGPT-based get_prompts/query_server
already used by test_qaic_disagg.py in this directory.

HARDWARE DEPENDENT: requires live QAIC devices; not run as part of the
non-hardware verification gate. See docs/qaic/disagg_spd_port.md and
docs/qaic/disagg_spd_hardware_e2e_findings.md for prior hardware run results
(under the old file locations) -- ngram/suffix pass generation but fail
data_consistency due to a known QAIC compiler row-independence bug;
draft_model passes after the Parts A-D fixes documented there.

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


# ---------------------------------------------------------------------------
# Tests: ngram SpD + disaggregated serving
# ---------------------------------------------------------------------------


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(speculative_method="ngram", **_BASE_CONFIG)
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
@pytest.mark.qaic_test_config(speculative_method="suffix", **_BASE_CONFIG)
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
@pytest.mark.qaic_test_config(speculative_method="ngram", **_BASE_CONFIG)
class TestKvProducerSpDGuards:
    """
    Validate that the KV producer (prefill node) does not execute SpD logic.

    These tests verify indirectly that guards are working by confirming:
    - The server starts successfully (guard prevents assertion failures)
    - Requests complete correctly (no draft token corruption on prefill)

    The actual unit-level guard verification is in
    tests/test_qaic/disaggregated_serving/test_disagg_spd_guards.py.
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

    def test_basic_request_succeeds(self, disagg_server):
        """A basic request should succeed, confirming guards don't break flow."""
        config = disagg_server
        port = config["disaggregated_server_port"]
        timeout = config["client_request_timeout"]

        prompts = get_prompts("/v1/completions", ctx_len=config["ctx_len"])[:3]
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


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(speculative_method="ngram", **_BASE_CONFIG)
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
    def test_ngram_acceptance_rate(self, disagg_server):
        config = disagg_server
        port = config["disaggregated_server_port"]
        decode_port = config["decode_worker_ports"]
        timeout = config["client_request_timeout"]

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


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(speculative_method="suffix", **_BASE_CONFIG)
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
    def test_suffix_acceptance_rate(self, disagg_server):
        config = disagg_server
        port = config["disaggregated_server_port"]
        decode_port = config["decode_worker_ports"]
        timeout = config["client_request_timeout"]

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


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    speculative_method="draft_model",
    speculative_model=DEFAULT_DRAFT_MODEL_NAME,
    **_BASE_CONFIG,
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
    def test_draft_model_acceptance_rate(self, disagg_server):
        config = disagg_server
        port = config["disaggregated_server_port"]
        decode_port = config["decode_worker_ports"]
        timeout = config["client_request_timeout"]

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
