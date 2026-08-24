# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device unit tests for QAIC disaggregated serving (prefill/decode split).

Pure-Python coverage of the disaggregated-serving config constraints
(LoRA/disagg mutual exclusion, unsupported SpD types, prefix-caching
auto-disable, "stages" pipeline-parallel assertion) lives in
unit/generic/test_platform.py::TestDisaggregatedServingConstraints. The
eager-mode rejection of kv_transfer_config entirely (disagg is AOT-only) is
covered on-device in test_unit_qaic_eager_mode.py.

Heavier scenarios — data consistency under request reordering, concurrent
load at scale, streaming vs. non-streaming latency comparison — live in the
e2e suite (disaggregated_serving/test_qaic_disagg.py,
disaggregated_serving/test_prefill_streaming.py). This file is the missing
on-device-unit layer in between: confirms the smallest possible
disaggregated-serving deployment (1 prefill worker, 1 decode worker) actually
serves a real request end-to-end through the real `qaic_disagg` server.

Coverage areas
--------------
1. A single request against a 1-prefill/1-decode disaggregated deployment
   returns a well-formed, non-empty completion
2. Two sequential requests for the same prompt produce identical output
   (disaggregated serving must not introduce nondeterminism at temperature=0)
"""

import pytest

from .utils import get_prompts, query_server

pytestmark = pytest.mark.qaic_disagg_installed


@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ctx_len=256,
    seq_len=64,
    dtype="mxfp6",
    decode_bsz=2,
    num_prefill_workers=1,
    prefill_device_group_size=1,
    num_decode_workers=1,
    decode_device_group_size=1,
    prefill_max_num_seqs=1,
)
class TestDisaggregatedServingOnDevice:
    def test_single_request_returns_completion(self, disagg_server):
        test_config = disagg_server
        prompts = get_prompts(
            "/v1/completions", ctx_len=test_config["ctx_len"], max_tokens=16
        )[:1]
        response = query_server(
            prompts,
            max_tokens=16,
            api_endpoint="/v1/completions",
            port=test_config["disaggregated_server_port"],
            timeout=test_config["client_request_timeout"] * 5,
        )
        assert response.status_code == 200, (
            f"Request failed with status {response.status_code}: {response.text}"
        )
        choices = response.json()["choices"]
        assert len(choices) == len(prompts)
        assert len(choices[0]["text"]) > 0

    def test_repeated_request_is_deterministic(self, disagg_server):
        """temperature=0 through disaggregated serving must be deterministic."""
        test_config = disagg_server
        prompts = get_prompts(
            "/v1/completions", ctx_len=test_config["ctx_len"], max_tokens=16
        )[:1]

        def _get_text():
            response = query_server(
                prompts,
                max_tokens=16,
                api_endpoint="/v1/completions",
                port=test_config["disaggregated_server_port"],
                timeout=test_config["client_request_timeout"] * 5,
            )
            assert response.status_code == 200
            return response.json()["choices"][0]["text"]

        assert _get_text() == _get_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
