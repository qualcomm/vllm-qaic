# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""AOT n-gram speculative-decoding end-to-end regressions."""

import pytest


_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_OVERRIDE = {"num_cores": 8, "prefill_seq_len": 128, "mxfp6_matmul": True}


def _counter_total(metrics, name: str) -> int:
    return sum(metric.value for metric in metrics if metric.name == name)


@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    model_name=_MODEL,
    ctx_len=256,
    seq_len=128,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=4,
)
class TestQaicNgram:
    def test_greedy_matches_nospec_and_records_draft_acceptance(
        self, qaic_runner_factory
    ):
        """N-gram must affect execution and preserve greedy token identities."""
        prompts = [
            "The capital of France is Paris. The capital of France is",
            "Two plus two equals four. Two plus two equals",
            "The opposite of hot is cold. The opposite of hot is",
        ]
        max_tokens = 16

        with qaic_runner_factory(
            _MODEL,
            ctx_len=256,
            seq_len=128,
            quantization="mxfp6",
            kv_dtype="mxint8",
            override_qaic_config=_OVERRIDE,
            device_group_size=4,
        ) as base_model:
            base = base_model.generate_greedy(prompts, max_tokens)

        with qaic_runner_factory(
            _MODEL,
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 4,
                "prompt_lookup_min": 1,
                "prompt_lookup_max": 4,
            },
            ctx_len=256,
            seq_len=128,
            quantization="mxfp6",
            kv_dtype="mxint8",
            override_qaic_config=_OVERRIDE,
            device_group_size=4,
            disable_log_stats=False,
        ) as ngram_model:
            before = ngram_model.llm.get_metrics()
            spec = ngram_model.generate_greedy(prompts, max_tokens)
            after = ngram_model.llm.get_metrics()

        assert [ids for ids, _ in spec] == [ids for ids, _ in base]
        drafted = _counter_total(
            after, "vllm:spec_decode_num_draft_tokens"
        ) - _counter_total(before, "vllm:spec_decode_num_draft_tokens")
        accepted = _counter_total(
            after, "vllm:spec_decode_num_accepted_tokens"
        ) - _counter_total(before, "vllm:spec_decode_num_accepted_tokens")
        assert drafted > 0, "ngram path did not draft any speculative tokens"
        assert accepted > 0, "ngram path drafted but accepted no speculative tokens"
