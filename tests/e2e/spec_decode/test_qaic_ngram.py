# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""N-gram / suffix speculative-decoding end-to-end regressions (AOT and eager)."""

import pytest
from vllm import SamplingParams
from vllm.platforms import current_platform

_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_CTX_LEN = 256
_SEQ_LEN = 128
_NUM_SPEC_TOKENS = 4

# Per-mode runner kwargs. AOT compiles an mxfp6 QPC over a 4-QID group; eager
# runs fp16 on one QID (the native paged-attention kernel requires fp16).
_MODE_RUNNER_KWARGS = {
    "aot": dict(
        decode_bsz=4,
        quantization="mxfp6",
        kv_dtype="mxint8",
        override_qaic_config={
            "num_cores": 8,
            "prefill_seq_len": _SEQ_LEN,
            "mxfp6_matmul": True,
        },
        device_group_size=4,
    ),
    "eager": dict(
        # Room for every prompt in one batch, so verify rows, single-token
        # decode rows and prefills share steps.
        decode_bsz=8,
        dtype="float16",
        device_group_size=1,
    ),
}

_SPEC_CONFIGS = {
    "ngram": {
        "method": "ngram",
        "num_speculative_tokens": _NUM_SPEC_TOKENS,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 4,
    },
    "suffix": {
        "method": "suffix",
        "num_speculative_tokens": _NUM_SPEC_TOKENS,
    },
}

# Mixed lengths. Repetitive prompts give the proposers matches; the ones whose
# continuation breaks the pattern force rejections (and KV/state rollback), and
# the short non-repetitive prompt yields zero-draft rows in the same batch.
_PROMPTS = [
    "The capital of France is Paris. The capital of France is",
    "Two plus two equals four. Two plus two equals",
    "The opposite of hot is cold. The opposite of hot is",
    "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
    "Count: one two three, one two three, one two four, one two",
    "My name is",
]
_MAX_TOKENS = 24


def _mode() -> str:
    return "aot" if current_platform.is_aot_inference() else "eager"


@pytest.fixture(autouse=True)
def _spawn_engine_core_in_eager(monkeypatch):
    """Each test builds several LLMs on the same QID in sequence. In eager mode
    the frontend's `import torch` opens the QAIC device; a forked EngineCore
    inherits that handle, and the NSPs it claims stay reserved until the
    frontend exits, so the next LLM fails with "QID: N has 0 free nsps".
    A spawned EngineCore owns its own handle and releases the NSPs on exit."""
    if _mode() == "eager":
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _counter_total(metrics, name: str) -> int:
    return sum(metric.value for metric in metrics if metric.name == name)


def _runner(factory, speculative_config=None, **kwargs):
    return factory(
        _MODEL,
        speculative_config=speculative_config,
        ctx_len=_CTX_LEN,
        seq_len=_SEQ_LEN,
        **_MODE_RUNNER_KWARGS[_mode()],
        **kwargs,
    )


def _spec_counter_deltas(before, after) -> tuple[int, int]:
    drafted, accepted = (
        _counter_total(after, name) - _counter_total(before, name)
        for name in (
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
        )
    )
    return drafted, accepted


# TODO: re-enable on AOT once the Numba rejection-sampler install targets the
# vllm 0.30.0 kernel dispatchers (`_rejection_greedy_sample` etc.): rebinding
# the `*_kernel` globals no longer takes effect, so the first verify step fails
# with "Cannot inspect kernel parameters".
@pytest.mark.skipif(
    current_platform.is_aot_inference(),
    reason="AOT Numba rejection sampler is not wired to vllm 0.30.0 dispatchers",
)
@pytest.mark.qaic_test_config(
    {
        "aot": dict(model_name=_MODEL, num_device_groups=1, device_group_size=4),
        "eager": dict(model_name=_MODEL, num_device_groups=1, device_group_size=1),
    }
)
@pytest.mark.parametrize(
    "method",
    ["ngram", pytest.param("suffix", marks=pytest.mark.arctic_inference_installed)],
)
class TestQaicNgram:
    def test_greedy_matches_nospec_and_records_draft_acceptance(
        self, qaic_runner_factory, method
    ):
        """SpD must preserve greedy token identities, and must both accept and
        reject drafts so the accept/bonus and rollback paths both run."""
        with _runner(qaic_runner_factory) as base_model:
            base = base_model.generate_greedy(_PROMPTS, _MAX_TOKENS)

        with _runner(
            qaic_runner_factory,
            speculative_config=_SPEC_CONFIGS[method],
            disable_log_stats=False,
        ) as spec_model:
            before = spec_model.llm.get_metrics()
            spec = spec_model.generate_greedy(_PROMPTS, _MAX_TOKENS)
            after = spec_model.llm.get_metrics()

        for index, ((spec_ids, _), (base_ids, _)) in enumerate(
            zip(spec, base, strict=True)
        ):
            assert spec_ids == base_ids, (
                f"{method} SpD diverged for prompt {index}:\n"
                f"base={base_ids}\nspec={spec_ids}"
            )
        drafted, accepted = _spec_counter_deltas(before, after)
        assert drafted > 0, f"{method} did not draft any speculative tokens"
        assert accepted > 0, f"{method} drafted but accepted no speculative tokens"
        assert accepted < drafted, (
            f"{method} accepted every draft ({accepted}/{drafted}); rejection "
            "and rollback were not exercised"
        )

    def test_random_sampling_runs(self, qaic_runner_factory, method):
        """Temperature>0 SpD runs the random-rejection and recovered-token
        kernels. Sampled output has no token-exact baseline, so this checks
        that generation completes with well-formed, drafted output."""
        sampling_params = SamplingParams(
            temperature=0.8, max_tokens=_MAX_TOKENS, seed=42
        )
        with _runner(
            qaic_runner_factory,
            speculative_config=_SPEC_CONFIGS[method],
            disable_log_stats=False,
        ) as spec_model:
            before = spec_model.llm.get_metrics()
            outputs = spec_model.llm.generate(_PROMPTS, sampling_params)
            after = spec_model.llm.get_metrics()
            vocab_size = spec_model.llm.model_config.get_vocab_size()

        drafted, _ = _spec_counter_deltas(before, after)
        assert drafted > 0, f"{method} did not draft any speculative tokens"
        assert len(outputs) == len(_PROMPTS)
        for index, output in enumerate(outputs):
            token_ids = output.outputs[0].token_ids
            assert token_ids, f"{method} prompt {index} produced no tokens"
            assert all(0 <= token < vocab_size for token in token_ids), (
                f"{method} prompt {index} produced out-of-vocab tokens: {token_ids}"
            )
