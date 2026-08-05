# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Non-disagg ngram SpD order-independence check on QAIC AOT.

Ported from vllm_0_15_0's tests/test_qaic/spec_decode/e2e/test_ngram_integration.py
::test_ngram_data_consistency_order_independence (commit 2502833607's
post-commit state). Uses the exact same dataset slice, prompt count, and SpD
config as
tests/test_qaic/disaggregated_serving/test_spd_disagg.py::TestNgramDisagg
.test_ngram_data_consistency, so a non-disagg AOT run can be directly
compared against that disagg test's known-failing hardware runs, to
determine whether shuffled-prompt-order divergence is disagg-specific or a
general property of ngram SpD outside disagg.

HARDWARE DEPENDENT: not run as part of the non-hardware verification gate for
this port. Requires tests/test_qaic/dataset/<dataset>.pkl, which does not yet
exist in vllm-qaic -- copy it from vllm_0_15_0's tests/test_qaic/dataset/
before running (see docs/qaic/disagg_spd_port.md).

Split into its own file (not appended to test_ngram_suffix_integration.py)
because that file is PYT-eager-mode-only (module-level
pytestmark = pytest.mark.skipif(QaicPlatform.is_aot, ...)), while this test
needs AOT-only concepts (quantization="mxfp6", kv_cache_dtype="mxint8").

Run::

    .venv/bin/python -m pytest -s \
        tests/test_qaic/spec_decode/e2e/test_ngram_order_independence_aot.py \
        --test-device-group '[0]' -v
"""

import gc
import pickle
import random
import time
from pathlib import Path

import pytest

from vllm import LLM, SamplingParams
from vllm_qaic.platform_base import QaicPlatform

# This validates the AOT path; skip in PYT/eager mode.
pytestmark = pytest.mark.skipif(
    not QaicPlatform.is_aot,
    reason="AOT-mode ngram SpD order-independence test; requires AOT (no torch_qaic).",
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Same dataset/prompt-count/SpD config as
# tests/test_qaic/disaggregated_serving/test_spd_disagg.py::TestNgramDisagg
# .test_ngram_data_consistency.
_ORDER_INDEPENDENCE_DATASET = (
    Path(__file__).parent.parent.parent
    / "dataset"
    / "open_orca_gpt4_tokenized_llama3.sampled_1024new_truncated_prompts_to_128.pkl"
)
_ORDER_INDEPENDENCE_NUM_PROMPTS = 5
_ORDER_INDEPENDENCE_NUM_SPEC_TOKENS = 3
_ORDER_INDEPENDENCE_MAX_NUM_SEQS = 2  # DEFAULT_DECODE_BSZ in test_spd_disagg.py
_ORDER_INDEPENDENCE_MAX_TOKENS = 50


@pytest.fixture(scope="session", autouse=True)
def _qaic_visible_devices(device_group):
    """Export QAIC_VISIBLE_DEVICES from the requested device group.

    Matches the convention in test_async_spec_aot.py -- QaicPlatform.
    check_and_update_config only reads device_group into os.environ when
    QAIC_VISIBLE_DEVICES is unset, so no additional_config is needed.
    """
    import os  # noqa: PLC0415

    key = QaicPlatform.device_control_env_var  # "QAIC_VISIBLE_DEVICES"
    prev = os.environ.get(key)
    os.environ[key] = ",".join(str(q) for q in device_group)
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def test_ngram_data_consistency_order_independence():
    """Same prompts in different order should produce the same per-prompt
    output. Non-disagg counterpart to
    tests/test_qaic/disaggregated_serving/test_spd_disagg.py::TestNgramDisagg
    .test_ngram_data_consistency, which reproducibly failed this invariant on
    hardware. Used to determine whether that failure is disagg-specific or a
    general property of ngram SpD outside disagg.
    """
    with open(_ORDER_INDEPENDENCE_DATASET, "rb") as f:
        dataset = pickle.load(f)
    prompts = dataset["input"].tolist()[:_ORDER_INDEPENDENCE_NUM_PROMPTS]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=_ORDER_INDEPENDENCE_MAX_TOKENS,
    )

    llm = LLM(
        model=MODEL,
        max_num_seqs=_ORDER_INDEPENDENCE_MAX_NUM_SEQS,
        max_model_len=256,
        long_prefill_token_threshold=128,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        disable_log_stats=False,
        gpu_memory_utilization=1.0,
        async_scheduling=False,
        enable_prefix_caching=False,
        speculative_config={
            "model": "ngram",
            "num_speculative_tokens": _ORDER_INDEPENDENCE_NUM_SPEC_TOKENS,
        },
    )

    try:

        def get_responses(prompt_list):
            outputs = llm.generate(prompt_list, sampling_params, use_tqdm=True)
            return {
                p: o.outputs[0].text for p, o in zip(prompt_list, outputs, strict=False)
            }

        initial = get_responses(prompts)

        for i in range(3):
            shuffled = prompts[:]
            random.shuffle(shuffled)
            shuffled_map = get_responses(shuffled)
            assert shuffled_map == initial, (
                f"ngram non-disagg data consistency failed on shuffle {i}"
            )
    finally:
        llm = None
        gc.collect()
        time.sleep(10)
