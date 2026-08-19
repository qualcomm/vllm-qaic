"""Eager n-gram and suffix speculative-decoding correctness tests.

Run with visible subprocess output::

    .venv_eager/bin/python -m pytest -s -v \
        tests/test_qaic/spec_decode/e2e/test_ngram_suffix_integration.py \
        --test-device-group '[8,9]'
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from vllm_qaic.platform_base import QaicPlatform

pytestmark = pytest.mark.skipif(
    QaicPlatform.is_aot,
    reason="Requires torch_qaic and eager PYT mode.",
)

_RUN_SCRIPT = r"""
import json
import sys

from vllm import LLM, SamplingParams

method = sys.argv[1]
output_path = sys.argv[2]
sampling_params = SamplingParams(temperature=0.0, max_tokens=32, seed=42)
kwargs = dict(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=16,
    max_model_len=256,
    long_prefill_token_threshold=128,
    tensor_parallel_size=1,
    enforce_eager=True,
    async_scheduling=False,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.9,
)
if method != "baseline":
    kwargs["speculative_config"] = {
        "num_speculative_tokens": 5,
        "method": method,
    }

llm = LLM(**kwargs)
try:
    outputs = llm.generate(
        [
            "The cat sat on the mat. The cat sat on the mat. The cat sat on the",
            "My name is",
        ],
        sampling_params,
    )
    token_ids = [list(output.outputs[0].token_ids) for output in outputs]
    with open(output_path, "w") as output_file:
        json.dump(token_ids, output_file)
finally:
    del llm
"""


def _run_generation(method: str, qid: int) -> list[list[int]]:
    env = os.environ.copy()
    env[QaicPlatform.device_control_env_var] = str(qid)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "token_ids.json"
        subprocess.run(
            [sys.executable, "-c", _RUN_SCRIPT, method, str(output_path)],
            check=True,
            env=env,
        )
        return json.loads(output_path.read_text())


@pytest.fixture(scope="session")
def baseline_token_ids(device_group) -> list[list[int]]:
    if len(device_group) < 2:
        pytest.fail(
            "Eager speculative-decoding E2E requires two ready single-QID devices: "
            "baseline and speculative decoding."
        )
    return _run_generation("baseline", device_group[0])


@pytest.mark.parametrize("method", ["ngram", "suffix"])
def test_eager_speculative_decoding_matches_greedy_baseline(
    baseline_token_ids,
    device_group,
    method: str,
):
    qid = device_group[1]
    speculative_token_ids = _run_generation(method, qid)

    assert len(speculative_token_ids) == len(baseline_token_ids)
    for request_index, (speculative, baseline) in enumerate(
        zip(speculative_token_ids, baseline_token_ids, strict=False)
    ):
        assert baseline, f"baseline request {request_index} produced no tokens"
        assert speculative == baseline, (
            f"{method} SpD diverged for request {request_index}:\n"
            f"baseline={baseline}\n"
            f"speculative={speculative}"
        )
