# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
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

_RUNNER_SCRIPT = Path(__file__).with_name("_run_ngram_suffix_generation.py")


def _run_generation(method: str, qid: int) -> list[list[int]]:
    env = os.environ.copy()
    env[QaicPlatform.device_control_env_var] = str(qid)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "token_ids.json"
        subprocess.run(
            [sys.executable, str(_RUNNER_SCRIPT), method, str(output_path)],
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
