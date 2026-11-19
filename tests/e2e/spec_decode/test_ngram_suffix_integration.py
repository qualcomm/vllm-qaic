# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Eager n-gram and suffix speculative-decoding correctness tests.

Run with visible subprocess output::

    .venv_eager/bin/python -m pytest -s -v \
        tests/e2e/spec_decode/test_ngram_suffix_integration.py \
        --device-id 8,9
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from vllm_qaic.platform import QaicPlatform

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
        result = subprocess.run(
            [sys.executable, str(_RUNNER_SCRIPT), method, str(output_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            # subprocess.run(check=True) would raise CalledProcessError without
            # the child's stdout/stderr, turning a real engine failure into an
            # opaque exit-1. Surface the captured output so the actual QAIC/vLLM
            # traceback is visible in pytest.
            pytest.fail(
                f"child generation ({method}, qid={qid}) failed with exit "
                f"{result.returncode}.\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        return json.loads(output_path.read_text())


@pytest.mark.parametrize("method", ["ngram", "suffix"])
@pytest.mark.qaic_test_config(num_device_groups=2, device_group_size=1)
def test_eager_speculative_decoding_matches_greedy_baseline(
    device_groups,
    method: str,
):
    """Greedy SpD output must equal the non-speculative baseline.

    Needs two free single-QID devices: one for the baseline, one for SpD.
    """
    baseline_token_ids = _run_generation("baseline", device_groups[0][0])
    speculative_token_ids = _run_generation(method, device_groups[1][0])

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
