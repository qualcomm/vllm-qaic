# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC ngram speculative decoding (SpD/PLD).

SpD is AOT-only (vllm_qaic.platform_base.QaicPlatform.check_and_update_config
raises ValueError for speculative_config in eager mode — see
unit/generic/test_platform.py::TestEagerModeBranches::
test_speculative_config_raises_value_error_in_eager for the pure-Python
coverage of that rejection). These tests exercise the AOT success path with
a real runner and are the on-device counterpart to the two tests removed
from unit/spd/test_spec_decode_unit.py's TestDraftModelConfigExtraction and
TestSpDConfigValidation (test_ngram_spd_output_non_empty and
test_ngram_spd_matches_baseline) when those were extracted out of unit/.

Coverage areas
--------------
1. ngram SpD produces non-empty output
2. ngram SpD greedy output matches the non-SpD greedy baseline
"""

import pytest

from vllm_qaic.platform_base import QaicPlatform

pytestmark = pytest.mark.skipif(
    not QaicPlatform.is_aot,
    reason="SpD is AOT-only; not supported under QAIC eager mode",
)

_SPEC_CONFIG = {
    "method": "ngram",
    "num_speculative_tokens": 3,
    "prompt_lookup_max": 4,
    "prompt_lookup_min": 1,
}


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
def test_ngram_spd_output_non_empty(device_group, make_runner):
    """ngram SpD must produce non-empty output on real hardware."""
    from vllm import SamplingParams

    with make_runner(False, device_group, speculative_config=_SPEC_CONFIG) as model:
        out = model.generate(
            ["My name is John and I work at"],
            SamplingParams(temperature=0.0, max_tokens=20),
        )
    texts = [texts[0] for _, texts in out]
    assert len(texts[0]) > 0


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
)
class TestNgramSpdMatchesBaseline:
    """Isolated in its own class (own qaic_test_config marker) so
    ci_scripts/collect_jobs.py's job sizing - which resolves num_devices from
    a single representative item per _item_scope() (nodeid.rsplit("::", 1)[0])
    - sees num_device_groups=2 for this test independently of
    test_ngram_spd_output_non_empty above (which only needs 1 device group).
    Sharing a module-level scope with it would size this job for 1 device and
    raise DevicePoolError at runtime."""

    def test_ngram_spd_matches_baseline(self, device_groups, make_runner):
        """ngram SpD greedy output must match the non-SpD greedy baseline."""
        from vllm import SamplingParams

        prompt = ["My name is John and I work at"]
        greedy = SamplingParams(temperature=0.0, max_tokens=20)

        with make_runner(False, device_groups[0]) as base_model:
            out_base = base_model.generate(prompt, greedy)

        with make_runner(
            False, device_groups[1], speculative_config=_SPEC_CONFIG
        ) as spd_model:
            out_spd = spd_model.generate(prompt, greedy)

        assert out_base[0][0] == out_spd[0][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
