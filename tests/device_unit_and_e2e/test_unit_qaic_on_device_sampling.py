# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device correctness tests for QAIC on-device sampling (ODS).

Pure-Python config parsing (aic_include_sampler bool coercion) and the
ODS + SpD mutual-exclusion assertion are covered in
unit/on_device_sampling/test_on_device_sampling.py. These tests exercise
ODS end-to-end against a real runner.

ODS is not yet supported in vllm-qaic — marked xfail(strict=False) so this
suite documents the intended behaviour without failing CI until support
lands (strict=False also tolerates an unexpected pass once it does).

Coverage areas
--------------
1. ODS-enabled runner produces non-empty output for presence/frequency/
   repetition penalty and top-p/top-k sampling
"""

import pytest

pytestmark = pytest.mark.xfail(
    reason="On-device sampling (ODS) is not yet supported in vllm-qaic",
    strict=False,
)


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    override_qaic_config={"aic_include_sampler": True},
)
class TestOnDeviceSamplingOnDevice:
    def test_presence_penalty_runs(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(
            ["My name is"],
            SamplingParams(temperature=0.0, max_tokens=20, presence_penalty=1.5),
        )
        assert len(out[0][0][0]) > 0

    def test_frequency_penalty_runs(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(
            ["My name is"],
            SamplingParams(temperature=0.0, max_tokens=20, frequency_penalty=1.0),
        )
        assert len(out[0][0][0]) > 0

    def test_top_p_sampling_runs(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(
            ["My name is"], SamplingParams(temperature=0.8, top_p=0.9, max_tokens=10)
        )
        assert len(out[0][0][0]) > 0

    def test_top_k_sampling_runs(self, qaic_model):
        from vllm import SamplingParams

        out = qaic_model.generate(
            ["My name is"], SamplingParams(temperature=0.8, top_k=50, max_tokens=10)
        )
        assert len(out[0][0][0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
