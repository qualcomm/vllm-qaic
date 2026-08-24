# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC runtime error handling.

Pure-Python config-validation errors (raised at check_and_update_config()
time, before any device is touched) are covered in
unit/generic/test_error_handling.py::TestConfigValidationErrors. These tests
cover errors/edge-cases that only surface once a real runner is generating
against real hardware.

Coverage areas
--------------
1. Prompt exceeding max_model_len raises rather than silently truncating
2. A failed/oversized request does not corrupt the engine for subsequent
   valid requests
"""

import pytest


@pytest.mark.qaic_test_config(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seq_len=64,
    ctx_len=128,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
)
class TestRuntimeErrorHandling:
    def test_prompt_exceeding_max_model_len_raises(self, qaic_model, ctx_len):
        """A prompt longer than max_model_len must raise rather than
        silently truncate or produce garbage output."""
        from vllm import SamplingParams

        oversized_prompt = "word " * (ctx_len * 2)
        with pytest.raises(Exception):
            qaic_model.generate(
                [oversized_prompt], SamplingParams(temperature=0.0, max_tokens=5)
            )

    def test_engine_recovers_after_failed_request(self, qaic_model, ctx_len):
        """After a request fails (e.g. prompt too long), the engine must
        still serve subsequent valid requests correctly."""
        from vllm import SamplingParams

        oversized_prompt = "word " * (ctx_len * 2)
        sp = SamplingParams(temperature=0.0, max_tokens=5)
        try:
            qaic_model.generate([oversized_prompt], sp)
        except Exception:
            pass

        out = qaic_model.generate(["Hello"], sp)
        assert len(out[0][0][0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
