# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device tests for QAIC LoRA adapter inference.

Pure-Python coverage of search_adapters_in_cache() and
verify_adaptername_to_id_consistency() lives in unit/lora/test_lora_unit.py.
Heavier multi-adapter / max-adapter-count / server-lifecycle scenarios live
in the e2e suite (lora/test_qaic_lora.py, lora/test_qaic_generate_multiple_loras.py).
This file is the missing on-device-unit layer in between: confirms a single
loaded LoRA adapter is actually selected and applied per-request against a
real qaic_model runner, and that unadapted requests in the same batch fall
back to the base model.

Coverage areas
--------------
1. LLM loads successfully with a single LoRA adapter enabled
2. A request with lora_request set produces non-empty output
3. A request with no lora_request (base model) produces non-empty output
4. LoRA-adapted output differs from base-model output for the same prompt
"""

import pytest
from huggingface_hub import snapshot_download
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

BASE_MODEL_NAME = "PY007/TinyLlama-1.1B-Chat-v0.3"
ADAPTER_ID_0 = "jashing/tinyllama-colorist-lora"


@pytest.mark.qaic_test_config(
    model_name=BASE_MODEL_NAME,
    seq_len=64,
    ctx_len=128,
    decode_bsz=2,
    dtype="mxfp6",
    kv_dtype="mxint8",
    enable_lora=True,
    max_loras=1,
)
class TestLoraOnDevice:
    @pytest.fixture(scope="class")
    def lora_request(self):
        return LoRARequest(
            lora_name="colorist", lora_int_id=1, lora_path=snapshot_download(repo_id=ADAPTER_ID_0)
        )

    def test_llm_loads_with_lora_enabled(self, qaic_model):
        assert qaic_model is not None

    def test_lora_adapted_request_produces_output(self, qaic_model, lora_request):
        out = qaic_model.generate(
            ["Describe the color blue."],
            SamplingParams(temperature=0.0, max_tokens=16),
            lora_request=lora_request,
        )
        assert len(out[0][0][0]) > 0

    def test_base_model_request_produces_output(self, qaic_model):
        """A request without lora_request must still work (base model path)."""
        out = qaic_model.generate(
            ["Describe the color blue."],
            SamplingParams(temperature=0.0, max_tokens=16),
        )
        assert len(out[0][0][0]) > 0

    def test_lora_output_differs_from_base(self, qaic_model, lora_request):
        prompt = ["Describe the color blue."]
        params = SamplingParams(temperature=0.0, max_tokens=16)
        base_out = qaic_model.generate(prompt, params)
        lora_out = qaic_model.generate(prompt, params, lora_request=lora_request)
        assert base_out[0][1][0] != lora_out[0][1][0], (
            "LoRA-adapted output must differ from base-model output"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
