# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
On-device unit tests for QAIC multimodal (vision-language) inference.

The e2e suite (multimodal/test_multimodal.py) exercises the "dual-QPC"
pipeline — a separate QEfficient-compiled vision encoder producing image
embeddings fed into a second generation LLM — across 6 VLM families
(Qwen2.5/3-VL, InternVL, Llava, Gemma3, Granite) plus Whisper audio. That
setup requires QEfficient model compilation outside of vLLM and is
appropriately e2e-only.

This file is the missing on-device-unit layer: confirms the simpler,
single-LLM multimodal path (image passed directly via
multi_modal_data["image"], no separate encoder QPC) produces well-formed
output for one lightweight VLM, using the same qaic_model fixture as every
other device-unit file in this directory. It intentionally does not
duplicate per-model-family coverage — that is what the e2e suite is for.

Coverage areas
--------------
1. A single-image request returns non-empty, well-formed output
2. A text-only request (no multi_modal_data) still works on a VLM
"""

import pytest
import requests
from PIL import Image
from vllm import SamplingParams

_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
)


@pytest.fixture(scope="class")
def rabbit_image():
    return Image.open(requests.get(_IMAGE_URL, stream=True).raw)


@pytest.mark.qaic_test_config(
    model_name="llava-hf/llava-interleave-qwen-0.5b-hf",
    ctx_len=4096,
    decode_bsz=1,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=2,
    device_group_size=1,
)
@pytest.mark.skip(
    reason="llava-hf/llava-interleave-qwen-0.5b-hf has "
    "vision_config.image_size=384; QEfficient's Llava dual-QPC path only "
    "supports img_size=336 (NotImplementedError otherwise) — known "
    "model/QEfficient-version incompatibility, also present in the e2e "
    "TestLlava suite. qaic_model is class-scoped, so LLM() construction (and "
    "the QEfficient compile crash) happens at fixture setup before either "
    "test body runs — a per-test pytest.skip() inside the test would never "
    "be reached, hence the class-level skip here."
)
class TestMultimodalOnDevice:
    def test_single_image_produces_output(self, qaic_model, rabbit_image):
        prompt = "USER: <image>\nWhat's in this image? ASSISTANT:"
        out = qaic_model.generate(
            [prompt],
            SamplingParams(temperature=0.0, max_tokens=16),
            images=[rabbit_image],
        )
        assert len(out[0][1][0]) > 0

    def test_text_only_request_produces_output(self, qaic_model):
        """A VLM must still serve a plain text prompt (no image)."""
        out = qaic_model.generate(
            ["My name is"],
            SamplingParams(temperature=0.0, max_tokens=16),
        )
        assert len(out[0][1][0]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
