# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""End-to-end disaggregated multimodal serving with the Encoder Cache (EC).

Runs Qwen3-VL-2B-Instruct in E-PD (1E/1PD) once with the EC connector and once
with the legacy embedding transfer, checks both serve image requests, and checks
that both transports produce the same output for the same images.
"""

import io
import json
from concurrent.futures import ThreadPoolExecutor

import pybase64 as base64
import pytest
import requests
from PIL import Image

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
CTX_LEN = 4096
SEQ_LEN = 128
HEIGHTS = [140, 280, 420, 354]
WIDTHS = [140, 280, 420, 536]

MM_PROCESSOR_KWARGS = {"min_pixels": 4 * 28 * 28, "max_pixels": 1280 * 28 * 28}

QAIC_OVERRIDE = {
    "prefill_seq_len": SEQ_LEN,
    "height": HEIGHTS,
    "width": WIDTHS,
    "split_model_io": True,
}


def _image_url(width, height, color):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


COLORS = [(220, 30, 30), (30, 30, 220), (30, 180, 30), (230, 200, 30)]
IMAGES = [
    _image_url(width, height, color)
    for width, height, color in zip(WIDTHS, HEIGHTS, COLORS, strict=True)
]

_EC_OUTPUTS: list[str] = []


def _chat(server, image_url, max_tokens=32):
    response = requests.post(
        f"http://localhost:{server['disaggregated_server_port']}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "What color is this image?"},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout=server["client_request_timeout"] * 5,
    )
    assert response.status_code == 200, response.text
    return response.json()["choices"][0]["message"]["content"]


@pytest.mark.qaic_disagg_installed
@pytest.mark.qaic_aot_mode
@pytest.mark.qaic_test_config(
    model_name=MODEL_NAME,
    ctx_len=CTX_LEN,
    decode_bsz=4,
    num_encode_workers=1,
    encode_device_group_size=1,
    num_decode_workers=1,
    decode_device_group_size=1,
    encode_override_qaic_config=QAIC_OVERRIDE,
    decode_override_qaic_config=QAIC_OVERRIDE,
    extra_args=[
        f"--mm-processor-kwargs={json.dumps(MM_PROCESSOR_KWARGS)}",
        f"--max-num-batched-tokens={SEQ_LEN}",
    ],
)
class TestDisaggregatedServingEC:
    """E-PD image serving, run once with EC and then once with the legacy path.

    The two transports run serially in the same pytest process, EC first, so
    test_ec_matches_legacy can compare them.
    """

    @pytest.mark.parametrize("image_index", range(len(IMAGES)))
    def test_image_request(self, ec_disagg_server, image_index):
        assert _chat(ec_disagg_server, IMAGES[image_index]).strip()

    def test_repeated_image_request(self, ec_disagg_server):
        outputs = [_chat(ec_disagg_server, IMAGES[0]) for _ in range(3)]
        assert all(output.strip() for output in outputs)

    def test_ec_matches_legacy(self, ec_disagg_server):
        outputs = [_chat(ec_disagg_server, url) for url in IMAGES]
        if ec_disagg_server["use_ec"]:
            _EC_OUTPUTS[:] = outputs
            return
        assert _EC_OUTPUTS, "[ec] transport must run before [legacy]"
        assert outputs == _EC_OUTPUTS

    def test_parallel_image_requests(self, ec_disagg_server):
        urls = [IMAGES[i % len(IMAGES)] for i in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda url: _chat(ec_disagg_server, url), urls))
        assert all(output.strip() for output in outputs)

    def test_text_only_request(self, ec_disagg_server):
        response = requests.post(
            f"http://localhost:{ec_disagg_server['disaggregated_server_port']}"
            "/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=ec_disagg_server["client_request_timeout"] * 5,
        )
        assert response.status_code == 200, response.text

    def test_ec_handoff_server_started(self, ec_disagg_server):
        log = ec_disagg_server["log_dir"] / "qaic_echandoff.log"
        if not ec_disagg_server["use_ec"]:
            assert not log.exists()
            return
        assert log.exists()
        assert "Server started." in log.read_text()
