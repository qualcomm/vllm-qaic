# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""End-to-end disaggregated multimodal serving with the Encoder Cache (EC).

Runs Qwen3.6-35B-A3B in E-P-D (1E/1P/1D) once with the EC connector and once
with the legacy embedding transfer, checks both serve image requests, and checks
that both transports produce the same output for the same images.
Precompiled QPCs can be supplied through QAIC_EC_TEST_{ENCODE,PREFILL,DECODE}_QPC.
"""

import io
import os
from concurrent.futures import ThreadPoolExecutor

import pybase64 as base64
import pytest
import requests
from PIL import Image

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
CTX_LEN = 14336
HEIGHTS = [120, 180, 240, 256, 354, 360, 480, 512, 576, 720, 1080, 2048]
WIDTHS = [360, 320, 360, 454, 536, 640, 720, 910, 720, 1280, 1920, 2048]


def _with_qpc(config, env_var):
    qpc_path = os.environ.get(env_var)
    return config | {"qpc_path": qpc_path} if qpc_path else config


ENCODE_OVERRIDE = _with_qpc(
    {
        "height": HEIGHTS,
        "width": WIDTHS,
        "num_frames": [1] * len(HEIGHTS),
        "mxfp6_matmul": False,
        "mxint8_kv_cache": False,
        "aic_enable_depth_first": True,
        "split_model_io": True,
        "use_onnx_subfunctions": True,
        "pretrained_extra_args": {"dtype": "float16"},
    },
    "QAIC_EC_TEST_ENCODE_QPC",
)
PREFILL_OVERRIDE = _with_qpc(
    {
        "height": HEIGHTS,
        "width": WIDTHS,
        "num_frames": [1] * len(HEIGHTS),
        "batch_size": 1,
        "full_batch_size": 1,
        "vision_batch_size": 1,
        "user_tiled": True,
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "split_model_io": True,
        "use_onnx_subfunctions": True,
        "enable_chunking": True,
        "retain_full_kv": True,
        "kv_offload": True,
        "prefill_seq_len": 512,
        "qaic_config": {
            "blocking_mode": "prefill_online",
            "gdn_full_state_update": True,
            "num_kv_blocks": 16,
            "num_q_blocks": 4,
            "n_rep_chunk": 1,
            "skip_kv": True,
            "gdn_chunk_size": 512,
            "ctx_len": CTX_LEN,
        },
        "pretrained_extra_args": {"dtype": "float16"},
    },
    "QAIC_EC_TEST_PREFILL_QPC",
)
DECODE_OVERRIDE = _with_qpc(
    {
        "height": HEIGHTS,
        "width": WIDTHS,
        "num_frames": [1] * len(HEIGHTS),
        "mxfp6_matmul": True,
        "mxint8_kv_cache": True,
        "split_model_io": True,
        "use_onnx_subfunctions": True,
        "user_tiled": True,
        "enable_chunking": True,
        "retain_full_kv": True,
        "prefill_seq_len": 1,
        "kv_offload": True,
        "qaic_config": {
            "blocking_mode": "kv_batch_fold",
            "num_kv_blocks": 64,
            "gdn_num_head_blocks": 8,
            "skip_kv": True,
            "moe_config": {
                "flavour": "expert_parallel",
                "cores_per_expert": 1,
                "tree_reduce": False,
                "expert_parallel_chunk_size": 256,
            },
            "ctx_len": CTX_LEN,
        },
        "pretrained_extra_args": {"dtype": "float16"},
    },
    "QAIC_EC_TEST_DECODE_QPC",
)


def _image_url(width, height, color):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


IMAGES = [
    _image_url(910, 512, (220, 30, 30)),
    _image_url(536, 354, (30, 30, 220)),
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
    decode_bsz=512,
    num_encode_workers=1,
    encode_device_group_size=1,
    num_prefill_workers=1,
    prefill_device_group_size=1,
    num_decode_workers=1,
    decode_device_group_size=4,
    encode_override_qaic_config=ENCODE_OVERRIDE,
    prefill_override_qaic_config=PREFILL_OVERRIDE,
    decode_override_qaic_config=DECODE_OVERRIDE,
    extra_args=["--prefill-max-num-batched-tokens=512"],
)
class TestDisaggregatedServingEC:
    """E-P-D image serving, run once with EC and once with the legacy path."""

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
        if not _EC_OUTPUTS:
            pytest.skip("EC outputs not available; run the [ec] transport first")
        assert outputs == _EC_OUTPUTS

    def test_parallel_image_requests(self, ec_disagg_server):
        urls = [IMAGES[i % len(IMAGES)] for i in range(16)]
        with ThreadPoolExecutor(max_workers=8) as pool:
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
