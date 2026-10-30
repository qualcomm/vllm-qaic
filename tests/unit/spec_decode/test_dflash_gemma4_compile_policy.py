# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from types import SimpleNamespace

import pytest

from vllm_qaic.model_loader.qaic import _get_qaic_compile_config


@pytest.mark.parametrize("role", [None, "kv_producer", "kv_consumer"])
@pytest.mark.parametrize("dflash", [False, True])
@pytest.mark.parametrize(
    "model_type,moe",
    [("gemma4", False), ("gemma4", True), ("qwen3_vl", False)],
)
def test_subfunction_policy_is_specific_to_dflash_gemma4(
    monkeypatch, role, dflash, model_type, moe
):
    monkeypatch.setattr(
        "vllm_qaic.model_loader.qaic.QAIC_DEVICE_CONFIG",
        {"default": {}, "target": {}, "draft": {}},
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization="mxfp6",
            max_model_len=4096,
            is_multimodal_model=True,
            hf_config=SimpleNamespace(
                model_type=model_type,
                text_config=SimpleNamespace(enable_moe_block=moe),
                vision_config=SimpleNamespace(image_size=448),
            ),
            multimodal_config=SimpleNamespace(mm_processor_kwargs={}),
            runner_type="generate",
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto", num_cpu_blocks=1, enable_prefix_caching=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        kv_transfer_config=SimpleNamespace(kv_role=role) if role else None,
        speculative_config=(
            SimpleNamespace(method="dflash", num_speculative_tokens=15)
            if dflash
            else None
        ),
        additional_config={
            "override_qaic_config": {
                "prefill_seq_len": 128,
                "num_cores": 8,
                "device_group": [0, 1, 2, 3],
            }
        },
    )
    if dflash:
        config.additional_config["dflash_cfg"] = {
            "block_size": 16,
            "target_layer_ids": [1, 3],
            "dlm_repo": "draft",
            "tlm_repo": "target",
            "tlm_model_type": model_type,
        }
    result = _get_qaic_compile_config(config, "target" if dflash else "default")
    assert bool(result.cfg.get("use_onnx_subfunctions")) == (
        dflash and model_type == "gemma4"
    )
    if dflash and model_type == "gemma4":
        assert result.cfg["node_precision_info"] is True
    if dflash:
        config.kv_transfer_config = None
        draft = _get_qaic_compile_config(config, "draft")
        assert not draft.cfg.get("use_onnx_subfunctions", False)
