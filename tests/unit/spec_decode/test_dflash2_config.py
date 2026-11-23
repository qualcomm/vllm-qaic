# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

from copy import deepcopy
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_qaic.model_loader.qaic import _derive_dflash_config, _get_qaic_compile_config


@pytest.fixture
def native_dflash2_registry():
    from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS

    if "DFlash2DraftModel" not in _SPECULATIVE_DECODING_MODELS:
        pytest.skip("Requires the native vLLM registry patch in tools/patches.")


def test_native_registry_patch(tmp_path):
    from vllm.model_executor.models import registry

    patch = shutil.which("patch")
    if patch is None:
        pytest.skip("The patch utility is required to validate the dependency patch.")
    destination = tmp_path / "model_executor/models/registry.py"
    destination.parent.mkdir(parents=True)
    source = Path(registry.__file__).read_text()
    source = (
        "\n".join(
            line for line in source.splitlines() if '"DFlash2DraftModel":' not in line
        )
        + "\n"
    )
    destination.write_text(source)
    patch_file = (
        Path(__file__).resolve().parents[3]
        / "tools/patches/vllm-dflash2-registry.patch"
    )
    result = subprocess.run(
        [patch, "--batch", "--forward", "-d", str(tmp_path), "-p2"],
        input=patch_file.read_text(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    module = ast.parse(destination.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "_SPECULATIVE_DECODING_MODELS"
            for t in node.targets
        )
    )
    entries = ast.literal_eval(assignment.value)
    assert entries["DFlash2DraftModel"] == entries["DFlashDraftModel"]


def test_native_registry_shares_dflash_capabilities(native_dflash2_registry):
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS

    expected = ("qwen3_dflash", "DFlashQwen3ForCausalLM")
    for architecture in ("DFlashDraftModel", "DFlash2DraftModel"):
        assert _SPECULATIVE_DECODING_MODELS[architecture] == expected
    assert (
        ModelRegistry.models["DFlashDraftModel"]
        == ModelRegistry.models["DFlash2DraftModel"]
    )


def test_native_registry_available_without_qaic_registration(native_dflash2_registry):
    # A fresh interpreter must see both entries without platform hooks.
    code = """
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.registry import _SPECULATIVE_DECODING_MODELS
for architecture in ("DFlashDraftModel", "DFlash2DraftModel"):
    assert architecture in ModelRegistry.get_supported_archs()
    assert _SPECULATIVE_DECODING_MODELS[architecture] == (
        "qwen3_dflash", "DFlashQwen3ForCausalLM"
    )
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "VLLM_PLUGINS": ""},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_platform_does_not_reregister_dflash(monkeypatch):
    from vllm.model_executor.models import ModelRegistry
    from vllm_qaic.platform import QaicPlatform

    register = Mock()
    monkeypatch.setattr(ModelRegistry, "register_model", register)
    QaicPlatform.pre_register_and_update()
    register.assert_not_called()


@pytest.mark.parametrize("dflash2", [False, True])
@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("capacity", [None, (1, 1), (1, 4), (4, 1)])
def test_dflash2_argmax_compile_policy(
    monkeypatch, dflash2, role, model_type, batch_size, capacity
):
    from vllm_qaic.model_loader import qaic

    clean_config = qaic._clean_config

    def clean_with_capacity(*args):
        result = clean_config(*args)
        if capacity is not None:
            result.update(batch_size=capacity[0], full_batch_size=capacity[1])
        return result

    monkeypatch.setattr(qaic, "_clean_config", clean_with_capacity)
    monkeypatch.setattr(
        "vllm_qaic.model_loader.qaic.QAIC_DEVICE_CONFIG", {"target": {}, "draft": {}}
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            quantization="mxfp6",
            max_model_len=4096,
            is_multimodal_model=role == "target",
            hf_config=SimpleNamespace(
                model_type=model_type, vision_config=SimpleNamespace(image_size=448)
            ),
            multimodal_config=SimpleNamespace(mm_processor_kwargs={}),
            runner_type="generate",
        ),
        cache_config=SimpleNamespace(
            cache_dtype="auto", num_cpu_blocks=1, enable_prefix_caching=False
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=batch_size),
        kv_transfer_config=None,
        speculative_config=SimpleNamespace(method="dflash", num_speculative_tokens=7),
        additional_config={
            "override_qaic_config": {
                "prefill_seq_len": 128,
                "num_cores": 8,
                "device_group": [0, 1],
                "pretrained_extra_args": {"continuous_batching": False},
            },
            "dflash_cfg": {
                "is_dflash2": dflash2,
                "block_size": 8,
                "target_layer_ids": [5, 19, 33, 47, 61],
                "dlm_repo": "draft",
                "tlm_repo": "target",
                "tlm_model_type": model_type,
            },
        },
    )
    original = deepcopy(config.additional_config)
    result = _get_qaic_compile_config(config, role)
    assert config.additional_config == original
    assert "continuous_batching" not in result.cfg
    assert "pretrained_extra_args" not in result.cfg
    assert result.cfg["prefill_only"] is True
    assert result.cfg["prefill_seq_len"] == (8 if role == "draft" else 128)
    assert not result.qaic_config.get("dflash_context_kv", False)
    if role == "target":
        assert result.qaic_config.get("dflash2_checkpoint") == (
            "draft" if dflash2 else None
        )
        assert result.num_logits_to_keep == 8
    elif dflash2:
        assert result.qaic_config["dflash_emit_selector_hidden"] is False
        assert result.cfg["node_precision_info"] is True
        assert result.cfg["use_onnx_subfunctions"] is False
    else:
        assert "dflash_emit_selector_hidden" not in result.qaic_config
        assert "node_precision_info" not in result.cfg


def test_dflash2_disaggregation_is_not_enabled():
    config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=SimpleNamespace(kv_role="kv_consumer"),
        speculative_config=SimpleNamespace(
            target_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3_5")
            ),
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["DFlash2DraftModel"])
            ),
        ),
    )
    with pytest.raises(ValueError, match="non-disaggregated"):
        _derive_dflash_config(config)
