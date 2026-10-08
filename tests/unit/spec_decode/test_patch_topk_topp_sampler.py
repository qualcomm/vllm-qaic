# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""CPU tests for the centralized AOT top-k/top-p patch."""

import subprocess
import sys

import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
from vllm_qaic.patch import patch_topk_topp_sampler
from vllm_qaic.platform import QaicPlatform
from vllm_qaic.v1.sample import topk_topp_sampler_shim


def test_import_time_patch_uses_upstream_pytorch_implementation(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(topk_topp_sampler, "apply_top_k_top_p", sentinel)
    monkeypatch.setattr(QaicPlatform, "is_aot", True)
    monkeypatch.setattr(patch_topk_topp_sampler, "_installed", False)

    patch_topk_topp_sampler.install()

    assert (
        topk_topp_sampler.apply_top_k_top_p
        is topk_topp_sampler.apply_top_k_top_p_pytorch
    )


def test_patch_is_aot_only(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(topk_topp_sampler, "apply_top_k_top_p", sentinel)
    monkeypatch.setattr(QaicPlatform, "is_aot", False)
    monkeypatch.setattr(patch_topk_topp_sampler, "_installed", False)

    patch_topk_topp_sampler.install()

    assert topk_topp_sampler.apply_top_k_top_p is sentinel


def test_legacy_shim_delegates_to_centralized_patch():
    assert topk_topp_sampler_shim.install is patch_topk_topp_sampler.install


def test_centralized_patch_matches_pytorch_reference(monkeypatch):
    monkeypatch.setattr(QaicPlatform, "is_aot", True)
    monkeypatch.setattr(patch_topk_topp_sampler, "_installed", False)
    patch_topk_topp_sampler.install()

    logits = torch.randn(16, 64, generator=torch.Generator().manual_seed(0))
    k = torch.full((16,), 5, dtype=torch.int32)
    p = torch.full((16,), 0.9)

    actual = topk_topp_sampler.apply_top_k_top_p(logits.clone(), k, p)
    expected = topk_topp_sampler.apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert torch.equal(actual, expected)


def test_import_time_patch_reaches_downstream_sampler_aliases():
    code = """
import vllm_qaic.patch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.rejection_sampler import apply_top_k_top_p as rejection
from vllm.v1.worker.gpu.sample.sampler import apply_top_k_top_p as sampler
from vllm.v1.worker.gpu.sample.states import apply_top_k_top_p as states

assert rejection is apply_top_k_top_p_pytorch
assert sampler is apply_top_k_top_p_pytorch
assert states is apply_top_k_top_p_pytorch
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
