# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""CPU tests for the centralized top-k/top-p selector patch."""

import subprocess
import sys

import pytest
import torch

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
from vllm_qaic.patch import patch_topk_topp_sampler


@pytest.mark.parametrize("batch", [1, 16])
def test_flag_off_routes_to_pytorch_at_any_batch(monkeypatch, batch):
    # Upstream picks Triton for batch >= 8 when HAS_TRITON; the QAIC selector
    # must not, unless VLLM_QAIC_TRITON_TOPK_TOPP=1.
    monkeypatch.setattr(patch_topk_topp_sampler, "triton_en", False)
    calls = []

    def _pytorch(logits, k, p):
        calls.append(logits.shape[0])
        return logits

    monkeypatch.setattr(patch_topk_topp_sampler, "apply_top_k_top_p_pytorch", _pytorch)
    logits = torch.zeros(batch, 8)
    patch_topk_topp_sampler._qaic_apply_top_k_top_p(
        logits, torch.full((batch,), 2), None
    )
    assert calls == [batch]


def test_centralized_patch_matches_pytorch_reference(monkeypatch):
    monkeypatch.setattr(patch_topk_topp_sampler, "triton_en", False)

    logits = torch.randn(16, 64, generator=torch.Generator().manual_seed(0))
    k = torch.full((16,), 5, dtype=torch.int32)
    p = torch.full((16,), 0.9)

    actual = patch_topk_topp_sampler._qaic_apply_top_k_top_p(logits.clone(), k, p)
    expected = topk_topp_sampler.apply_top_k_top_p_pytorch(logits.clone(), k, p)

    assert torch.equal(actual, expected)


def test_import_time_patch_reaches_downstream_sampler_aliases():
    code = """
import vllm_qaic.patch
from vllm_qaic.patch.patch_topk_topp_sampler import _qaic_apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p as definition
from vllm.v1.sample.rejection_sampler import apply_top_k_top_p as rejection
from vllm.v1.worker.gpu.sample.sampler import apply_top_k_top_p as sampler
from vllm.v1.worker.gpu.sample.states import apply_top_k_top_p as states

assert definition is _qaic_apply_top_k_top_p
assert rejection is _qaic_apply_top_k_top_p
assert sampler is _qaic_apply_top_k_top_p
assert states is _qaic_apply_top_k_top_p
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
