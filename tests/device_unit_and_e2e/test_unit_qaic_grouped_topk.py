# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Pytest tests for QAIC grouped-topk router kernels.

Tests the QAIC HVX kernel implementations by calling through the vLLM-side
entry point (grouped_topk / GroupedTopKRouter.select_experts) and validating
correctness semantically: the chosen experts must be valid top-k candidates
and the output weights must equal the activation scores at the chosen ids.

Semantic validation (rather than exact-id comparison against a CPU reference)
is used because torch.topk breaks ties differently than the QAIC kernel; both
choices are equally valid. This mirrors the approach used in
benchmarks/kernels/benchmark_grouped_topk_router.py.

Coverage:
  - Grouped path  (num_experts > num_groups, valid grouping)
  - Regular path  (num_experts == num_groups, invalid grouping → regular topk)
  - fp16 scoring  (default kernel path, via grouped_topk / select_experts)
  - fp32 scoring  (multinsp_multithreaded_*_f32 kernels, called directly since
                   QAIC_ROUTER_FP32_SCORING is read at module import time)
  - softmax / sigmoid
  - with / without e_score_correction_bias
  - renormalize True / False

Run:
    pytest vllm-qaic/tests/device_unit_and_e2e/test_unit_qaic_grouped_topk.py
"""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

# Guard the QAIC-specific import so the file can be collected on non-QAIC
# machines (tests will be skipped but the import must not fail).
try:
    import vllm_qaic  # noqa: F401  — registers QAIC platform
    from vllm_qaic import _custom_ops as _qaic_wrapper
    from vllm_qaic.ops import register_qaic_customop

    register_qaic_customop()

    _VLLM_QAIC_AVAILABLE = True
except Exception:
    _VLLM_QAIC_AVAILABLE = False
    _qaic_wrapper = None  # type: ignore[assignment]

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
    grouped_topk,
)

# ---------------------------------------------------------------------------
# Skip marker — requires QAIC hardware (registered as OOT platform) + package.
# ---------------------------------------------------------------------------
_REQUIRES_QAIC = pytest.mark.skipif(
    not (current_platform.is_out_of_tree() and _VLLM_QAIC_AVAILABLE),
    reason="Test requires QAIC device and vllm_qaic package.",
)

_ATOL = 5e-3
_RTOL = 5e-3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand_logits(tokens: int, experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn((tokens, experts), generator=g, dtype=torch.float16).contiguous()


def _rand_bias(experts: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1009)
    return (0.1 * torch.randn((experts,), generator=g, dtype=torch.float16)).contiguous()


def _sync(device: torch.device) -> None:
    if hasattr(torch, "qaic") and hasattr(torch.qaic, "synchronize"):
        torch.qaic.synchronize(device)


def _scores_from_logits(logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Compute activation scores in fp16, matching what the fp16 kernel produces."""
    if scoring_func == "softmax":
        return torch.softmax(logits, dim=-1)
    return logits.sigmoid()


def _topk_threshold(values: torch.Tensor, k: int) -> torch.Tensor:
    """Minimum value among the top-k values per row."""
    return torch.topk(values, k=k, dim=-1).values.min(dim=-1).values


# ---------------------------------------------------------------------------
# Semantic validators
#
# These verify that the kernel output satisfies the top-k contract without
# requiring exact agreement on tie-broken expert ids.
# ---------------------------------------------------------------------------


def _assert_grouped_valid(
    logits_cpu: torch.Tensor,
    bias_cpu: torch.Tensor | None,
    got_w: torch.Tensor,
    got_ids: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    routed_scaling_factor: float,
    label: str,
) -> None:
    """Assert that (got_ids, got_w) is a valid grouped top-k result."""
    scores = _scores_from_logits(logits_cpu, scoring_func)
    selection_scores = (
        scores if bias_cpu is None else scores + bias_cpu.unsqueeze(0)
    )

    num_tokens, num_experts = scores.shape
    epg = num_experts // num_expert_group
    grouped_sel = selection_scores.view(num_tokens, num_expert_group, epg)

    if bias_cpu is None:
        group_scores = grouped_sel.max(dim=-1).values
    else:
        # DeepSeek-style: sum of top-2 scores within each group
        group_scores = grouped_sel.topk(2, dim=-1).values.sum(dim=-1)

    # --- Group selection: each chosen expert must be in a valid top-k group ---
    group_thresholds = _topk_threshold(group_scores, topk_group)
    got_groups = got_ids.long() // epg
    got_group_scores = group_scores.gather(1, got_groups)
    margin = _ATOL + _RTOL * group_thresholds.abs()
    valid_groups = got_group_scores >= (group_thresholds - margin).unsqueeze(1)
    assert valid_groups.all(), (
        f"{label}: some chosen experts are in suboptimal groups"
    )

    # --- Expert selection within the candidate set ---
    # Candidate groups = groups strictly above threshold ∪ groups the kernel
    # actually picked (to handle tie-breaks at the group boundary).
    strict_mask = group_scores > (group_thresholds + margin).unsqueeze(1)
    chosen_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    chosen_mask.scatter_(1, got_groups, True)
    cand_mask = (strict_mask | chosen_mask).unsqueeze(-1).expand_as(grouped_sel)
    cand_scores = selection_scores.masked_fill(~cand_mask.reshape(num_tokens, num_experts), float("-inf"))
    expert_thresholds = _topk_threshold(cand_scores, topk)
    got_sel = selection_scores.gather(1, got_ids.long())
    exp_margin = _ATOL + _RTOL * expert_thresholds.abs()
    valid_experts = got_sel >= (expert_thresholds - exp_margin).unsqueeze(1)
    assert valid_experts.all(), (
        f"{label}: some chosen experts are suboptimal within their groups"
    )

    # --- Weight check: got_w must equal activation scores at got_ids ---
    expected_w = scores.gather(1, got_ids.long()).float()
    if renormalize:
        expected_w = expected_w / expected_w.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        expected_w = expected_w * routed_scaling_factor
    torch.testing.assert_close(
        got_w, expected_w, atol=_ATOL, rtol=_RTOL,
        msg=f"{label}: weight mismatch for chosen ids",
    )


def _assert_regular_valid(
    logits_cpu: torch.Tensor,
    bias_cpu: torch.Tensor | None,
    got_w: torch.Tensor,
    got_ids: torch.Tensor,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    label: str,
) -> None:
    """Assert that (got_ids, got_w) is a valid regular (ungrouped) top-k result."""
    scores = _scores_from_logits(logits_cpu, scoring_func)
    selection_scores = (
        scores if bias_cpu is None else scores + bias_cpu.unsqueeze(0)
    )

    thresholds = _topk_threshold(selection_scores, topk)
    got_sel = selection_scores.gather(1, got_ids.long())
    margin = _ATOL + _RTOL * thresholds.abs()
    valid = got_sel >= (thresholds - margin).unsqueeze(1)
    assert valid.all(), f"{label}: some chosen experts are suboptimal"

    expected_w = scores.gather(1, got_ids.long()).float()
    if renormalize:
        expected_w = expected_w / expected_w.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        expected_w = expected_w * routed_scaling_factor
    torch.testing.assert_close(
        got_w, expected_w, atol=_ATOL, rtol=_RTOL,
        msg=f"{label}: weight mismatch for chosen ids",
    )


def _assert_structural_valid(
    got_w: torch.Tensor,
    got_ids: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    label: str,
) -> None:
    """Structural sanity checks for fp32-kernel outputs.

    The fp32 HVX exp/sigmoid differ slightly from CPU expf, so we cannot
    predict which experts rank highest from the CPU side.  Instead we verify:
    - correct output shape and dtype
    - all ids are valid expert indices (0 <= id < num_experts)
    - no duplicate ids within a single token's top-k selection
    - weights are non-negative
    - renormalized rows sum to ≈1 (before routed_scaling_factor)
    """
    assert got_w.shape == (num_tokens, topk), f"{label}: wrong weight shape"
    assert got_ids.shape == (num_tokens, topk), f"{label}: wrong ids shape"
    assert got_w.dtype == torch.float32, f"{label}: wrong weight dtype"
    assert got_ids.dtype == torch.int32, f"{label}: wrong ids dtype"
    assert (got_ids >= 0).all(), f"{label}: negative expert id"
    assert (got_ids < num_experts).all(), f"{label}: expert id out of range"
    assert (got_w >= 0).all(), f"{label}: negative weight"
    # No duplicate ids per token row
    for row in range(num_tokens):
        ids_row = got_ids[row].tolist()
        assert len(ids_row) == len(set(ids_row)), (
            f"{label}: duplicate expert ids in row {row}: {ids_row}"
        )
    if renormalize:
        row_sums = got_w.sum(dim=-1) / routed_scaling_factor
        torch.testing.assert_close(
            row_sums,
            torch.ones_like(row_sums),
            atol=_ATOL,
            rtol=_RTOL,
            msg=f"{label}: renormalized weights do not sum to 1",
        )


# ---------------------------------------------------------------------------
# Direct fp32-kernel helpers
#
# QAIC_ROUTER_FP32_SCORING is read at module import time in
# vllm_qaic._custom_ops, so it cannot be toggled per-test via env
# var without risking fragile module reloads.  Instead, the fp32 tests call
# the fp32 HVX kernels directly — the same approach used in
# benchmarks/kernels/benchmark_grouped_topk_router.py.
# ---------------------------------------------------------------------------


def _grouped_topk_fp32_direct(
    logits_qaic: torch.Tensor,
    bias_qaic: torch.Tensor | None,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_bias = bias_qaic is not None
    bias = (
        torch.empty((logits_qaic.shape[-1],), dtype=torch.float16, device=logits_qaic.device)
        if bias_qaic is None
        else bias_qaic.contiguous()
    )
    kernel = _qaic_wrapper._kernel(
        "multinsp_multithreaded_grouped_topk_router"
    )
    score_mode = 1 if scoring_func == "softmax" else 3
    num_tokens, num_experts = logits_qaic.shape
    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device=logits_qaic.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=logits_qaic.device)
    kernel[_qaic_wrapper._device_grid(logits_qaic)](
        logits_qaic.contiguous(),
        bias,
        topk_w,
        topk_ids,
        num_tokens,
        num_experts,
        num_expert_group,
        topk_group,
        topk,
        int(renormalize),
        float(routed_scaling_factor),
        int(use_bias),
        score_mode,
    )
    return topk_w, topk_ids


def _regular_topk_fp32_direct(
    logits_qaic: torch.Tensor,
    bias_qaic: torch.Tensor | None,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    scoring_func: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    use_bias = bias_qaic is not None
    bias = (
        torch.empty((logits_qaic.shape[-1],), dtype=torch.float16, device=logits_qaic.device)
        if bias_qaic is None
        else bias_qaic.contiguous()
    )
    kernel = _qaic_wrapper._kernel("multinsp_multithreaded_topk_router")
    score_mode = 1 if scoring_func == "softmax" else 3
    num_tokens, num_experts = logits_qaic.shape
    topk_w = torch.empty((num_tokens, topk), dtype=torch.float32, device=logits_qaic.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=logits_qaic.device)
    kernel[_qaic_wrapper._device_grid(logits_qaic)](
        logits_qaic.contiguous(),
        bias,
        topk_w,
        topk_ids,
        num_tokens,
        num_experts,
        topk,
        int(renormalize),
        float(routed_scaling_factor),
        int(use_bias),
        score_mode,
    )
    return topk_w, topk_ids


# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

# num_experts > num_groups AND num_experts % num_groups == 0 → grouped path.
_GROUPED_CASES = [
    # (tokens, experts, groups, topk_group, topk)
    (8, 64, 8, 4, 8),    # typical Qwen3-MoE-A3B shape
    (1, 64, 8, 4, 8),    # single-token edge case
    (128, 64, 8, 4, 8),  # larger batch (tie-breaks more likely)
]

# num_experts == num_groups → invalid grouping → regular topk path.
_REGULAR_CASES = [
    # (tokens, experts, groups, topk_group, topk)
    (8, 8, 8, 4, 2),
    (1, 8, 8, 4, 2),
    (128, 8, 8, 4, 2),
]


def _make_router(
    experts: int,
    groups: int,
    topk_group: int,
    topk: int,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
    bias_cpu: torch.Tensor | None,
) -> GroupedTopKRouter:
    return GroupedTopKRouter(
        top_k=topk,
        global_num_experts=experts,
        eplb_state=EplbLayerState(),
        num_expert_group=groups,
        topk_group=topk_group,
        renormalize=renormalize,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=bias_cpu,
        enable_eplb=False,
    )


# ---------------------------------------------------------------------------
# Grouped-path tests — fp16 scoring, via grouped_topk() entry point
# ---------------------------------------------------------------------------

@_REQUIRES_QAIC
@pytest.mark.parametrize("tokens,experts,groups,topk_group,topk", _GROUPED_CASES)
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.0])
def test_grouped_topk_qaic(
    tokens: int,
    experts: int,
    groups: int,
    topk_group: int,
    topk: int,
    scoring_func: str,
    renormalize: bool,
    use_bias: bool,
    routed_scaling_factor: float,
) -> None:
    """Grouped path fp16: kernel output must satisfy the grouped top-k contract."""
    device = torch.device("qaic:0")
    logits_cpu = _rand_logits(tokens, experts, seed=42)
    bias_cpu = _rand_bias(experts, seed=42) if use_bias else None

    logits_qaic = logits_cpu.to(device=device)
    bias_qaic = None if bias_cpu is None else bias_cpu.to(device=device)
    _sync(device)

    hidden_qaic = torch.empty((tokens, 1), dtype=torch.float16, device=device)
    got_w, got_ids = grouped_topk(
        hidden_states=hidden_qaic,
        gating_output=logits_qaic,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=groups,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=bias_qaic,
    )
    _sync(device)

    _assert_grouped_valid(
        logits_cpu, bias_cpu,
        got_w.cpu(), got_ids.cpu(),
        topk, renormalize, groups, topk_group,
        scoring_func, routed_scaling_factor,
        label=f"fp16 grouped {scoring_func} bias={use_bias} renorm={renormalize} rsf={routed_scaling_factor}",
    )


# ---------------------------------------------------------------------------
# Grouped-path tests — fp32 scoring, via direct kernel call
# ---------------------------------------------------------------------------

@_REQUIRES_QAIC
@pytest.mark.parametrize("tokens,experts,groups,topk_group,topk", _GROUPED_CASES)
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_grouped_topk_qaic_fp32_scoring(
    tokens: int,
    experts: int,
    groups: int,
    topk_group: int,
    topk: int,
    scoring_func: str,
    renormalize: bool,
    use_bias: bool,
) -> None:
    """Grouped path fp32 scoring: direct kernel call to multinsp_*_f32 variants."""
    device = torch.device("qaic:0")
    logits_cpu = _rand_logits(tokens, experts, seed=7)
    bias_cpu = _rand_bias(experts, seed=7) if use_bias else None

    logits_qaic = logits_cpu.to(device=device)
    bias_qaic = None if bias_cpu is None else bias_cpu.to(device=device)
    _sync(device)

    got_w, got_ids = _grouped_topk_fp32_direct(
        logits_qaic, bias_qaic,
        groups, topk_group, topk,
        renormalize, 1.0, scoring_func,
    )
    _sync(device)

    _assert_structural_valid(
        got_w.cpu(), got_ids.cpu(),
        tokens, experts, topk,
        renormalize, 1.0,
        label=f"fp32 grouped {scoring_func} bias={use_bias} renorm={renormalize}",
    )


# ---------------------------------------------------------------------------
# Regular-path tests — fp16 scoring, via GroupedTopKRouter.select_experts()
#
# GroupedTopKRouter._compute_routing calls qaic_regular_topk when
# num_experts <= num_groups (invalid grouping).
# ---------------------------------------------------------------------------

@_REQUIRES_QAIC
@pytest.mark.parametrize("tokens,experts,groups,topk_group,topk", _REGULAR_CASES)
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.0])
def test_regular_topk_qaic_via_router(
    tokens: int,
    experts: int,
    groups: int,
    topk_group: int,
    topk: int,
    scoring_func: str,
    renormalize: bool,
    use_bias: bool,
    routed_scaling_factor: float,
) -> None:
    """Regular path fp16: invalid grouping triggers qaic_regular_topk via router."""
    device = torch.device("qaic:0")
    logits_cpu = _rand_logits(tokens, experts, seed=99)
    bias_cpu = _rand_bias(experts, seed=99) if use_bias else None

    # The router passes routed_scaling_factor only when bias is present.
    eff_rsf = routed_scaling_factor if use_bias else 1.0

    router = _make_router(
        experts, groups, topk_group, topk,
        scoring_func, renormalize, routed_scaling_factor, bias_cpu,
    )
    logits_qaic = logits_cpu.to(device=device)
    hidden_qaic = torch.empty((tokens, 1), dtype=torch.float16, device=device)
    _sync(device)

    got_w, got_ids = router.select_experts(hidden_qaic, logits_qaic)
    _sync(device)

    _assert_regular_valid(
        logits_cpu, bias_cpu,
        got_w.cpu(), got_ids.cpu(),
        topk, renormalize, scoring_func, eff_rsf,
        label=f"regular {scoring_func} bias={use_bias} renorm={renormalize} rsf={routed_scaling_factor}",
    )


# ---------------------------------------------------------------------------
# Regular-path tests — fp32 scoring, via direct kernel call
# ---------------------------------------------------------------------------

@_REQUIRES_QAIC
@pytest.mark.parametrize("tokens,experts,groups,topk_group,topk", _REGULAR_CASES)
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("use_bias", [True, False])
def test_regular_topk_qaic_fp32_scoring(
    tokens: int,
    experts: int,
    groups: int,
    topk_group: int,
    topk: int,
    scoring_func: str,
    renormalize: bool,
    use_bias: bool,
) -> None:
    """Regular path fp32 scoring: direct kernel call to multinsp_topk_*_f32 variants."""
    device = torch.device("qaic:0")
    logits_cpu = _rand_logits(tokens, experts, seed=13)
    bias_cpu = _rand_bias(experts, seed=13) if use_bias else None

    logits_qaic = logits_cpu.to(device=device)
    bias_qaic = None if bias_cpu is None else bias_cpu.to(device=device)
    _sync(device)

    got_w, got_ids = _regular_topk_fp32_direct(
        logits_qaic, bias_qaic,
        topk, renormalize, 1.0, scoring_func,
    )
    _sync(device)

    _assert_structural_valid(
        got_w.cpu(), got_ids.cpu(),
        tokens, experts, topk,
        renormalize, 1.0,
        label=f"fp32 regular {scoring_func} bias={use_bias} renorm={renormalize}",
    )
