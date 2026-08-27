# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
Unit tests for pure-Python helper functions in test_prefill_streaming.py.

These helpers compute latency statistics and throughput from request outputs.
No hardware, model loading, or network access required.

Coverage areas
--------------
1. _compute_stats() — mean, median, p99 from a list of floats
2. _throughput() — tokens/second from e2e latencies and token counts
3. Edge cases: empty lists, single element, all-equal values
"""

import math

import pytest


# ---------------------------------------------------------------------------
# Re-implement the helpers under test (extracted from test_prefill_streaming.py,
# which lives at
# vllm-qaic/tests/device_unit_and_e2e/disaggregated_serving/test_prefill_streaming.py
# in the public vllm-qaic repo — reimplemented here rather than imported since
# that module requires a real disaggregated-serving deployment to import
# cleanly).
# ---------------------------------------------------------------------------

def _compute_stats(values: list[float]) -> dict:
    """Compute mean, median, p99 from a list of floats."""
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p99": float("nan")}
    import numpy as np
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p99": float(np.percentile(arr, 99)),
    }


def _throughput(e2e_list: list[float], toks_list: list[int]) -> float:
    """Compute throughput in tokens/second."""
    if not e2e_list:
        return float("nan")
    return sum(toks_list) / max(e2e_list)


# ===========================================================================
# 1. _compute_stats()
# ===========================================================================

class TestComputeStats:
    def test_empty_list_returns_nan(self):
        """Empty list must return NaN for all stats."""
        result = _compute_stats([])
        assert math.isnan(result["mean"])
        assert math.isnan(result["median"])
        assert math.isnan(result["p99"])

    def test_single_element(self):
        """Single element: mean == median == p99 == that element."""
        result = _compute_stats([42.0])
        assert result["mean"] == pytest.approx(42.0)
        assert result["median"] == pytest.approx(42.0)
        assert result["p99"] == pytest.approx(42.0)

    def test_all_equal_values(self):
        """All-equal list: mean == median == p99 == that value."""
        result = _compute_stats([5.0, 5.0, 5.0, 5.0])
        assert result["mean"] == pytest.approx(5.0)
        assert result["median"] == pytest.approx(5.0)
        assert result["p99"] == pytest.approx(5.0)

    def test_mean_correct(self):
        """Mean must be the arithmetic average."""
        result = _compute_stats([1.0, 2.0, 3.0, 4.0])
        assert result["mean"] == pytest.approx(2.5)

    def test_median_even_count(self):
        """Median of even-count list is average of two middle values."""
        result = _compute_stats([1.0, 2.0, 3.0, 4.0])
        assert result["median"] == pytest.approx(2.5)

    def test_median_odd_count(self):
        """Median of odd-count list is the middle value."""
        result = _compute_stats([1.0, 2.0, 3.0])
        assert result["median"] == pytest.approx(2.0)

    def test_p99_near_max_for_large_list(self):
        """p99 of a uniform list must be near the maximum."""
        values = list(range(1, 101))  # 1..100
        result = _compute_stats([float(v) for v in values])
        # p99 of 1..100 is ~99.01
        assert result["p99"] >= 99.0

    def test_p99_equals_max_for_two_elements(self):
        """p99 of [1, 100] must be close to 100."""
        result = _compute_stats([1.0, 100.0])
        assert result["p99"] >= 99.0

    def test_result_keys_present(self):
        """Result dict must have 'mean', 'median', 'p99' keys."""
        result = _compute_stats([1.0, 2.0])
        assert "mean" in result
        assert "median" in result
        assert "p99" in result

    def test_result_values_are_floats(self):
        """All result values must be Python floats."""
        result = _compute_stats([1.0, 2.0, 3.0])
        assert isinstance(result["mean"], float)
        assert isinstance(result["median"], float)
        assert isinstance(result["p99"], float)


# ===========================================================================
# 2. _throughput()
# ===========================================================================

class TestThroughput:
    def test_empty_e2e_list_returns_nan(self):
        """Empty e2e list must return NaN."""
        result = _throughput([], [])
        assert math.isnan(result)

    def test_basic_throughput(self):
        """throughput = sum(tokens) / max(e2e_latencies)."""
        # 100 tokens, max e2e = 2.0 seconds → 50 tok/s
        result = _throughput([1.0, 2.0], [50, 50])
        assert result == pytest.approx(50.0)

    def test_single_request(self):
        """Single request: throughput = tokens / e2e."""
        result = _throughput([4.0], [200])
        assert result == pytest.approx(50.0)

    def test_throughput_uses_max_e2e(self):
        """Throughput denominator must be max(e2e), not sum(e2e)."""
        # max e2e = 10.0, total tokens = 100 → 10 tok/s
        result = _throughput([5.0, 10.0, 3.0], [30, 40, 30])
        assert result == pytest.approx(10.0)

    def test_zero_tokens(self):
        """Zero total tokens → throughput = 0."""
        result = _throughput([1.0, 2.0], [0, 0])
        assert result == pytest.approx(0.0)

    def test_large_throughput(self):
        """Large token count with small latency → large throughput."""
        result = _throughput([0.001], [1000])
        assert result == pytest.approx(1_000_000.0)

    def test_equal_e2e_latencies(self):
        """When all e2e latencies are equal, max == any element."""
        result = _throughput([2.0, 2.0, 2.0], [10, 20, 30])
        assert result == pytest.approx(30.0)  # 60 / 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
