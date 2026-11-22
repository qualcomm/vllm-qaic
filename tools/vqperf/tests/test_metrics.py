# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Metric extraction, aggregation and verdict tests."""

from __future__ import annotations

import json
from pathlib import Path

from vqperf import metrics

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_extract_known_scalars_and_percentiles() -> None:
    extracted = metrics.extract(load("result_base.json"))
    assert extracted["output_throughput"] == 512.0
    assert extracted["median_ttft_ms"] == 100.0
    assert extracted["p99_ttft_ms"] == 180.0
    # List-valued keys from upstream must not leak into the scalar table.
    assert "itls" not in extracted
    assert "ttfts" not in extracted


def test_extract_tolerates_missing_keys() -> None:
    extracted = metrics.extract({"output_throughput": 1.0})
    assert extracted == {"output_throughput": 1.0}


def test_extract_ignores_non_numeric_and_bools() -> None:
    extracted = metrics.extract(
        {"output_throughput": "fast", "completed": True, "duration": 3}
    )
    assert extracted == {"duration": 3.0}


def test_extract_picks_up_unusual_percentiles() -> None:
    extracted = metrics.extract({"p99.9_ttft_ms": 5.0})
    assert extracted["p99.9_ttft_ms"] == 5.0


def test_aggregate_uses_median_as_point_estimate() -> None:
    runs = [{"output_throughput": v} for v in (100.0, 102.0, 400.0)]
    aggregated = metrics.aggregate(runs)
    agg = aggregated["output_throughput"]
    assert agg.median == 102.0  # one slow/fast outlier must not move it
    assert agg.n == 3
    assert agg.minimum == 100.0 and agg.maximum == 400.0


def test_higher_better_regression() -> None:
    base = metrics.aggregate([{"output_throughput": 100.0}])
    new = metrics.aggregate([{"output_throughput": 80.0}])
    row = _only(metrics.compare(base, new))
    assert row.direction == "higher_better"
    assert row.delta == -20.0
    assert row.improvement_pct == -20.0
    assert row.verdict == "REGRESSION"


def test_lower_better_latency_improvement() -> None:
    base = metrics.aggregate([{"median_ttft_ms": 200.0}])
    new = metrics.aggregate([{"median_ttft_ms": 100.0}])
    row = _only(metrics.compare(base, new))
    assert row.direction == "lower_better"
    # Latency halved: delta is negative but the improvement is positive.
    assert row.delta == -100.0
    assert row.improvement_pct == 50.0
    assert row.verdict == "IMPROVEMENT"


def test_within_threshold_is_neutral() -> None:
    base = metrics.aggregate([{"output_throughput": 100.0}])
    new = metrics.aggregate([{"output_throughput": 102.0}])
    assert _only(metrics.compare(base, new)).verdict == "NEUTRAL"


def test_noise_band_demotes_to_noisy_with_enough_repeats() -> None:
    base = metrics.aggregate([{"output_throughput": v} for v in (100.0, 120.0, 80.0)])
    new = metrics.aggregate([{"output_throughput": v} for v in (90.0, 110.0, 70.0)])
    row = _only(metrics.compare(base, new))
    assert row.verdict == "NOISY"
    assert any("noise band" in w for w in row.warnings)


def test_single_run_cannot_be_demoted_but_is_flagged() -> None:
    base = metrics.aggregate([{"output_throughput": 100.0}])
    new = metrics.aggregate([{"output_throughput": 50.0}])
    row = _only(metrics.compare(base, new))
    assert row.verdict == "REGRESSION"
    assert any("no noise estimate" in w for w in row.warnings)


def test_two_repeats_is_not_a_noise_estimate() -> None:
    base = metrics.aggregate([{"output_throughput": v} for v in (100.0, 101.0)])
    new = metrics.aggregate([{"output_throughput": v} for v in (50.0, 51.0)])
    row = _only(metrics.compare(base, new))
    assert any("no noise estimate" in w for w in row.warnings)


def test_metric_missing_on_one_side_is_unknown() -> None:
    base = metrics.aggregate([{"output_throughput": 100.0}])
    new = metrics.aggregate([{"total_token_throughput": 100.0}])
    rows = {row.metric: row for row in metrics.compare(base, new)}
    assert rows["output_throughput"].verdict == "UNKNOWN"
    assert rows["total_token_throughput"].verdict == "UNKNOWN"


def test_zero_base_leaves_percentage_undefined() -> None:
    base = metrics.aggregate([{"output_throughput": 0.0}])
    new = metrics.aggregate([{"output_throughput": 10.0}])
    row = _only(metrics.compare(base, new))
    assert row.delta_pct is None
    assert row.verdict == "UNKNOWN"


def test_workload_metrics_are_informational_not_regressions() -> None:
    base = metrics.aggregate([{"completed": 64.0}])
    new = metrics.aggregate([{"completed": 32.0}])
    row = _only(metrics.compare(base, new))
    assert row.verdict == "INFO"
    assert any("workload differs" in w for w in row.warnings)


def test_custom_threshold_respected() -> None:
    base = metrics.aggregate([{"output_throughput": 100.0}])
    new = metrics.aggregate([{"output_throughput": 98.0}])
    row = _only(metrics.compare(base, new, threshold_pct=1.0))
    assert row.verdict == "REGRESSION"


def test_per_family_threshold_override() -> None:
    base = metrics.aggregate([{"median_ttft_ms": 100.0}])
    new = metrics.aggregate([{"median_ttft_ms": 103.0}])
    row = _only(metrics.compare(base, new, thresholds={"ttft": 2.0}))
    assert row.verdict == "REGRESSION"


def test_regressions_sort_first() -> None:
    base = metrics.aggregate(
        [{"output_throughput": 100.0, "total_token_throughput": 100.0}]
    )
    new = metrics.aggregate(
        [{"output_throughput": 50.0, "total_token_throughput": 100.0}]
    )
    rows = metrics.compare(base, new)
    assert rows[0].metric == "output_throughput"
    assert rows[0].verdict == "REGRESSION"


def test_worst_verdict_prefers_regression() -> None:
    base = metrics.aggregate([{"output_throughput": 100.0, "median_ttft_ms": 100.0}])
    new = metrics.aggregate([{"output_throughput": 50.0, "median_ttft_ms": 50.0}])
    assert metrics.worst_verdict(metrics.compare(base, new)) == "REGRESSION"


def test_end_to_end_fixture_comparison() -> None:
    base = metrics.aggregate([metrics.extract(load("result_base.json"))])
    new = metrics.aggregate([metrics.extract(load("result_new.json"))])
    rows = {row.metric: row for row in metrics.compare(base, new)}
    assert rows["output_throughput"].verdict == "REGRESSION"
    assert rows["median_ttft_ms"].verdict == "REGRESSION"


def _only(rows: list[metrics.Comparison]) -> metrics.Comparison:
    assert len(rows) == 1, [row.metric for row in rows]
    return rows[0]
