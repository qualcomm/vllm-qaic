# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Metric extraction, aggregation across repeats, and A/B verdicts.

Extraction is deliberately schema-tolerant: the two commits under comparison
may pin different vllm versions, so a key present on one side and absent on the
other must degrade to an empty cell, never a crash.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal, TypeGuard

Direction = Literal["higher_better", "lower_better"]

#: Scalar metrics with their unit and which way is an improvement.
SCALAR_METRICS: dict[str, tuple[str, Direction]] = {
    "duration": ("s", "lower_better"),
    "completed": ("requests", "higher_better"),
    "total_input_tokens": ("tokens", "higher_better"),
    "total_output_tokens": ("tokens", "higher_better"),
    "request_throughput": ("req/s", "higher_better"),
    "output_throughput": ("tok/s", "higher_better"),
    "total_token_throughput": ("tok/s", "higher_better"),
    "max_output_tokens_per_s": ("tok/s", "higher_better"),
    "max_concurrent_requests": ("requests", "higher_better"),
}

#: Latency families emitted per ``--percentile-metrics``; all lower-is-better.
LATENCY_FAMILIES = ("ttft", "tpot", "itl", "e2el")
LATENCY_STATS = ("mean", "median", "std", "p50", "p90", "p95", "p99")

#: Metrics promoted into ``comparison_wide.csv`` for at-a-glance review.
HEADLINE_METRICS = (
    "output_throughput",
    "total_token_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "median_itl_ms",
    "median_e2el_ms",
)

#: Metrics that describe the workload rather than performance. Compared, but
#: never used for a regression verdict -- a differing value means the two runs
#: did different amounts of work, which is a validity problem, not a result.
CONTEXT_METRICS = frozenset(
    {"completed", "total_input_tokens", "total_output_tokens", "duration"}
)

DEFAULT_THRESHOLD_PCT = 5.0

Verdict = Literal["REGRESSION", "IMPROVEMENT", "NEUTRAL", "NOISY", "UNKNOWN", "INFO"]


def metric_unit(name: str) -> str:
    if name in SCALAR_METRICS:
        return SCALAR_METRICS[name][0]
    if name.endswith("_ms"):
        return "ms"
    return ""


def metric_direction(name: str) -> Direction:
    if name in SCALAR_METRICS:
        return SCALAR_METRICS[name][1]
    return "lower_better"


def extract(result_json: dict[str, Any]) -> dict[str, float]:
    """Pull comparable scalars out of one upstream benchmark result JSON."""
    out: dict[str, float] = {}

    for name in SCALAR_METRICS:
        value = result_json.get(name)
        if _is_number(value):
            out[name] = float(value)

    for family in LATENCY_FAMILIES:
        for stat in LATENCY_STATS:
            key = f"{stat}_{family}_ms"
            value = result_json.get(key)
            if _is_number(value):
                out[key] = float(value)

    # Percentiles beyond the known set (e.g. --metric-percentiles 99.9).
    for key, value in result_json.items():
        if key in out or not key.endswith("_ms") or not _is_number(value):
            continue
        parts = key.split("_")
        if len(parts) == 3 and parts[1] in LATENCY_FAMILIES and parts[0][0] == "p":
            out[key] = float(value)

    return out


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class Aggregate:
    """Aggregation of one metric over N repeats of one (sha, config)."""

    metric: str
    values: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float | None:
        # Median is the point estimate: benchmark noise is right-skewed by
        # stray host activity, and one slow repeat must not move the verdict.
        return statistics.median(self.values) if self.values else None

    @property
    def mean(self) -> float | None:
        return statistics.fmean(self.values) if self.values else None

    @property
    def stdev(self) -> float | None:
        return statistics.stdev(self.values) if len(self.values) > 1 else None

    @property
    def minimum(self) -> float | None:
        return min(self.values) if self.values else None

    @property
    def maximum(self) -> float | None:
        return max(self.values) if self.values else None


def aggregate(runs: list[dict[str, float]]) -> dict[str, Aggregate]:
    """Aggregate per-metric values across the repeats of one cell."""
    out: dict[str, Aggregate] = {}
    for run in runs:
        for metric, value in run.items():
            out.setdefault(metric, Aggregate(metric)).values.append(value)
    return out


@dataclass
class Comparison:
    metric: str
    unit: str
    direction: Direction
    base: Aggregate | None
    new: Aggregate | None
    delta: float | None
    delta_pct: float | None
    improvement_pct: float | None
    verdict: Verdict
    warnings: list[str] = field(default_factory=list)


def compare(
    base: dict[str, Aggregate],
    new: dict[str, Aggregate],
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    thresholds: dict[str, float] | None = None,
) -> list[Comparison]:
    """Compare two aggregated metric sets, most-moved metric first."""
    rows: list[Comparison] = []
    for metric in sorted(set(base) | set(new)):
        rows.append(
            _compare_one(
                metric,
                base.get(metric),
                new.get(metric),
                threshold_pct=_threshold_for(metric, threshold_pct, thresholds),
            )
        )
    rows.sort(key=_severity_key)
    return rows


def _threshold_for(
    metric: str, default: float, thresholds: dict[str, float] | None
) -> float:
    if not thresholds:
        return default
    if metric in thresholds:
        return thresholds[metric]
    for family in LATENCY_FAMILIES:
        if metric.endswith(f"_{family}_ms") and family in thresholds:
            return thresholds[family]
    return thresholds.get("default", default)


def _compare_one(
    metric: str,
    base: Aggregate | None,
    new: Aggregate | None,
    *,
    threshold_pct: float,
) -> Comparison:
    unit = metric_unit(metric)
    direction = metric_direction(metric)
    warnings: list[str] = []

    base_value = base.median if base else None
    new_value = new.median if new else None

    if base_value is None or new_value is None:
        missing = "base" if base_value is None else "new"
        warnings.append(f"metric absent on {missing} side")
        return Comparison(
            metric, unit, direction, base, new, None, None, None, "UNKNOWN", warnings
        )

    delta = new_value - base_value
    delta_pct = (100.0 * delta / base_value) if base_value != 0 else None
    if delta_pct is None:
        warnings.append("base median is 0; percentage undefined")

    improvement_pct = None
    if delta_pct is not None:
        improvement_pct = delta_pct if direction == "higher_better" else -delta_pct

    if metric in CONTEXT_METRICS:
        # Workload-shape metrics: report, but never call them a regression.
        if delta_pct is not None and abs(delta_pct) > 1.0 and metric != "duration":
            warnings.append(
                "workload differs between commits; performance deltas for this "
                "config may not be comparable"
            )
        verdict: Verdict = "INFO"
        return Comparison(
            metric,
            unit,
            direction,
            base,
            new,
            delta,
            delta_pct,
            improvement_pct,
            verdict,
            warnings,
        )

    noise = _noise_band(base, new)
    if noise is None:
        warnings.append("single run: no noise estimate")

    if improvement_pct is None:
        verdict = "UNKNOWN"
    elif improvement_pct <= -threshold_pct:
        verdict = "REGRESSION"
    elif improvement_pct >= threshold_pct:
        verdict = "IMPROVEMENT"
    else:
        verdict = "NEUTRAL"

    if (
        verdict in ("REGRESSION", "IMPROVEMENT")
        and noise is not None
        and abs(delta) <= noise
    ):
        verdict = "NOISY"
        warnings.append(
            f"|delta|={abs(delta):.4g} is within the noise band {noise:.4g}"
        )

    return Comparison(
        metric,
        unit,
        direction,
        base,
        new,
        delta,
        delta_pct,
        improvement_pct,
        verdict,
        warnings,
    )


def _noise_band(base: Aggregate | None, new: Aggregate | None) -> float | None:
    """2-sigma band on the difference of two medians, or None if unknowable.

    Requires >= 3 repeats on both sides: a stdev from two samples is not a
    noise estimate, and treating it as one overstates confidence.
    """
    if base is None or new is None or base.n < 3 or new.n < 3:
        return None
    base_sd, new_sd = base.stdev, new.stdev
    if base_sd is None or new_sd is None:
        return None
    pooled = math.sqrt(base_sd**2 + new_sd**2)
    return 2.0 * pooled / math.sqrt(min(base.n, new.n))


_SEVERITY_ORDER = {
    "REGRESSION": 0,
    "IMPROVEMENT": 1,
    "NOISY": 2,
    "NEUTRAL": 3,
    "UNKNOWN": 4,
    "INFO": 5,
}


def _severity_key(row: Comparison) -> tuple[int, float, str]:
    magnitude = abs(row.improvement_pct) if row.improvement_pct is not None else 0.0
    headline = 0 if row.metric in HEADLINE_METRICS else 1
    return (_SEVERITY_ORDER[row.verdict], -magnitude, f"{headline}{row.metric}")


def worst_verdict(rows: list[Comparison]) -> Verdict:
    """The headline verdict for a configuration."""
    for verdict in ("REGRESSION", "UNKNOWN", "NOISY", "IMPROVEMENT", "NEUTRAL"):
        if any(row.verdict == verdict for row in rows):
            return verdict  # type: ignore[return-value]
    return "INFO"
