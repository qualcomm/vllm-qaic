# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Report generation: comparison CSVs, summary.md and the run manifest."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path

from . import metrics, profiler
from .config import BenchConfig, describe
from .metrics import Aggregate, Comparison
from .runner import RunState, aggregate_state

logger = logging.getLogger("vqperf.report")

SINGLE_RUN_BANNER = (
    "Single run per commit: no noise estimate is possible, so verdicts rest on "
    "the threshold alone. Use --repeats 3 or more before trusting a verdict."
)


def write_all(state: RunState, *, threshold_pct: float, top_ops: int) -> list[Path]:
    out = state.out_dir
    out.mkdir(parents=True, exist_ok=True)
    configs = {config.config_id: config for config in state.configs}
    aggregates = aggregate_state(state)

    comparisons: dict[str, list[Comparison]] = {}
    for config_id, sides in aggregates.items():
        comparisons[config_id] = metrics.compare(
            sides.get("base", {}),
            sides.get("new", {}),
            threshold_pct=threshold_pct,
        )

    written = [
        _write_per_run(out, state),
        _write_comparison(out, state, configs, comparisons),
        _write_comparison_wide(out, configs, comparisons),
        _write_manifest(out, state),
        _write_summary(out, state, configs, comparisons, threshold_pct=threshold_pct),
    ]

    if state.profiles:
        written.append(_write_profile_index(out, state))
        if state.op_tables:
            written.extend(profiler.write_op_csvs(out, state.op_tables, top=top_ops))
    return written


# ---------------------------------------------------------------------------
# per-run
# ---------------------------------------------------------------------------


def _write_per_run(out: Path, state: RunState) -> Path:
    path = out / "per_run.csv"
    metric_names = sorted({name for cell in state.cells for name in cell.metrics})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "side",
                "sha",
                "config_id",
                "repeat",
                "status",
                "reason",
                "server_ready_s",
                "bench_duration_s",
                "started_at",
                *metric_names,
            ]
        )
        for cell in state.cells:
            writer.writerow(
                [
                    cell.side,
                    cell.sha,
                    cell.config_id,
                    cell.repeat,
                    cell.status,
                    cell.reason,
                    _fmt(cell.server_ready_s),
                    _fmt(cell.bench_duration_s),
                    cell.started_at,
                    *[_fmt(cell.metrics.get(name)) for name in metric_names],
                ]
            )
    return path


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

_CONTEXT_COLUMNS = (
    "config_id",
    "model",
    "max_model_len",
    "random_input_len",
    "random_output_len",
    "num_prompts",
    "request_rate",
    "max_concurrency",
)


def _write_comparison(
    out: Path,
    state: RunState,
    configs: dict[str, BenchConfig],
    comparisons: dict[str, list[Comparison]],
) -> Path:
    path = out / "comparison.csv"
    label_columns = sorted({key for config in state.configs for key in config.labels})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                *_CONTEXT_COLUMNS,
                "metric",
                "unit",
                "direction",
                "base_median",
                "base_mean",
                "base_stdev",
                "base_min",
                "base_max",
                "base_n",
                "new_median",
                "new_mean",
                "new_stdev",
                "new_min",
                "new_max",
                "new_n",
                "delta",
                "delta_pct",
                "improvement_pct",
                "verdict",
                "warnings",
                *label_columns,
            ]
        )
        for config_id in sorted(comparisons):
            config = configs[config_id]
            described = describe(config)
            context = [described.get(column, "") for column in _CONTEXT_COLUMNS]
            for row in comparisons[config_id]:
                writer.writerow(
                    [
                        *context,
                        row.metric,
                        row.unit,
                        row.direction,
                        *_agg_cells(row.base),
                        *_agg_cells(row.new),
                        _fmt(row.delta),
                        _fmt(row.delta_pct),
                        _fmt(row.improvement_pct),
                        row.verdict,
                        "; ".join(row.warnings + config.warnings),
                        *[config.labels.get(key, "") for key in label_columns],
                    ]
                )
    return path


def _agg_cells(agg: Aggregate | None) -> list[str]:
    if agg is None:
        return ["", "", "", "", "", "0"]
    return [
        _fmt(agg.median),
        _fmt(agg.mean),
        _fmt(agg.stdev),
        _fmt(agg.minimum),
        _fmt(agg.maximum),
        str(agg.n),
    ]


def _write_comparison_wide(
    out: Path,
    configs: dict[str, BenchConfig],
    comparisons: dict[str, list[Comparison]],
) -> Path:
    path = out / "comparison_wide.csv"
    headline = metrics.HEADLINE_METRICS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["config_id", "model", "verdict"]
        for name in headline:
            header += [f"{name}_base", f"{name}_new", f"{name}_improvement_pct"]
        writer.writerow(header)
        for config_id in sorted(comparisons):
            rows = {row.metric: row for row in comparisons[config_id]}
            line = [
                config_id,
                configs[config_id].model,
                metrics.worst_verdict(comparisons[config_id]),
            ]
            for name in headline:
                row = rows.get(name)
                if row is None:
                    line += ["", "", ""]
                else:
                    line += [
                        _fmt(row.base.median if row.base else None),
                        _fmt(row.new.median if row.new else None),
                        _fmt(row.improvement_pct),
                    ]
            writer.writerow(line)
    return path


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------


def _write_profile_index(out: Path, state: RunState) -> Path:
    path = out / "profile_index.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# {profiler.HOST_ONLY_CAVEAT}"])
        writer.writerow(
            [
                "config_id",
                "side",
                "sha",
                "profile_mode",
                "status",
                "reason",
                "kind",
                "path",
                "size_bytes",
                "note",
            ]
        )
        for result in state.profiles:
            if not result.artifacts:
                writer.writerow(
                    [
                        result.config_id,
                        result.side,
                        result.sha,
                        result.mode,
                        result.status,
                        result.reason,
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue
            for artifact in result.artifacts:
                writer.writerow(
                    [
                        result.config_id,
                        result.side,
                        result.sha,
                        result.mode,
                        result.status,
                        result.reason,
                        artifact["kind"],
                        artifact["path"],
                        artifact["size_bytes"],
                        artifact["note"],
                    ]
                )
    return path


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------


def _write_manifest(out: Path, state: RunState) -> Path:
    path = out / "manifest.json"
    envs = {}
    for side, env in state.envs.items():
        envs[side] = {
            "sha": env.sha,
            "short_sha": env.checkout.short,
            "subject": env.checkout.subject,
            "committed_at": env.checkout.committed_at,
            "remote": env.checkout.remote_url,
            "mode": env.mode,
            "python_version": env.python_version,
            "env_prefix": str(env.prefix),
            "cache_key": env.cache_key,
            "versions": env.versions,
            "bench_invocation_form": env.bench.form if env.bench else "",
            "bench_argv": env.bench.argv if env.bench else [],
            "profiler_style": env.profiler_style,
            "pip_freeze": str(
                out / "provision" / env.checkout.short / "pip-freeze.txt"
            ),
        }

    payload = {
        "run_id": state.run_id,
        "started_at": state.started_at,
        "finished_at": state.finished_at,
        "args": state.args,
        "host": state.host,
        "environments": envs,
        "configurations": [
            {
                **describe(config),
                "profile": profiler.describe_spec(config.profile),
                "server_args": config.server_args(),
                "client_args": config.client_args(),
                "warnings": config.warnings,
            }
            for config in state.configs
        ],
        "cells": [asdict(cell) for cell in state.cells],
        "profiles": [asdict(result) for result in state.profiles],
        "warnings": state.warnings,
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def _write_summary(
    out: Path,
    state: RunState,
    configs: dict[str, BenchConfig],
    comparisons: dict[str, list[Comparison]],
    *,
    threshold_pct: float,
) -> Path:
    path = out / "summary.md"
    lines: list[str] = [f"# vqperf run `{state.run_id}`", ""]

    base = state.envs.get("base")
    new = state.envs.get("new")
    if base and new:
        lines += [
            "| | base | new |",
            "|---|---|---|",
            f"| sha | `{base.checkout.short}` | `{new.checkout.short}` |",
            f"| subject | {base.checkout.subject} | {new.checkout.subject} |",
            f"| committed | {base.checkout.committed_at} | "
            f"{new.checkout.committed_at} |",
            f"| vllm | {base.vllm_version} | {new.vllm_version} |",
            f"| vllm_qaic | {base.versions.get('vllm_qaic', '?')} | "
            f"{new.versions.get('vllm_qaic', '?')} |",
            f"| torch | {base.versions.get('torch', '?')} | "
            f"{new.versions.get('torch', '?')} |",
            f"| profiler | {base.profiler_style} | {new.profiler_style} |",
            "",
        ]

    args = state.args
    lines += [
        f"- mode: `{args.get('mode')}`, python `{args.get('python')}`, "
        f"devices `{args.get('devices')}`",
        f"- repeats: {args.get('repeats')}, order: `{args.get('order')}`, "
        f"warmup requests: {args.get('warmup_requests')}",
        f"- regression threshold: {threshold_pct:g}%",
        f"- host: {state.host.get('hostname')} "
        f"(kernel {state.host.get('kernel')}, "
        f"SDK {state.host.get('sdk_version') or '?'})",
        "",
    ]

    if int(args.get("repeats", 1)) < 3:
        lines += [f"> **{SINGLE_RUN_BANNER}**", ""]
    for warning in state.warnings:
        lines += [f"> WARNING: {warning}", ""]

    lines += _verdict_sections(configs, comparisons)
    lines += _profile_sections(state, comparisons)
    lines += _failure_section(state)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _verdict_sections(
    configs: dict[str, BenchConfig], comparisons: dict[str, list[Comparison]]
) -> list[str]:
    lines: list[str] = []
    sections = (("Regressions", "REGRESSION"), ("Improvements", "IMPROVEMENT"))
    for title, verdict in sections:
        rows = [
            (config_id, row)
            for config_id in sorted(comparisons)
            for row in comparisons[config_id]
            if row.verdict == verdict
        ]
        lines += [f"## {title}", ""]
        if not rows:
            lines += ["None at the configured threshold.", ""]
            continue
        lines += [
            "| config | metric | base | new | change |",
            "|---|---|---|---|---|",
        ]
        for config_id, row in rows:
            lines.append(
                f"| {config_id} | {row.metric} | "
                f"{_fmt(row.base.median if row.base else None)} | "
                f"{_fmt(row.new.median if row.new else None)} | "
                f"{_signed(row.improvement_pct)} |"
            )
        lines.append("")

    lines += ["## Full matrix", ""]
    for config_id in sorted(comparisons):
        config = configs[config_id]
        lines += [
            f"### `{config_id}` — {config.model}",
            "",
            f"verdict: **{metrics.worst_verdict(comparisons[config_id])}**",
            "",
            "| metric | unit | base | new | delta | improvement | verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in comparisons[config_id]:
            if row.metric not in metrics.HEADLINE_METRICS and row.verdict in (
                "NEUTRAL",
                "INFO",
            ):
                continue
            lines.append(
                f"| {row.metric} | {row.unit} | "
                f"{_fmt(row.base.median if row.base else None)} | "
                f"{_fmt(row.new.median if row.new else None)} | "
                f"{_fmt(row.delta)} | {_signed(row.improvement_pct)} | "
                f"{row.verdict} |"
            )
        lines.append("")
        for warning in config.warnings:
            lines.append(f"- note: {warning}")
        if config.warnings:
            lines.append("")
    return lines


def _profile_sections(
    state: RunState, comparisons: dict[str, list[Comparison]]
) -> list[str]:
    if not state.profiles:
        return []
    lines = ["## Where the time moved", "", f"_{profiler.HOST_ONLY_CAVEAT}_", ""]

    by_config: dict[str, list] = {}
    for result in state.profiles:
        by_config.setdefault(result.config_id, []).append(result)

    for config_id, results in sorted(by_config.items()):
        verdict = (
            metrics.worst_verdict(comparisons[config_id])
            if config_id in comparisons
            else "UNKNOWN"
        )
        lines += [f"### `{config_id}` (performance verdict: {verdict})", ""]

        unsupported = [r for r in results if r.status == "unsupported"]
        failed = [r for r in results if r.status == "failed"]
        if unsupported:
            lines += [
                f"- profiling unsupported: {unsupported[0].reason}. Absence of "
                f"attribution here is not absence of change.",
                "",
            ]
        for result in failed:
            lines += [f"- {result.side}: profiling failed — {result.reason}", ""]

        tables = state.op_tables.get(config_id, {})
        if "base" not in tables or "new" not in tables:
            lines += ["- no comparable op table for both sides.", ""]
        else:
            diffs = profiler.diff_op_tables(tables["base"], tables["new"], top=10)
            lines += [
                "| op | status | base self (ms) | new self (ms) | delta (ms) |",
                "|---|---|---|---|---|",
            ]
            for diff in diffs[:10]:
                lines.append(
                    f"| `{diff.name}` | {diff.status} | "
                    f"{_fmt(diff.base_self_ms)} | {_fmt(diff.new_self_ms)} | "
                    f"{_fmt(diff.delta_self_ms)} |"
                )
            lines.append("")

        for result in results:
            if result.lrt_trace_count:
                lines.append(
                    f"- {result.side}: {result.lrt_trace_count} LRT trace "
                    f"directories under `{result.lrt_dir}` "
                    f"(analyse with Cloud AI SDK tooling)"
                )
        lines.append("")
    return lines


def _failure_section(state: RunState) -> list[str]:
    failures = [cell for cell in state.cells if not cell.ok]
    profile_failures = [r for r in state.profiles if r.status == "failed"]
    if not failures and not profile_failures:
        return []
    lines = ["## Failures", ""]
    for cell in failures:
        lines += [
            f"- **{cell.side} {cell.config_id} r{cell.repeat}** "
            f"(`{cell.sha[:12]}`): {cell.reason}",
        ]
    for result in profile_failures:
        lines.append(f"- **profile {result.side} {result.config_id}**: {result.reason}")
    lines.append("")
    return lines


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _signed(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.2f}%"
