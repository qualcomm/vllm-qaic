# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Report artifacts: CSV shape, manifest contents and summary prose."""

from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import pytest
from vqperf import metrics, profiler, report
from vqperf.config import parse_configs
from vqperf.env import BenchInvocation, Environment
from vqperf.repo import Checkout
from vqperf.runner import CellResult, ProfileResult, RunState

FIXTURES = Path(__file__).parent / "fixtures"

CONFIG_CSV = """\
config_id,model,max_model_len,num_prompts,random_input_len,random_output_len,max_concurrency
cfg_a,meta-llama/Llama-3.1-8B-Instruct,2048,64,1024,128,8
"""


def env(side: str, sha_char: str) -> Environment:
    sha = sha_char * 40
    checkout = Checkout(
        sha=sha,
        path=Path(f"/tmp/vqperf/src/{sha_char}"),
        remote_url="https://github.com/qualcomm/vllm-qaic.git",
        subject=f"{side} commit subject",
        committed_at="2026-09-30T10:00:00+00:00",
    )
    return Environment(
        sha=sha,
        prefix=Path(f"/tmp/vqperf/envs/{sha_char}"),
        mode="pyt",
        python_version="3.12",
        checkout=checkout,
        cache_key=f"key-{sha_char}",
        versions={"vllm": "0.30.0", "vllm_qaic": "0.1.0", "torch": "2.13.0"},
        bench=BenchInvocation(["vllm", "bench", "serve"], "cli"),
        profiler_style="profiler_config",
    )


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def state(tmp_path: Path, *, repeats: int = 1, profiles: bool = False) -> RunState:
    path = tmp_path / "configs.csv"
    path.write_text(textwrap.dedent(CONFIG_CSV), encoding="utf-8")
    configs = parse_configs(path, mode="pyt")

    built = RunState(
        run_id="20260930-120000",
        out_dir=tmp_path / "out",
        args={
            "mode": "pyt",
            "python": "3.12",
            "devices": "0",
            "repeats": repeats,
            "order": "interleave",
            "warmup_requests": 8,
            "base": "a" * 40,
            "new": "b" * 40,
        },
        configs=configs,
        envs={"base": env("base", "a"), "new": env("new", "b")},
        host={"hostname": "host1", "kernel": "6.8.0", "sdk_version": "1.23"},
        started_at="2026-09-30T12:00:00+00:00",
        finished_at="2026-09-30T12:40:00+00:00",
    )

    for repeat in range(repeats):
        for side, payload in (
            ("base", "result_base.json"),
            ("new", "result_new.json"),
        ):
            built.cells.append(
                CellResult(
                    side=side,
                    sha=built.envs[side].sha,
                    config_id="cfg_a",
                    repeat=repeat,
                    status="ok",
                    metrics=metrics.extract(load(payload)),
                    bench_duration_s=20.0,
                    started_at="2026-09-30T12:05:00+00:00",
                )
            )

    if profiles:
        base_rows = profiler.parse_op_table(
            (FIXTURES / "profiler_out_0.base.txt").read_text(encoding="utf-8")
        )
        new_rows = profiler.parse_op_table(
            (FIXTURES / "profiler_out_0.new.txt").read_text(encoding="utf-8")
        )
        built.op_tables["cfg_a"] = {"base": base_rows, "new": new_rows}
        for side, rows in (("base", base_rows), ("new", new_rows)):
            built.profiles.append(
                ProfileResult(
                    side=side,
                    sha=built.envs[side].sha,
                    config_id="cfg_a",
                    mode="torch",
                    status="ok",
                    torch_dir=f"/tmp/vqperf/out/profile/{side}",
                    op_rows=len(rows),
                    artifacts=[
                        {
                            "path": (
                                f"/tmp/vqperf/out/profile/{side}/w.pt.trace.json.gz"
                            ),
                            "kind": "torch_trace",
                            "size_bytes": 1024,
                            "note": "",
                        }
                    ],
                )
            )
    return built


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        lines = [line for line in handle if not line.startswith("#")]
    return list(csv.DictReader(lines))


# -- file set --------------------------------------------------------------


def test_write_all_produces_the_documented_file_set(tmp_path: Path) -> None:
    written = report.write_all(state(tmp_path), threshold_pct=3.0, top_ops=10)
    assert {path.name for path in written} == {
        "per_run.csv",
        "comparison.csv",
        "comparison_wide.csv",
        "manifest.json",
        "summary.md",
    }


def test_profile_artifacts_only_written_when_profiling_ran(tmp_path: Path) -> None:
    written = report.write_all(
        state(tmp_path, profiles=True), threshold_pct=3.0, top_ops=10
    )
    names = {path.name for path in written}
    assert "profile_index.csv" in names
    assert "profile_ops.csv" in names
    assert "profile_ops_diff.csv" in names


# -- per_run.csv -----------------------------------------------------------


def test_per_run_has_one_row_per_cell(tmp_path: Path) -> None:
    built = state(tmp_path, repeats=3)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    data = rows(built.out_dir / "per_run.csv")
    assert len(data) == 6  # 2 sides x 1 config x 3 repeats
    assert {row["side"] for row in data} == {"base", "new"}
    assert float(data[0]["output_throughput"]) == pytest.approx(512.0)


# -- comparison.csv --------------------------------------------------------


def test_comparison_is_long_format_with_verdicts(tmp_path: Path) -> None:
    built = state(tmp_path)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    data = {row["metric"]: row for row in rows(built.out_dir / "comparison.csv")}
    throughput = data["output_throughput"]
    assert throughput["config_id"] == "cfg_a"
    assert throughput["verdict"] == "REGRESSION"
    assert float(throughput["base_median"]) == pytest.approx(512.0)
    assert float(throughput["new_median"]) == pytest.approx(400.0)
    assert float(throughput["improvement_pct"]) == pytest.approx(-21.875)


def test_latency_regression_is_direction_normalized(tmp_path: Path) -> None:
    built = state(tmp_path)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    data = {row["metric"]: row for row in rows(built.out_dir / "comparison.csv")}
    ttft = data["median_ttft_ms"]
    # TTFT rose 100 -> 150ms: delta positive, improvement negative, REGRESSION.
    assert float(ttft["delta"]) == pytest.approx(50.0)
    assert float(ttft["improvement_pct"]) == pytest.approx(-50.0)
    assert ttft["verdict"] == "REGRESSION"


def test_context_metrics_are_info_not_regressions(tmp_path: Path) -> None:
    built = state(tmp_path)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    data = {row["metric"]: row for row in rows(built.out_dir / "comparison.csv")}
    # duration differs (16s vs 20.5s) but it describes the workload, not perf.
    assert data["duration"]["verdict"] == "INFO"


def test_comparison_wide_has_one_row_per_config(tmp_path: Path) -> None:
    built = state(tmp_path)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    data = rows(built.out_dir / "comparison_wide.csv")
    assert len(data) == 1
    assert data[0]["config_id"] == "cfg_a"
    assert data[0]["verdict"] == "REGRESSION"


# -- manifest.json ---------------------------------------------------------


def test_manifest_records_both_shas_and_argv(tmp_path: Path) -> None:
    built = state(tmp_path)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    manifest = json.loads((built.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["environments"]["base"]["sha"] == "a" * 40
    assert manifest["environments"]["new"]["sha"] == "b" * 40
    assert manifest["environments"]["base"]["bench_argv"] == [
        "vllm",
        "bench",
        "serve",
    ]
    config = manifest["configurations"][0]
    # The exact argv is the only durable record of what was measured.
    assert "--max-model-len" in config["server_args"]
    assert "--num-prompts" in config["client_args"]


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    built = state(tmp_path, profiles=True)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    text = (built.out_dir / "manifest.json").read_text(encoding="utf-8")
    # No Path or dataclass leaks: a second parse must succeed unchanged.
    assert json.loads(text) == json.loads(text)
    assert "/tmp/vqperf/envs/a" in text


# -- summary.md ------------------------------------------------------------


def test_summary_flags_single_run_statistics(tmp_path: Path) -> None:
    built = state(tmp_path, repeats=1)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    text = (built.out_dir / "summary.md").read_text(encoding="utf-8")
    assert report.SINGLE_RUN_BANNER in text


def test_summary_drops_banner_once_repeats_support_a_noise_estimate(
    tmp_path: Path,
) -> None:
    built = state(tmp_path, repeats=3)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    text = (built.out_dir / "summary.md").read_text(encoding="utf-8")
    assert report.SINGLE_RUN_BANNER not in text


def test_summary_lists_regressions_and_shas(tmp_path: Path) -> None:
    built = state(tmp_path)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    text = (built.out_dir / "summary.md").read_text(encoding="utf-8")
    assert "Regressions" in text
    assert "output_throughput" in text
    assert "aaaaaaaaaaaa" in text and "bbbbbbbbbbbb" in text
    assert "base commit subject" in text


def test_summary_profile_section_carries_the_host_only_caveat(tmp_path: Path) -> None:
    built = state(tmp_path, profiles=True)
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    text = (built.out_dir / "summary.md").read_text(encoding="utf-8")
    assert "host-side" in text
    assert "aten::copy_" in text


def test_summary_reports_failed_cells(tmp_path: Path) -> None:
    built = state(tmp_path)
    built.cells.append(
        CellResult(
            side="new",
            sha="b" * 40,
            config_id="cfg_a",
            repeat=1,
            status="failed",
            reason="bench_timeout after 3600s",
        )
    )
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    text = (built.out_dir / "summary.md").read_text(encoding="utf-8")
    assert "bench_timeout" in text


def test_failed_cells_are_excluded_from_aggregation(tmp_path: Path) -> None:
    built = state(tmp_path)
    built.cells.append(
        CellResult(
            side="new",
            sha="b" * 40,
            config_id="cfg_a",
            repeat=1,
            status="failed",
            reason="bench_error rc=1",
            metrics={"output_throughput": 1.0},
        )
    )
    report.write_all(built, threshold_pct=3.0, top_ops=10)
    data = {row["metric"]: row for row in rows(built.out_dir / "comparison.csv")}
    # The bogus 1.0 from the failed cell must not drag the median down.
    assert float(data["output_throughput"]["new_median"]) == pytest.approx(400.0)
