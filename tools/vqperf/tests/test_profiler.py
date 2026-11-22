# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""ProfilerConfig flag construction and op-table parsing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from vqperf import profiler
from vqperf.config import ProfileSpec

FIXTURES = Path(__file__).parent / "fixtures"


# -- flag construction -----------------------------------------------------


def flags(spec: ProfileSpec, path: str = "/tmp/vqperf/trace") -> dict[str, str]:
    out = {}
    for flag in profiler.torch_profiler_flags(spec, Path(path)):
        key, _, value = flag.partition("=")
        out[key.removeprefix("--profiler-config.")] = value
    return out


def test_profiler_and_dir_always_emitted_together() -> None:
    # torch_profiler_dir is only valid when profiler == "torch", and required
    # when it is; emitting one without the other raises in ProfilerConfig.
    emitted = flags(ProfileSpec(mode="torch"))
    assert emitted["profiler"] == "torch"
    assert emitted["torch_profiler_dir"] == "/tmp/vqperf/trace"


def test_no_flags_when_torch_not_requested() -> None:
    assert profiler.torch_profiler_flags(ProfileSpec(mode="lrt"), Path("/tmp/x")) == []
    assert profiler.torch_profiler_flags(ProfileSpec(mode="off"), Path("/tmp/x")) == []


def test_relative_trace_dir_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        profiler.torch_profiler_flags(ProfileSpec(mode="torch"), Path("rel/dir"))


def test_with_stack_defaults_off_unlike_upstream() -> None:
    # Upstream defaults torch_profiler_with_stack=True; stack tracing is the
    # largest source of profiling skew, so vqperf defaults it off.
    assert flags(ProfileSpec(mode="torch"))["torch_profiler_with_stack"] == "false"


def test_booleans_render_lowercase() -> None:
    emitted = flags(ProfileSpec(mode="torch", with_stack=True, record_shapes=True))
    assert emitted["torch_profiler_with_stack"] == "true"
    assert emitted["torch_profiler_record_shapes"] == "true"


def test_ignore_frontend_auto_enabled_with_delay() -> None:
    # Upstream warns that front-end profiling ignores iteration windows and
    # captures the entire range, swamping the requested window.
    assert flags(ProfileSpec(mode="torch", delay_iterations=5))["ignore_frontend"] == (
        "true"
    )
    assert flags(ProfileSpec(mode="torch", max_iterations=5))["ignore_frontend"] == (
        "true"
    )
    assert flags(ProfileSpec(mode="torch"))["ignore_frontend"] == "false"


def test_explicit_ignore_frontend_wins() -> None:
    spec = ProfileSpec(mode="torch", delay_iterations=5, ignore_frontend=False)
    assert flags(spec)["ignore_frontend"] == "false"


def test_schedule_knobs_passed_through() -> None:
    spec = ProfileSpec(
        mode="torch", wait_iterations=1, warmup_iterations=2, active_iterations=7
    )
    emitted = flags(spec)
    assert emitted["wait_iterations"] == "1"
    assert emitted["warmup_iterations"] == "2"
    assert emitted["active_iterations"] == "7"


def test_lrt_env_uses_the_qaic_specific_variable() -> None:
    # vllm_qaic/envs.py defines VLLM_TORCH_QAIC_PROFILER_DIR; the upstream
    # VLLM_TORCH_PROFILER_DIR is gone and must never be set.
    env = profiler.lrt_env(ProfileSpec(mode="lrt"), Path("/tmp/lrt"))
    assert env == {"VLLM_TORCH_QAIC_PROFILER_DIR": "/tmp/lrt"}
    assert profiler.lrt_env(ProfileSpec(mode="torch"), Path("/tmp/lrt")) == {}


def test_no_legacy_torch_env_var() -> None:
    # Host profiling goes through --profiler-config only.
    assert not hasattr(profiler, "legacy_torch_env")
    assert not hasattr(profiler, "LEGACY_TORCH_ENV")
    flags = profiler.torch_profiler_flags(ProfileSpec(mode="torch"), Path("/tmp/t"))
    assert "--profiler-config.profiler=torch" in flags
    assert "--profiler-config.torch_profiler_dir=/tmp/t" in flags


def test_describe_spec_records_host_only_activities() -> None:
    described = profiler.describe_spec(ProfileSpec(mode="torch"))
    assert "CPU" in described["activities"]
    assert profiler.describe_spec(ProfileSpec(mode="off")) == {"profile_mode": "off"}


# -- durations -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.000ms", 1.0),
        ("500.000us", 0.5),
        ("2.500s", 2500.0),
        ("750.000ns", 0.00075),
        ("  1.5ms ", 1.5),
    ],
)
def test_parse_duration_ms(raw: str, expected: float) -> None:
    assert profiler.parse_duration_ms(raw) == pytest.approx(expected)


def test_parse_duration_rejects_junk() -> None:
    assert profiler.parse_duration_ms("--") is None
    assert profiler.parse_duration_ms("") is None


# -- op table --------------------------------------------------------------


def base_rows() -> list[profiler.OpRow]:
    return profiler.parse_op_table(
        (FIXTURES / "profiler_out_0.base.txt").read_text(encoding="utf-8")
    )


def new_rows() -> list[profiler.OpRow]:
    return profiler.parse_op_table(
        (FIXTURES / "profiler_out_0.new.txt").read_text(encoding="utf-8")
    )


def test_parse_op_table_reads_every_row() -> None:
    rows = base_rows()
    assert len(rows) == 7
    top = rows[0]
    assert top.name == "aten::mm"
    assert top.self_cpu_ms == pytest.approx(1.76)
    assert top.cpu_total_ms == pytest.approx(1.805)
    assert top.calls == 20
    assert top.self_cpu_pct == pytest.approx(35.2)


def test_parse_op_table_keeps_names_with_spaces_and_parens() -> None:
    names = {row.name for row in base_rows()}
    assert "execute_context_2(2048)_generation_6(6)" in names


def test_parse_op_table_stops_before_total_line() -> None:
    assert all("Self CPU time total" not in row.name for row in base_rows())


def test_parse_op_table_sorted_by_self_time() -> None:
    times = [row.self_cpu_ms or 0.0 for row in base_rows()]
    assert times == sorted(times, reverse=True)


def test_parse_op_table_on_garbage_returns_empty() -> None:
    assert profiler.parse_op_table("not a profiler table at all") == []
    assert profiler.parse_op_table("") == []


def test_diff_marks_added_and_removed_ops() -> None:
    diffs = {d.name: d for d in profiler.diff_op_tables(base_rows(), new_rows())}
    assert diffs["qaic::fused_rmsnorm"].status == "added"
    assert diffs["aten::silu"].status == "removed"
    assert diffs["aten::mm"].status == "both"


def test_diff_computes_deltas_and_ranks() -> None:
    diffs = {d.name: d for d in profiler.diff_op_tables(base_rows(), new_rows())}
    copy_ = diffs["aten::copy_"]
    assert copy_.delta_self_ms == pytest.approx(0.485)
    assert copy_.delta_self_pct == pytest.approx(48.5)
    assert copy_.rank_base == 2
    assert copy_.rank_new == 2


def test_diff_sorted_by_absolute_movement() -> None:
    diffs = profiler.diff_op_tables(base_rows(), new_rows())
    # aten::copy_ moved +0.485ms, the largest absolute change.
    assert diffs[0].name == "aten::copy_"


def test_diff_keeps_single_side_ops_even_with_tiny_top_n() -> None:
    diffs = {d.name for d in profiler.diff_op_tables(base_rows(), new_rows(), top=1)}
    assert "qaic::fused_rmsnorm" in diffs
    assert "aten::silu" in diffs


def test_load_op_table_merges_ranks(tmp_path: Path) -> None:
    text = (FIXTURES / "profiler_out_0.base.txt").read_text(encoding="utf-8")
    (tmp_path / "profiler_out_0.txt").write_text(text, encoding="utf-8")
    (tmp_path / "profiler_out_1.txt").write_text(text, encoding="utf-8")
    rows = {row.name: row for row in profiler.load_op_table(tmp_path)}
    # Two ranks with identical tables: self time and call counts add up.
    assert rows["aten::mm"].self_cpu_ms == pytest.approx(3.52)
    assert rows["aten::mm"].calls == 40


# -- artifact inspection ---------------------------------------------------


def test_torch_trace_is_empty_detects_missing_output(tmp_path: Path) -> None:
    assert profiler.torch_trace_is_empty(tmp_path)
    (tmp_path / "worker0.pt.trace.json.gz").write_bytes(b"x")
    assert not profiler.torch_trace_is_empty(tmp_path)


def test_op_table_file_also_counts_as_output(tmp_path: Path) -> None:
    (tmp_path / "profiler_out_0.txt").write_text("x", encoding="utf-8")
    assert not profiler.torch_trace_is_empty(tmp_path)


def test_index_lrt_traces_matches_naming(tmp_path: Path) -> None:
    (tmp_path / "lrt_trace_iter0_pid0").mkdir()
    (tmp_path / "lrt_trace_iter0_pid0" / "trace.bin").write_bytes(b"1234")
    (tmp_path / "lrt_trace_iter1_pid0").mkdir()
    (tmp_path / "unrelated_dir").mkdir()
    indexed = profiler.index_lrt_traces(tmp_path)
    assert len(indexed) == 2
    assert indexed[0].kind == "lrt_trace"
    assert "iter=0" in indexed[0].note
    assert indexed[0].size_bytes == 4


def test_index_torch_traces_classifies_kinds(tmp_path: Path) -> None:
    (tmp_path / "profiler_out_0.txt").write_text("x", encoding="utf-8")
    (tmp_path / "w.pt.trace.json.gz").write_bytes(b"x")
    (tmp_path / "notes.md").write_text("x", encoding="utf-8")
    indexed = profiler.index_torch_traces(tmp_path)
    kinds = {item.path.name: item.kind for item in indexed}
    assert kinds["profiler_out_0.txt"] == "op_table"
    assert kinds["w.pt.trace.json.gz"] == "torch_trace"
    assert kinds["notes.md"] == "other"


def test_write_op_csvs_includes_host_only_caveat(tmp_path: Path) -> None:
    written = profiler.write_op_csvs(
        tmp_path, {"cfg_a": {"base": base_rows(), "new": new_rows()}}, top=5
    )
    assert len(written) == 2
    for path in written:
        text = path.read_text(encoding="utf-8")
        assert "host-side" in text.splitlines()[0]
    diff_text = (tmp_path / "profile_ops_diff.csv").read_text(encoding="utf-8")
    assert "qaic::fused_rmsnorm" in diff_text
    assert "added" in diff_text
