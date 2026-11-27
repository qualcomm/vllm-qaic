# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""vLLM ``ProfilerConfig`` wiring and profile-artifact analysis.

Profiling in vllm-qaic is PyT/eager-only. Verified in-tree:

* ``QaicWorkerPyt.profile()`` builds a ``TorchProfilerWrapper`` from
  ``vllm_config.profiler_config``, requesting ``activities=["CPU"]`` -- there is
  no torch-level device activity on QAIC, so traces show host time only.
* ``QaicWorkerAoT.profile()`` logs "profile is not supported with QAIC AoT
  mode" and returns. ``/start_profile`` answers 200 and produces nothing.
* The QAIC LRT device-trace hook (``optional_qaic_profiling``, keyed on
  ``VLLM_TORCH_QAIC_PROFILER_DIR``) exists only in ``QaicModelRunnerPyt``;
  ``QaicModelRunnerAoT`` initialises ``profiling_iter`` but never uses it.

:mod:`vqperf.config` rejects profiling under ``--mode aot`` at parse time on
that basis, so nothing here has to cope with the silent-no-op case.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import ProfileSpec

logger = logging.getLogger("vqperf.profiler")

#: Host (torch) profiling is configured only through ``--profiler-config``.
#: Every vLLM release vllm-qaic pins (0.23 onward) has ``ProfilerConfig``, and
#: the old ``VLLM_TORCH_PROFILER_DIR`` env var is gone upstream, so there is no
#: env-var fallback: an environment without the flag cannot torch-profile.
ProfilerStyle = Literal["profiler_config", "unavailable"]

#: Env var read by ``QaicModelRunnerPyt`` (via ``vllm_qaic/envs.py``) for QAIC
#: LRT device traces.
QAIC_LRT_ENV = "VLLM_TORCH_QAIC_PROFILER_DIR"

LRT_TRACE_DIR_RE = re.compile(r"^lrt_trace_iter(\d+)_pid(\d+)$")

_DURATION_RE = re.compile(r"^([0-9]*\.?[0-9]+)\s*(ns|us|ms|s)$", re.IGNORECASE)
_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}


class ProfilingUnsupported(Exception):
    """Raised when the requested profiling cannot work in this environment."""


# ---------------------------------------------------------------------------
# server wiring
# ---------------------------------------------------------------------------


def torch_profiler_flags(spec: ProfileSpec, trace_dir: Path) -> list[str]:
    """Dotted ``--profiler-config.*`` flags for ``vllm serve``.

    ``trace_dir`` must be absolute: upstream abspath-normalises it anyway, and
    passing it pre-resolved keeps the trace where the report expects it.
    """
    if not spec.wants_torch:
        return []
    if not trace_dir.is_absolute():
        raise ValueError(f"trace_dir must be absolute, got {trace_dir}")

    def boolean(value: bool) -> str:
        return "true" if value else "false"

    # torch_profiler_dir is only valid when profiler == "torch", and is
    # *required* when it is -- both violations raise in ProfilerConfig's
    # validator. We always emit the pair together.
    flags = [
        "--profiler-config.profiler=torch",
        f"--profiler-config.torch_profiler_dir={trace_dir}",
        f"--profiler-config.wait_iterations={spec.wait_iterations}",
        f"--profiler-config.warmup_iterations={spec.warmup_iterations}",
        f"--profiler-config.active_iterations={spec.active_iterations}",
        f"--profiler-config.delay_iterations={spec.delay_iterations}",
        f"--profiler-config.max_iterations={spec.max_iterations}",
        f"--profiler-config.torch_profiler_with_stack={boolean(spec.with_stack)}",
        f"--profiler-config.torch_profiler_record_shapes={boolean(spec.record_shapes)}",
        f"--profiler-config.torch_profiler_with_memory={boolean(spec.with_memory)}",
        f"--profiler-config.torch_profiler_with_flops={boolean(spec.with_flops)}",
        f"--profiler-config.detailed_trace_annotation="
        f"{boolean(spec.detailed_annotation)}",
        f"--profiler-config.ignore_frontend={boolean(spec.resolved_ignore_frontend())}",
    ]
    return flags


def lrt_env(spec: ProfileSpec, lrt_dir: Path) -> dict[str, str]:
    """Env for QAIC LRT device traces.

    Not request-scoped: there is no start/stop endpoint, so every forward pass
    from server start writes a trace directory. Callers keep the prompt count
    small for this reason.
    """
    if not spec.wants_lrt:
        return {}
    return {QAIC_LRT_ENV: str(lrt_dir)}


def describe_spec(spec: ProfileSpec) -> dict[str, str]:
    """Flat record of the resolved profiling knobs, for the manifest."""
    if not spec.enabled:
        return {"profile_mode": "off"}
    return {
        "profile_mode": spec.mode,
        "wait_iterations": str(spec.wait_iterations),
        "warmup_iterations": str(spec.warmup_iterations),
        "active_iterations": str(spec.active_iterations),
        "delay_iterations": str(spec.delay_iterations),
        "max_iterations": str(spec.max_iterations),
        "with_stack": str(spec.with_stack),
        "record_shapes": str(spec.record_shapes),
        "with_memory": str(spec.with_memory),
        "with_flops": str(spec.with_flops),
        "detailed_trace_annotation": str(spec.detailed_annotation),
        "ignore_frontend": str(spec.resolved_ignore_frontend()),
        "activities": "CPU (host-side only; QAIC has no torch device activity)",
    }


# ---------------------------------------------------------------------------
# artifact inspection
# ---------------------------------------------------------------------------

TRACE_SUFFIXES = (".pt.trace.json", ".pt.trace.json.gz", ".json", ".json.gz")


@dataclass
class ArtifactIndex:
    kind: str
    path: Path
    size_bytes: int
    note: str = ""


def index_torch_traces(trace_dir: Path) -> list[ArtifactIndex]:
    out: list[ArtifactIndex] = []
    if not trace_dir.exists():
        return out
    for path in sorted(trace_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name.startswith("profiler_out_") and name.endswith(".txt"):
            kind = "op_table"
        elif any(name.endswith(suffix) for suffix in TRACE_SUFFIXES):
            kind = "torch_trace"
        else:
            kind = "other"
        out.append(ArtifactIndex(kind, path, path.stat().st_size))
    return out


def index_lrt_traces(lrt_dir: Path) -> list[ArtifactIndex]:
    """Index ``lrt_trace_iter<N>_pid<rank>/`` directories.

    LRT payloads are binary and analysed with Cloud AI SDK tooling, so they are
    indexed and cross-linked rather than parsed.
    """
    out: list[ArtifactIndex] = []
    if not lrt_dir.exists():
        return out
    for path in sorted(lrt_dir.iterdir()):
        match = LRT_TRACE_DIR_RE.match(path.name)
        if not path.is_dir() or not match:
            continue
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        out.append(
            ArtifactIndex(
                "lrt_trace",
                path,
                size,
                note=f"iter={match.group(1)} pid={match.group(2)}",
            )
        )
    return out


def torch_trace_is_empty(trace_dir: Path) -> bool:
    """True when no torch trace landed.

    On PyT a 200 from /start_profile with no files means the profiler never
    started; that must be a cell failure, not a silent pass.
    """
    return not any(
        item.kind in ("torch_trace", "op_table")
        for item in index_torch_traces(trace_dir)
    )


# ---------------------------------------------------------------------------
# op-table parsing
# ---------------------------------------------------------------------------


@dataclass
class OpRow:
    name: str
    self_cpu_ms: float | None
    cpu_total_ms: float | None
    calls: int | None
    self_cpu_pct: float | None = None


def parse_duration_ms(raw: str) -> float | None:
    """Parse a torch-profiler duration cell (``1.234ms``) into milliseconds."""
    match = _DURATION_RE.match(raw.strip())
    if not match:
        return None
    return float(match.group(1)) * _TO_MS[match.group(2).lower()]


def parse_op_table(text: str) -> list[OpRow]:
    """Parse the ``key_averages().table()`` text written by vLLM.

    Column widths are taken from the profiler's own dashed rule line, which is
    the only reliable delimiter -- operator names contain spaces, and numeric
    cells contain unit suffixes.
    """
    lines = text.splitlines()
    spans: list[tuple[int, int]] = []
    header_index = -1

    for index, line in enumerate(lines):
        if not _is_rule(line):
            continue
        candidate = _dash_spans(line)
        if len(candidate) < 3:
            continue
        # The header row follows the first rule line.
        if index + 1 < len(lines) and "Name" in lines[index + 1]:
            spans = candidate
            header_index = index + 1
            break

    if header_index < 0 or not spans:
        return []

    columns = [_slice(lines[header_index], span) for span in spans]
    col_index = {_normalize(name): pos for pos, name in enumerate(columns)}

    def column_for(*candidates: str) -> int | None:
        for candidate in candidates:
            if candidate in col_index:
                return col_index[candidate]
        return None

    name_col = column_for("name")
    if name_col is None:
        return []
    self_col = column_for("selfcpu", "selfcputotal")
    total_col = column_for("cputotal")
    calls_col = column_for("ofcalls", "numberofcalls", "calls")
    pct_col = column_for("selfcpu%")

    rows: list[OpRow] = []
    for line in lines[header_index + 2 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_rule(line):
            break
        if stripped.lower().startswith(("self cpu time total", "self cpu total")):
            break
        cells = [_slice(line, span) for span in spans]
        name = cells[name_col] if name_col < len(cells) else ""
        if not name:
            continue
        rows.append(
            OpRow(
                name=name,
                self_cpu_ms=_cell_duration(cells, self_col),
                cpu_total_ms=_cell_duration(cells, total_col),
                calls=_cell_int(cells, calls_col),
                self_cpu_pct=_cell_pct(cells, pct_col),
            )
        )
    return rows


def _is_rule(line: str) -> bool:
    """True for a table rule line: dashes separated by runs of spaces."""
    squeezed = line.replace(" ", "")
    return bool(squeezed) and set(squeezed) == {"-"}


def _dash_spans(line: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"-+", line)]


def _slice(line: str, span: tuple[int, int]) -> str:
    return line[span[0] : span[1]].strip()


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9%]", "", name.lower())


def _cell_duration(cells: list[str], index: int | None) -> float | None:
    if index is None or index >= len(cells):
        return None
    return parse_duration_ms(cells[index])


def _cell_int(cells: list[str], index: int | None) -> int | None:
    if index is None or index >= len(cells):
        return None
    try:
        return int(cells[index].replace(",", ""))
    except ValueError:
        return None


def _cell_pct(cells: list[str], index: int | None) -> float | None:
    if index is None or index >= len(cells):
        return None
    try:
        return float(cells[index].rstrip("%"))
    except ValueError:
        return None


def load_op_table(trace_dir: Path) -> list[OpRow]:
    """Load and merge every ``profiler_out_<rank>.txt`` under ``trace_dir``."""
    merged: dict[str, OpRow] = {}
    for item in index_torch_traces(trace_dir):
        if item.kind != "op_table":
            continue
        try:
            text = item.path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # pragma: no cover - unreadable artifact
            continue
        for row in parse_op_table(text):
            existing = merged.get(row.name)
            if existing is None:
                merged[row.name] = row
                continue
            existing.self_cpu_ms = _add(existing.self_cpu_ms, row.self_cpu_ms)
            existing.cpu_total_ms = _add(existing.cpu_total_ms, row.cpu_total_ms)
            existing.calls = _add_int(existing.calls, row.calls)
    return sorted(merged.values(), key=lambda r: -(r.self_cpu_ms or 0.0))


def _add(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _add_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


# ---------------------------------------------------------------------------
# op-table diff
# ---------------------------------------------------------------------------


@dataclass
class OpDiff:
    name: str
    status: str  # both | added | removed
    base_self_ms: float | None
    new_self_ms: float | None
    delta_self_ms: float | None
    delta_self_pct: float | None
    base_total_ms: float | None
    new_total_ms: float | None
    base_calls: int | None
    new_calls: int | None
    rank_base: int | None
    rank_new: int | None


def diff_op_tables(
    base: list[OpRow], new: list[OpRow], *, top: int = 30
) -> list[OpDiff]:
    """Diff two op tables, largest absolute self-time movement first.

    Keeps the union of each side's top-N as well as the biggest movers, so an
    operator that is hot in only one commit still appears.
    """
    base_by_name = {row.name: row for row in base}
    new_by_name = {row.name: row for row in new}
    base_rank = {row.name: pos + 1 for pos, row in enumerate(base)}
    new_rank = {row.name: pos + 1 for pos, row in enumerate(new)}

    diffs: list[OpDiff] = []
    for name in set(base_by_name) | set(new_by_name):
        base_row = base_by_name.get(name)
        new_row = new_by_name.get(name)
        base_self = base_row.self_cpu_ms if base_row else None
        new_self = new_row.self_cpu_ms if new_row else None

        if base_row is None:
            status = "added"
        elif new_row is None:
            status = "removed"
        else:
            status = "both"

        delta = None
        delta_pct = None
        if base_self is not None and new_self is not None:
            delta = new_self - base_self
            delta_pct = (100.0 * delta / base_self) if base_self else None

        diffs.append(
            OpDiff(
                name=name,
                status=status,
                base_self_ms=base_self,
                new_self_ms=new_self,
                delta_self_ms=delta,
                delta_self_pct=delta_pct,
                base_total_ms=base_row.cpu_total_ms if base_row else None,
                new_total_ms=new_row.cpu_total_ms if new_row else None,
                base_calls=base_row.calls if base_row else None,
                new_calls=new_row.calls if new_row else None,
                rank_base=base_rank.get(name),
                rank_new=new_rank.get(name),
            )
        )

    keep = {row.name for row in base[:top]} | {row.name for row in new[:top]}
    movers = sorted(
        (d for d in diffs if d.delta_self_ms is not None),
        key=lambda d: -abs(d.delta_self_ms or 0.0),
    )[:top]
    keep.update(d.name for d in movers)
    # An op present on only one side is usually the most informative row after
    # a kernel or patch change; never drop those.
    keep.update(d.name for d in diffs if d.status != "both")

    selected = [d for d in diffs if d.name in keep]
    selected.sort(key=lambda d: (-abs(d.delta_self_ms or 0.0), -(d.new_self_ms or 0.0)))
    return selected


HOST_ONLY_CAVEAT = (
    "activities=CPU only (vllm_qaic/worker/worker.py requests ['CPU']): these "
    "are host-side times. A regression entirely on-device appears in the "
    "throughput/latency comparison and NOT in this table."
)


def write_op_csvs(
    out_dir: Path,
    per_config: dict[str, dict[str, list[OpRow]]],
    *,
    top: int,
) -> list[Path]:
    """Write ``profile_ops.csv`` and ``profile_ops_diff.csv``.

    ``per_config`` maps ``config_id -> {"base": rows, "new": rows}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    ops_path = out_dir / "profile_ops.csv"
    with ops_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# {HOST_ONLY_CAVEAT}"])
        writer.writerow(
            ["config_id", "side", "rank", "op", "self_cpu_ms", "cpu_total_ms", "calls"]
        )
        for config_id, sides in sorted(per_config.items()):
            for side, rows in sorted(sides.items()):
                for rank, row in enumerate(rows, start=1):
                    writer.writerow(
                        [
                            config_id,
                            side,
                            rank,
                            row.name,
                            _fmt(row.self_cpu_ms),
                            _fmt(row.cpu_total_ms),
                            "" if row.calls is None else row.calls,
                        ]
                    )
    written.append(ops_path)

    diff_path = out_dir / "profile_ops_diff.csv"
    with diff_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"# {HOST_ONLY_CAVEAT}"])
        writer.writerow(
            [
                "config_id",
                "op",
                "status",
                "base_self_cpu_ms",
                "new_self_cpu_ms",
                "delta_self_cpu_ms",
                "delta_self_cpu_pct",
                "base_cpu_total_ms",
                "new_cpu_total_ms",
                "base_calls",
                "new_calls",
                "rank_base",
                "rank_new",
            ]
        )
        for config_id, sides in sorted(per_config.items()):
            diffs = diff_op_tables(sides.get("base", []), sides.get("new", []), top=top)
            for diff in diffs:
                writer.writerow(
                    [
                        config_id,
                        diff.name,
                        diff.status,
                        _fmt(diff.base_self_ms),
                        _fmt(diff.new_self_ms),
                        _fmt(diff.delta_self_ms),
                        _fmt(diff.delta_self_pct),
                        _fmt(diff.base_total_ms),
                        _fmt(diff.new_total_ms),
                        "" if diff.base_calls is None else diff.base_calls,
                        "" if diff.new_calls is None else diff.new_calls,
                        "" if diff.rank_base is None else diff.rank_base,
                        "" if diff.rank_new is None else diff.rank_new,
                    ]
                )
    written.append(diff_path)
    return written


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"
