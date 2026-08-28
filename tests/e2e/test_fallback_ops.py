# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Fallback-ops coverage: does an ATen op with no registered QAIC dispatch-key
kernel actually produce the "CPU fallback op: ..." lines
ci_scripts/eager/fallback_parser.py greps for, and does that parser (plus
parse_logs_to_excel.py's re-parsing of its printed output) correctly turn
those lines into the stats dict the Excel summary is built from.

The hardware test requires QAIC_DEBUG=1 and QAIC_VISIBLE_DEVICES (set to the
same list as --device-id) to be exported in the shell *before* pytest is
launched, e.g.:

    QAIC_DEBUG=1 QAIC_VISIBLE_DEVICES=9 python -m pytest \
        tests/e2e/test_fallback_ops.py --device-id 9

Both are read once by torch_qaic the first time it's imported, and it gets
imported during collection (tests/e2e/conftest.py's pytest_collection_modifyitems
checks current_platform.is_aot_inference() for every item) -- long before this
test's body runs. Setting either from inside the test function is too late:
QAIC_VISIBLE_DEVICES silently falls back to the unrestricted device list
(so "qaic:0" resolves to physical QID 0, not the device this test acquired
from the pool) and QAIC_DEBUG's fallback logging never turns on at all.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import regex as re
import torch

_EAGER_DIR = Path(__file__).resolve().parents[2] / "ci_scripts" / "eager"


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, _EAGER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fallback_parser = _load_module("fallback_parser", "fallback_parser.py")
parse_logs_to_excel = _load_module("parse_logs_to_excel", "parse_logs_to_excel.py")

# aten::std.correction has no QAIC dispatch-key kernel today, so it's a
# reliable, minimal trigger for the dispatcher's CPU-fallback path. If a
# future torch_qaic ever implements it natively, swap in another op that's
# still unimplemented.
_FALLBACK_OP = "aten::std.correction"

# A synthetic stand-in for a real QAIC_DEBUG=1 sweep log: two hits of one op
# and one hit of another, interleaved with unrelated output.
_SAMPLE_LOG = """\
Some unrelated startup line
CPU fallback op: aten::std.correction, execution started.
CPU fallback op: aten::std.correction, time elapsed (us): 422
some other line
CPU fallback op: aten::std.correction, execution started.
CPU fallback op: aten::std.correction, time elapsed (us): 295
CPU fallback op: aten::topk, execution started.
CPU fallback op: aten::topk, time elapsed (us): 10
"""


def test_extract_fallback_ops_counts_and_sums_time(tmp_path):
    log_file = tmp_path / "log_1_model_tp1.log"
    log_file.write_text(_SAMPLE_LOG)

    stats = fallback_parser.extract_fallback_ops(str(log_file))

    assert dict(stats) == {
        "aten::std.correction": {"tc": 2, "tt": 717},
        "aten::topk": {"tc": 1, "tt": 10},
    }


def test_extract_fallback_ops_empty_log_returns_empty_dict(tmp_path):
    log_file = tmp_path / "log_1_model_tp1.log"
    log_file.write_text("no fallback lines here\njust normal output\n")

    stats = fallback_parser.extract_fallback_ops(str(log_file))

    assert dict(stats) == {}


def test_parse_logs_to_excel_reads_back_fallback_parser_output(tmp_path, capsys):
    """extract_fallback_ops' printed defaultdict repr must still be exactly
    what parse_log_file's `defaultdict\\([^,]+,\\s*(\\{.*\\})\\)` regex expects."""
    log_file = tmp_path / "log_1_model_tp1.log"
    log_file.write_text(_SAMPLE_LOG)

    fallback_parser.extract_fallback_ops(str(log_file))
    printed = capsys.readouterr().out

    parse_log = tmp_path / "parse_1_model_tp1.log"
    parse_log.write_text(printed)

    ops = parse_logs_to_excel.parse_log_file(str(parse_log))

    assert ops == {
        "aten::std.correction": {"tc": 2, "tt": 717},
        "aten::topk": {"tc": 1, "tt": 10},
    }


class TestFallbackOps:
    def test_unimplemented_op_logs_cpu_fallback(
        self, capfd, tmp_path, device_group, device_pool_ids
    ):
        if (
            os.environ.get("QAIC_DEBUG") != "1"
            or "QAIC_VISIBLE_DEVICES" not in os.environ
        ):
            pytest.skip(
                "requires QAIC_DEBUG=1 and QAIC_VISIBLE_DEVICES (matching "
                "--device-id) exported before launching pytest -- see module "
                "docstring"
            )
        pytest.importorskip("torch_qaic")

        # QAIC_VISIBLE_DEVICES was fixed before this process started (see
        # docstring), so "qaic:<local index>" addresses devices by position
        # within that fixed list, not by physical QID.
        local_idx = device_pool_ids.index(device_group[0])

        x = torch.randn(8, 8, device=f"qaic:{local_idx}")
        torch.qaic.synchronize()
        capfd.readouterr()  # drop device-init/context-creation noise

        x.std()
        torch.qaic.synchronize()

        captured = capfd.readouterr()
        combined = captured.out + captured.err

        assert re.search(
            rf"CPU fallback op:\s*{re.escape(_FALLBACK_OP)},\s*execution started",
            combined,
        ), (
            f"expected the QAIC runtime to log a CPU-fallback 'execution "
            f"started' line for {_FALLBACK_OP}; captured output:\n{combined}"
        )
        assert re.search(
            rf"CPU fallback op:\s*{re.escape(_FALLBACK_OP)},"
            rf"\s*time elapsed \(us\):\s*\d+",
            combined,
        ), (
            f"expected the QAIC runtime to log a CPU-fallback 'time elapsed' "
            f"line for {_FALLBACK_OP}; captured output:\n{combined}"
        )

        # End-to-end: the real runtime output must also be consumable by our
        # own parser, not just by this test's inline regex.
        log_file = tmp_path / "log_1_direct_tp1.log"
        log_file.write_text(combined)
        stats = fallback_parser.extract_fallback_ops(str(log_file))
        assert stats[_FALLBACK_OP]["tc"] >= 1
        assert stats[_FALLBACK_OP]["tt"] > 0
