# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Discover scheduler jobs from pytest's own collection. A "job" is one
xdist-loadscope-equivalent scope group, mirroring `_item_scope` in
`tests/e2e/conftest.py`.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_MARKER = "qaic_test_config"


def _is_aot_mode() -> bool:
    return importlib.util.find_spec("torch_qaic") is None


def _marker_kwargs(marker):
    if marker is None:
        return {}
    if marker.args:
        mode = "aot" if _is_aot_mode() else "eager"
        return marker.args[0].get(mode, {})
    return marker.kwargs


def _resolve(item, config, key, default=None):
    marker = item.get_closest_marker(_MARKER)
    kwargs = _marker_kwargs(marker)
    if key in kwargs:
        return kwargs[key]
    cli_value = config.getoption(key, default=None)
    return cli_value if cli_value is not None else default


def _item_scope(item) -> str:
    """Mirror `conftest._item_scope`: class-scoped tests share one job,
    a bare function's scope is its module."""
    return item.nodeid.rsplit("::", 1)[0]


def _num_devices(representative, config) -> int:
    """Device requirement for a scope. Two mutually-exclusive shapes exist:
    - `device_groups`-style tests: num_device_groups * device_group_size.
    - disaggregated-serving tests (`disagg_server`/`_disagg_test_config`):
      num_prefill_workers*prefill_device_group_size +
      num_decode_workers*decode_device_group_size.
    """
    marker = representative.get_closest_marker(_MARKER)
    kwargs = _marker_kwargs(marker)
    if "num_prefill_workers" in kwargs or "num_decode_workers" in kwargs:
        return kwargs.get("num_prefill_workers", 0) * kwargs.get(
            "prefill_device_group_size", 1
        ) + kwargs.get("num_decode_workers", 0) * kwargs.get(
            "decode_device_group_size", 1
        )
    num_device_groups = _resolve(representative, config, "num_device_groups", default=1)
    device_group_size = _resolve(representative, config, "device_group_size", default=1)
    return num_device_groups * device_group_size


class _JobCollectorPlugin:
    def __init__(self, base_args):
        self.base_args = base_args
        self.jobs = []

    def pytest_collection_modifyitems(self, session, config, items):
        # Drop tests a conftest.py hook already marked skipped (device-pool
        # sizing, qaic_aot_mode, qaic_disagg_installed, etc.)
        skipped, kept = [], []
        for item in items:
            marker = item.get_closest_marker("skip")
            (skipped if marker is not None else kept).append((item, marker))
        for item, marker in skipped:
            reason = marker.kwargs.get("reason", "") if marker.kwargs else ""
            print(f"Skipping {item.nodeid}: {reason}", file=sys.stderr)
        items[:] = [item for item, _ in kept]

        scopes: dict[str, list] = {}
        for item in items:
            scopes.setdefault(_item_scope(item), []).append(item)

        for scope_id, scope_items in scopes.items():
            representative = scope_items[0]
            model_name = _resolve(representative, config, "model_name")
            seq_len = _resolve(representative, config, "seq_len", default=128)
            ctx_len = _resolve(representative, config, "ctx_len")
            decode_bsz = config.getoption("decode_bsz")
            dtype = _resolve(representative, config, "dtype")
            kv_dtype = _resolve(representative, config, "kv_dtype", default="auto")

            self.jobs.append(
                {
                    "scope_id": scope_id,
                    "nodeids": [item.nodeid for item in scope_items],
                    "model_name": model_name,
                    "config_key": [
                        model_name,
                        seq_len,
                        ctx_len,
                        decode_bsz,
                        dtype,
                        kv_dtype,
                    ],
                    "num_devices": _num_devices(representative, config),
                    "base_args": self.base_args,
                }
            )

        # Nothing to run for the collect-only session - it exists purely to
        # populate self.jobs above.
        items[:] = []


def collect_jobs(test_paths, extra_pytest_args) -> list[dict]:
    plugin = _JobCollectorPlugin(base_args=extra_pytest_args)
    args = [
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        *test_paths,
        *extra_pytest_args,
    ]
    exit_code = pytest.main(args, plugins=[plugin])
    if exit_code not in (0, pytest.ExitCode.NO_TESTS_COLLECTED):
        raise RuntimeError(
            f"pytest collection failed (exit code {exit_code}) for args: {args}"
        )
    return plugin.jobs


def main() -> int:
    # Split raw argv on the first bare "--" ourselves: argparse's `nargs='+'`
    # for test_paths would otherwise greedily swallow everything meant for
    # pytest_args before REMAINDER ever sees it.
    raw_args = sys.argv[1:]
    if "--" in raw_args:
        sep = raw_args.index("--")
        own_args, pytest_args = raw_args[:sep], raw_args[sep + 1 :]
    else:
        own_args, pytest_args = raw_args, []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "test_paths", nargs="+", help="pytest test paths/files to collect from"
    )
    parser.add_argument(
        "--output", required=True, help="path to write the jobs JSON file to"
    )
    args = parser.parse_args(own_args)

    jobs = collect_jobs(args.test_paths, pytest_args)
    Path(args.output).write_text(json.dumps(jobs, indent=2))
    print(f"Collected {len(jobs)} job(s) -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
