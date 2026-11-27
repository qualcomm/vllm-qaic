# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Argv construction for the server and the serving benchmark client."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from vqperf import bench
from vqperf.config import parse_configs
from vqperf.env import BenchInvocation
from vqperf.util import RunResult


def one(tmp_path: Path, text: str, *, mode: str = "aot"):
    path = tmp_path / "configs.csv"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return parse_configs(path, mode=mode)[0]


MINIMAL = """\
config_id,model,max_model_len,num_prompts,random_input_len,random_output_len
cfg_a,meta-llama/Llama-3.1-8B-Instruct,2048,64,1024,128
"""


def pairs(argv: list[str]) -> dict[str, str]:
    out = {}
    for index, token in enumerate(argv):
        if token.startswith("--") and index + 1 < len(argv):
            out.setdefault(token, argv[index + 1])
    return out


# -- server argv -----------------------------------------------------------


def test_model_is_positional_not_a_flag(tmp_path: Path) -> None:
    # `vllm serve <model>` takes the model positionally; passing --model too
    # is an error in some versions and ignored in others.
    argv = one(tmp_path, MINIMAL).server_args()
    assert argv[0] == "meta-llama/Llama-3.1-8B-Instruct"
    assert "--model" not in argv


def test_server_argv_carries_typed_columns(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,max_num_seqs,"
        "tensor_parallel_size,kv_cache_dtype\ncfg_a,m,2048,8,16,4,fp8\n"
    )
    flags = pairs(one(tmp_path, text).server_args())
    assert flags["--max-model-len"] == "2048"
    assert flags["--max-num-seqs"] == "16"
    assert flags["--tensor-parallel-size"] == "4"
    assert flags["--kv-cache-dtype"] == "fp8"


def test_tensor_parallel_size_omitted_when_one(tmp_path: Path) -> None:
    # Emitting --tensor-parallel-size 1 is noise, and on some SHAs it changes
    # the executor selection path.
    assert "--tensor-parallel-size" not in one(tmp_path, MINIMAL).server_args()


def test_extra_server_args_appended_last(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,extra_server_args\n"
        "cfg_a,m,2048,8,--enforce-eager --swap-space 2\n"
    )
    argv = one(tmp_path, text).server_args()
    assert argv[-3:] == ["--enforce-eager", "--swap-space", "2"]
    assert one(tmp_path, text).server_args(include_extras=False)[-1] == "8"


# -- client argv -----------------------------------------------------------


def test_client_argv_shape(tmp_path: Path) -> None:
    flags = pairs(one(tmp_path, MINIMAL).client_args())
    assert flags["--model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert flags["--dataset-name"] == "random"
    assert flags["--num-prompts"] == "64"
    assert flags["--random-input-len"] == "1024"
    assert flags["--random-output-len"] == "128"


def test_num_prompts_override_only_affects_that_flag(tmp_path: Path) -> None:
    config = one(tmp_path, MINIMAL)
    assert pairs(config.client_args(num_prompts=4))["--num-prompts"] == "4"
    assert pairs(config.client_args())["--num-prompts"] == "64"


def test_ignore_eos_emitted_as_a_bare_flag(tmp_path: Path) -> None:
    # Upstream --ignore-eos is store_true; "--ignore-eos true" would be read
    # as a positional and rejected.
    argv = one(tmp_path, MINIMAL).client_args()
    assert "--ignore-eos" in argv
    assert "true" not in argv


# -- full bench command ----------------------------------------------------


class FakeServer:
    base_url = "http://127.0.0.1:8321"


def capture(monkeypatch, tmp_path: Path, **kwargs) -> list[str]:
    seen: list[str] = []

    def fake_run(argv, **_):
        seen.extend(argv)
        return RunResult(argv=argv, returncode=1, duration_s=0.5, log_path=None)

    monkeypatch.setattr(bench, "run", fake_run)
    env = type("E", (), {"bench": BenchInvocation(["vllm", "bench", "serve"], "cli")})()
    bench.run_benchmark(
        env,
        one(tmp_path, MINIMAL),
        FakeServer(),
        cell_dir=tmp_path / "cell",
        sha="a" * 40,
        repeat=0,
        **kwargs,
    )
    return seen


def test_bench_argv_pins_the_result_path(monkeypatch, tmp_path: Path) -> None:
    # Without --result-filename the name embeds a timestamp and the report
    # cannot find the file it just produced.
    argv = capture(monkeypatch, tmp_path)
    assert argv[:3] == ["vllm", "bench", "serve"]
    assert "--save-result" in argv
    flags = pairs(argv)
    assert flags["--result-dir"] == str(tmp_path / "cell")
    assert flags["--result-filename"] == bench.RESULT_FILENAME
    assert flags["--base-url"] == "http://127.0.0.1:8321"
    assert flags["--backend"] == "openai-chat"


def test_metadata_is_multi_valued_and_last(monkeypatch, tmp_path: Path) -> None:
    # --metadata takes nargs="*", so every token after it is swallowed; it has
    # to come last or it eats the following flag.
    argv = capture(monkeypatch, tmp_path, label="profile")
    tail = argv[argv.index("--metadata") :]
    assert tail == [
        "--metadata",
        f"vqperf_sha={'a' * 40}",
        "vqperf_config=cfg_a",
        "vqperf_repeat=0",
        "vqperf_pass=profile",
    ]


def test_profile_flag_only_when_requested(monkeypatch, tmp_path: Path) -> None:
    assert "--profile" not in capture(monkeypatch, tmp_path)
    assert "--profile" in capture(monkeypatch, tmp_path, profile=True)


# -- outcome interpretation ------------------------------------------------


def interpret(
    tmp_path: Path, outcome: RunResult, payload: dict | None
) -> bench.BenchOutcome:
    result_path = tmp_path / "result.json"
    if payload is not None:
        result_path.write_text(json.dumps(payload), encoding="utf-8")
    log_path = tmp_path / "bench.log"
    log_path.write_text("some log output\n", encoding="utf-8")
    return bench._interpret(
        outcome, one(tmp_path, MINIMAL), result_path, log_path, label="measure"
    )


def ok_run() -> RunResult:
    return RunResult(argv=["x"], returncode=0, duration_s=1.0, log_path=None)


def test_timeout_reported_as_timeout(tmp_path: Path) -> None:
    timed_out = RunResult(
        argv=["x"], returncode=-9, duration_s=42.0, log_path=None, timed_out=True
    )
    outcome = interpret(tmp_path, timed_out, {"completed": 64})
    assert outcome.status == "failed"
    assert "bench_timeout" in outcome.reason


def test_nonzero_exit_includes_log_tail(tmp_path: Path) -> None:
    failed = RunResult(argv=["x"], returncode=2, duration_s=1.0, log_path=None)
    outcome = interpret(tmp_path, failed, None)
    assert outcome.status == "failed"
    assert "rc=2" in outcome.reason
    assert "some log output" in outcome.reason


def test_missing_result_json_is_a_failure(tmp_path: Path) -> None:
    assert interpret(tmp_path, ok_run(), None).status == "failed"


def test_unreadable_result_json_is_a_failure(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text("{not json", encoding="utf-8")
    outcome = bench._interpret(
        ok_run(),
        one(tmp_path, MINIMAL),
        tmp_path / "result.json",
        tmp_path / "bench.log",
        label="measure",
    )
    assert outcome.status == "failed"
    assert "unreadable" in outcome.reason


def test_partial_completion_keeps_the_numbers_but_flags_them(tmp_path: Path) -> None:
    outcome = interpret(tmp_path, ok_run(), {"completed": 30, "num_prompts": 64})
    assert outcome.status == "partial"
    assert outcome.ok  # still usable, just caveated
    assert "30/64" in outcome.reason


def test_full_completion_is_ok(tmp_path: Path) -> None:
    outcome = interpret(tmp_path, ok_run(), {"completed": 64, "num_prompts": 64})
    assert outcome.status == "ok"
    assert outcome.reason == ""
