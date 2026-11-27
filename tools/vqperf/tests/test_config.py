# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Config CSV schema and validation tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from vqperf.config import ConfigError, parse_configs

VALID = """\
config_id,model,max_model_len,num_prompts,random_input_len,random_output_len
cfg_a,meta-llama/Llama-3.1-8B-Instruct,2048,64,1024,128
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "configs.csv"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_parses_minimal_row(tmp_path: Path) -> None:
    configs = parse_configs(write(tmp_path, VALID), mode="aot")
    assert len(configs) == 1
    config = configs[0]
    assert config.config_id == "cfg_a"
    assert config.max_num_seqs == 8
    assert config.ignore_eos is True
    assert config.profile.mode == "off"


def test_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    text = "# a comment\n" + VALID + "\n# trailing\n"
    assert len(parse_configs(write(tmp_path, text), mode="aot")) == 1


def test_unknown_column_rejected(tmp_path: Path) -> None:
    text = VALID.replace("num_prompts", "num_prompt_count")
    with pytest.raises(ConfigError, match="unknown column"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_passthrough_label_column_allowed(tmp_path: Path) -> None:
    text = "config_id,model,max_model_len,num_prompts,x_note\ncfg_a,m,2048,8,sdk-1.23\n"
    configs = parse_configs(write(tmp_path, text), mode="aot")
    assert configs[0].labels == {"x_note": "sdk-1.23"}


def test_context_window_overflow_is_hard_error(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,random_input_len,"
        "random_output_len\ncfg_a,m,1024,8,1024,128\n"
    )
    with pytest.raises(ConfigError, match="max_model_len"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_duplicate_config_id_rejected(tmp_path: Path) -> None:
    text = VALID + "cfg_a,m,2048,8,1024,128\n"
    with pytest.raises(ConfigError, match="duplicate config_id"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_config_id_charset_enforced(tmp_path: Path) -> None:
    text = VALID.replace("cfg_a", "cfg a/b")
    with pytest.raises(ConfigError, match="config_id"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_all_errors_reported_together(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts\n"
        "bad one,m,2048,8\n"
        "cfg_b,m,notanint,8\n"
    )
    with pytest.raises(ConfigError) as excinfo:
        parse_configs(write(tmp_path, text), mode="aot")
    assert "2 problem(s)" in str(excinfo.value)


def test_duplicate_flag_detection(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,extra_server_args\n"
        "cfg_a,m,2048,8,--max-num-seqs 16\n"
    )
    with pytest.raises(ConfigError, match="duplicates typed column"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_reserved_flags_rejected(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,extra_server_args\n"
        "cfg_a,m,2048,8,--port 9000\n"
    )
    with pytest.raises(ConfigError, match="vqperf owns it"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_ignore_eos_false_warns(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,ignore_eos\ncfg_a,m,2048,8,false\n"
    )
    config = parse_configs(write(tmp_path, text), mode="aot")[0]
    assert any("ignore_eos=false" in warning for warning in config.warnings)


def test_saturation_row_warns(tmp_path: Path) -> None:
    config = parse_configs(write(tmp_path, VALID), mode="aot")[0]
    assert any("saturation throughput" in w for w in config.warnings)


def test_env_column_parsed(tmp_path: Path) -> None:
    text = "config_id,model,max_model_len,num_prompts,env\ncfg_a,m,2048,8,A=1;B=two\n"
    config = parse_configs(write(tmp_path, text), mode="aot")[0]
    assert config.env == {"A": "1", "B": "two"}


def test_client_args_include_random_only_for_random_dataset(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,dataset_name,dataset_path\n"
        "cfg_a,m,2048,8,sharegpt,/data/sharegpt.json\n"
    )
    config = parse_configs(write(tmp_path, text), mode="aot")[0]
    assert "--random-input-len" not in config.client_args()
    assert "--dataset-path" in config.client_args()


# -- profiling gates -------------------------------------------------------


def test_profile_rejected_in_aot_mode(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_mode\ncfg_a,m,2048,8,torch\n"
    )
    with pytest.raises(ConfigError, match="QaicWorkerAoT.profile"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_lrt_profile_rejected_in_aot_mode(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_mode\ncfg_a,m,2048,8,lrt\n"
    )
    with pytest.raises(ConfigError, match="QaicModelRunnerPyt"):
        parse_configs(write(tmp_path, text), mode="aot")


def test_profile_allowed_in_pyt_mode(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_mode\ncfg_a,m,2048,8,both\n"
    )
    config = parse_configs(write(tmp_path, text), mode="pyt")[0]
    assert config.profile.wants_torch and config.profile.wants_lrt


def test_profile_tuning_without_mode_is_error(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_with_stack\n"
        "cfg_a,m,2048,8,true\n"
    )
    with pytest.raises(ConfigError, match="profile_mode is 'off'"):
        parse_configs(write(tmp_path, text), mode="pyt")


def test_no_profile_overrides_csv(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_mode\ncfg_a,m,2048,8,torch\n"
    )
    configs = parse_configs(write(tmp_path, text), mode="aot", force_profile_off=True)
    assert configs[0].profile.mode == "off"


def test_default_profile_mode_applies(tmp_path: Path) -> None:
    config = parse_configs(
        write(tmp_path, VALID), mode="pyt", default_profile_mode="torch"
    )[0]
    assert config.profile.mode == "torch"


def test_active_iterations_lower_bound(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_mode,"
        "profile_active_iterations\ncfg_a,m,2048,8,torch,0\n"
    )
    with pytest.raises(ConfigError, match="profile_active_iterations"):
        parse_configs(write(tmp_path, text), mode="pyt")


def test_profile_prompt_count_defaults_to_capped_num_prompts(tmp_path: Path) -> None:
    text = (
        "config_id,model,max_model_len,num_prompts,profile_mode\n"
        "cfg_a,m,2048,64,torch\n"
    )
    config = parse_configs(write(tmp_path, text), mode="pyt")[0]
    assert config.effective_profile_prompts == 16
