# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import json
import os

import pytest
from huggingface_hub import snapshot_download
from vllm.entrypoints.openai.models.protocol import LoRAModulePath

from vllm_qaic.model_loader.qaic import (
    search_adapters_in_cache,
    verify_adaptername_to_id_consistency,
)

BASE_MODEL_NAME = "PY007/TinyLlama-1.1B-Chat-v0.3"
ADAPTER_ID_0 = "jashing/tinyllama-colorist-lora"
ADAPTER_ID_1 = "jashing/tinyllama-energy-lora"


@pytest.mark.qaic_aot_mode
def test_qaic_search_adapters_in_cache(tmp_path):
    """Test adapter search functionality in cache."""
    hf_home = os.environ.get("HF_HOME", None)
    os.environ["HF_HOME"] = str(tmp_path)

    adapter_list_before = search_adapters_in_cache(BASE_MODEL_NAME)

    snapshot_download(repo_id=ADAPTER_ID_0, cache_dir=f"{tmp_path}/hub")
    snapshot_download(repo_id=ADAPTER_ID_1, cache_dir=f"{tmp_path}/hub")

    # cache_adapter_search_and_load
    adapter_list_after = search_adapters_in_cache(BASE_MODEL_NAME)

    assert len(adapter_list_after) == len(adapter_list_before) + 2

    # revert HF_HOME setting
    os.environ["HF_HOME"] = hf_home

    # test time: 7.55s (first case) v.s. 225.02s (2 cases)


@pytest.mark.qaic_aot_mode
def test_qaic_get_qaic_model_dump_adaptername_to_id(tmp_path):
    """Test adapter name to ID mapping and persistence."""
    from QEfficient.peft.lora import QEffAutoLoraModelForCausalLM

    qeff_model = QEffAutoLoraModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_NAME,
        num_hidden_layers=1,
    )
    qeff_model.load_adapter(ADAPTER_ID_0, "adapter_0")
    qeff_model.load_adapter(ADAPTER_ID_1, "adapter_1")

    # dump adaptername_to_id to folder for the first compilation
    lora_config = True
    lora_modules = [
        LoRAModulePath(name="adapter_0", path="/path_0"),
        LoRAModulePath(name="adapter_1", path="/path_1"),
    ]

    qpc_path = tmp_path
    adaptername_to_id = qeff_model.active_adapter_to_id
    if lora_config and not os.path.exists(f"{qpc_path}/adaptername_to_id.json"):
        with open(f"{qpc_path}/adaptername_to_id.json", "w") as file:
            json.dump(adaptername_to_id, file)

    # load adaptername_to_id to folder for correct qpc_path passed in
    qpc_path = tmp_path
    adaptername_to_id = {}
    if lora_config and qpc_path:
        # check if json file exist
        if os.path.exists(f"{qpc_path}/adaptername_to_id.json"):
            with open(f"{qpc_path}/adaptername_to_id.json") as file:
                adaptername_to_id = json.load(file)
        else:
            raise FileNotFoundError(
                f"The file at {qpc_path}/adaptername_to_id.json was not found. "
                f"Please provide a correct VLLM_QAIC_QPC_PATH."
            )

    assert len(adaptername_to_id) == 2

    # load adaptername_to_id to folder for wrong qpc_path passed in
    qpc_path = f"{tmp_path}/tmp"
    with pytest.raises(FileNotFoundError):
        adaptername_to_id = {}
        if lora_config and qpc_path:
            # check if json file exist
            if os.path.exists(f"{qpc_path}/adaptername_to_id.json"):
                with open(f"{qpc_path}/adaptername_to_id.json") as file:
                    adaptername_to_id = json.load(file)
            else:
                raise FileNotFoundError(
                    f"The file at {qpc_path}/adaptername_to_id.json was not "
                    f"found. Please provide a correct VLLM_QAIC_QPC_PATH."
                )

    # load inconsistent file content in adaptername_to_id.json
    qpc_path = tmp_path
    with open(f"{qpc_path}/adaptername_to_id.json", "w") as file:
        json.dump({"abc": 0, "def": 1}, file)

    adaptername_to_id = {}
    with pytest.raises(ValueError):
        if lora_config and qpc_path:
            # check if json file exist
            if os.path.exists(f"{qpc_path}/adaptername_to_id.json"):
                with open(f"{qpc_path}/adaptername_to_id.json") as file:
                    adaptername_to_id = json.load(file)

            # check if json file content is correct
            if not verify_adaptername_to_id_consistency(
                adaptername_to_id, lora_modules
            ):
                raise ValueError(
                    f"Inconsistent file content in "
                    f"{qpc_path}/adaptername_to_id.json and input lora modules."
                )
