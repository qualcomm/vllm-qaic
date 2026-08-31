# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import gc
import os
import random
from time import perf_counter

import pytest
from huggingface_hub import snapshot_download
from vllm.entrypoints.openai.models.protocol import LoRAModulePath

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Server startup timeout: real QPC compilation for a 7B model can take up
# to ~30 min; ServerRunner's own default (300s) is sized for the smaller
# models used elsewhere in tests/e2e/, not this file's Mistral-7B tests.
SERVER_STARTUP_TIMEOUT = 3000

BASE_MODEL_NAME = "PY007/TinyLlama-1.1B-Chat-v0.3"
ADAPTER_ID_0 = "jashing/tinyllama-colorist-lora"
ADAPTER_ID_1 = "jashing/tinyllama-energy-lora"


@pytest.mark.qaic_test_config(
    model_name=BASE_MODEL_NAME,
    seq_len=32,
    ctx_len=64,
    decode_bsz=2,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_llm_lora_max_adapter_load(device_group, make_runner):
    """Test loading maximum number of LoRA adapters."""
    os.environ["VLLM_QAIC_LORA_MAX_ID_SUPPORTED"] = "64"
    qaic_max_adapters = int(os.environ["VLLM_QAIC_LORA_MAX_ID_SUPPORTED"])

    lora_modules = [
        LoRAModulePath(name=str(i), path=snapshot_download(repo_id=ADAPTER_ID_0))
        for i in range(qaic_max_adapters)
    ]

    with make_runner(
        async_scheduling=False,
        dg=device_group,
        enable_lora=True,
        max_loras=2,
        lora_modules=lora_modules,
    ):
        pass

    # test time: 27.07s


@pytest.mark.qaic_test_config(
    model_name=BASE_MODEL_NAME,
    seq_len=32,
    ctx_len=64,
    decode_bsz=2,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_llm_lora_offline_init_caching(device_group):
    """Test offline LoRA initialization and QPC caching."""
    lora_modules = [
        LoRAModulePath(name="adapter_0", path=snapshot_download(repo_id=ADAPTER_ID_0)),
        LoRAModulePath(name="adapter_1", path=snapshot_download(repo_id=ADAPTER_ID_1)),
    ]

    llm_kwargs = dict(
        model=BASE_MODEL_NAME,
        max_num_seqs=2,
        max_model_len=64,
        long_prefill_token_threshold=32,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        gpu_memory_utilization=1.0,
        enable_lora=True,
        max_loras=2,
        enable_prefix_caching=False,
        async_scheduling=False,
        additional_config={
            "device_group": device_group,
            "lora_modules": lora_modules,
        },
    )

    start = perf_counter()
    llm = LLM(**llm_kwargs)
    end = perf_counter()
    init_time_0 = end - start

    del llm
    gc.collect()

    start = perf_counter()
    llm = LLM(**llm_kwargs)
    end = perf_counter()
    init_time_1 = end - start

    del llm
    gc.collect()

    # test model compile caching
    assert init_time_0 > init_time_1


@pytest.mark.qaic_test_config(num_device_groups=1, device_group_size=1)
def test_llm_lora_online_openai_init(device_group, host, port, server_runner):
    """Test online LoRA initialization with OpenAI API server."""
    lora_module_1 = f"adapter_0={snapshot_download(repo_id=ADAPTER_ID_0)}"
    lora_module_2 = f"adapter_1={snapshot_download(repo_id=ADAPTER_ID_1)}"

    additional_config = {"device_group": device_group}

    with server_runner(
        server_runner.Backend.OPENAI_API_SERVER_MODULE,
        BASE_MODEL_NAME,
        host,
        port,
        32,
        64,
        2,
        "mxfp6",
        "mxint8",
        additional_config,
        timeout=SERVER_STARTUP_TIMEOUT,
        enable_lora=True,
        max_loras=2,
        lora_modules=[lora_module_1, lora_module_2],
    ):
        pass


@pytest.mark.qaic_test_config(
    model_name=BASE_MODEL_NAME,
    seq_len=128,
    ctx_len=256,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_llm_lora_consistency(device_group, make_runner):
    """Test output consistency across multiple LoRA inference runs."""
    lora_modules = [
        LoRAModulePath(name="adapter_0", path=snapshot_download(repo_id=ADAPTER_ID_0)),
        LoRAModulePath(name="adapter_1", path=snapshot_download(repo_id=ADAPTER_ID_1)),
    ]

    sampling_params = SamplingParams(temperature=0, max_tokens=None)

    outputDict = dict()

    prompts = [
        "Please answer the following question: Harry slept 9 hours last night. "
        "His friend James slept only 2/3 of what Harry slept. How many more "
        "hours did Harry sleep than James?\nAnswer:",
        "The following headline is the headline of a news report. Please write "
        "the content of the news passage based on only this headline.\n\n"
        "Headline: Netflix is currently down for many users around the world "
        "\n\nContent:",
        "Please answer the following question: Jean has 30 lollipops. Jean eats "
        "2 of the lollipops. With the remaining lollipops, Jean wants to "
        "package 2 lollipops in one bag. How many bags can Jean fill?\nAnswer:",
        "The following headline is the headline of a news report. Please write "
        "the content of the news passage based on only this headline.\n\n"
        "Headline: Reddit buys TikTok rival Dubsmash \n\nContent:",
        "Please answer the following question: John adopts a dog. He takes the "
        "dog to the groomer, which costs $100. The groomer offers him a 30% "
        "discount for being a new customer. How much does the grooming "
        "cost?\nAnswer:",
        "The following headline is the headline of a news report. Please write "
        "the content of the news passage based on only this headline.\n\n"
        "Headline: GPS and water don't mix. So scientists have found a new way "
        "to navigate under the sea \n\nContent:",
        "Please answer the following question: Christina is planning a "
        "birthday party and needs .75 gift bags per invited guest, because "
        "1/4 of attendees don't show up. She invited 16 friends. Gift bags "
        "are $2 each. How much will she spend?\nAnswer:",
        "The following headline is the headline of a news report. Please write "
        "the content of the news passage based on only this headline.\n\n"
        "Headline: SpaceX’s Third Starship Prototype Collapsed Last Night "
        "\n\nContent:",
    ]

    for p in prompts:
        outputDict[p] = []

    runner = make_runner(
        async_scheduling=False,
        dg=device_group,
        enable_lora=True,
        max_loras=4,
        lora_modules=lora_modules,
    )

    with runner as model:
        llm: LLM = model.llm

        lora_requests = [
            LoRARequest(
                lora_name=lora_modules[i % 2].name,
                lora_int_id=(i % 2 + 1),
                lora_path=lora_modules[i % 2].path,
            )
            for i in range(len(prompts))
        ]

        for _ in range(5):
            # Combine the lists
            combined = list(zip(prompts, lora_requests, strict=False))

            # Shuffle the combined list
            random.shuffle(combined)

            # Unzip the lists back into separate lists
            prompts, lora_requests = zip(*combined, strict=False)

            output = llm.generate(prompts, sampling_params, lora_request=lora_requests)
            for i, op in enumerate(output):
                generated_text = op.outputs[0].text
                outputDict[prompts[i]].append(str(prompts[i] + generated_text))

        for key in outputDict:
            assert len(set(outputDict[key])) == 1, (
                "Same prompts have different outputs generated! Exist "
                "generation variation"
            )
