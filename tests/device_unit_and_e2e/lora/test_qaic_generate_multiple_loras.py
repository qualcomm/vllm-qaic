# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import pytest
from huggingface_hub import snapshot_download
from vllm.entrypoints.openai.models.protocol import LoRAModulePath

from vllm.lora.request import LoRARequest

MODEL_NAME = "PY007/TinyLlama-1.1B-Chat-v0.3"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

LORA_NAME = "jashing/tinyllama-colorist-lora"


@pytest.mark.qaic_test_config(
    model_name=MODEL_NAME,
    seq_len=32,
    ctx_len=64,
    decode_bsz=4,
    dtype="mxfp6",
    kv_dtype="mxint8",
    num_device_groups=1,
    device_group_size=1,
)
def test_multiple_lora_requests(device_group, make_runner):
    lora_modules = [
        LoRAModulePath(
            name=LORA_NAME + str(idx),
            path=snapshot_download(repo_id=LORA_NAME),
        )
        for idx in range(len(PROMPTS))
    ]

    with make_runner(
        async_scheduling=False,
        dg=device_group,
        enable_lora=True,
        max_loras=4,
        lora_modules=lora_modules,
    ) as model:
        llm = model.llm

        lora_request = [
            LoRARequest(
                lora_name=(LORA_NAME + str(idx)),
                lora_int_id=(idx + 1),
                lora_path=snapshot_download(repo_id=LORA_NAME),
            )
            for idx in range(len(PROMPTS))
        ]
        # Multiple SamplingParams should be matched with each prompt
        outputs = llm.generate(PROMPTS, lora_request=lora_request)
        assert len(PROMPTS) == len(outputs)

        # Exception raised, if the size of params does not match the size of prompts
        with pytest.raises(ValueError):
            outputs = llm.generate(PROMPTS, lora_request=lora_request[:1])

        # Single LoRARequest should be applied to every prompt
        single_lora_request = lora_request[0]
        outputs = llm.generate(PROMPTS, lora_request=single_lora_request)
        assert len(PROMPTS) == len(outputs)
