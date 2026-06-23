# LoRA Adapters

QAIC supports serving LoRA (Low-Rank Adaptation) adapters on top of base models without recompilation.

## Overview

LoRA enables efficient fine-tuning by adding small adapter layers to a pre-trained base model. On QAIC, LoRA adapters are loaded at serving time and hot-swapped per request — multiple adapters can be served within the same batch.

## Usage

```python
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.entrypoints.openai.models.protocol import LoRAModulePath

# Sampling parameters
sampling_params = SamplingParams(temperature=0, max_tokens=None)

# LoRA adapters to serve
repo_ids = [
    "predibase/gsm8k",
    "predibase/tldr_content_gen",
    "predibase/agnews_explained",
    "predibase/e2e_nlg",
    "predibase/viggo",
    "predibase/hellaswag_processed",
    "predibase/bc5cdr",
    "predibase/conllpp",
    "predibase/tldr_headline_gen",
]

# Create LLM with LoRA enabled
llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    max_num_seqs=4,
    max_model_len=1024,
    long_prefill_token_threshold=128,
    quantization="mxfp6",
    kv_cache_dtype="mxint8",
    enable_lora=True,
    max_loras=9,
    additional_config={
        "device_group": [0],
        "lora_modules": [
            LoRAModulePath(
                name=repo_id.split("/")[1],
                path=snapshot_download(repo_id=repo_id),
            )
            for repo_id in repo_ids
        ],
    },
)

# Build LoRA requests — one per prompt
lora_requests = [
    LoRARequest(
        lora_name=repo_ids[i].split("/")[1],
        lora_int_id=(i + 1),
        lora_path=snapshot_download(repo_id=repo_ids[i]),
    )
    for i in range(len(prompts))
]

# Generate
outputs = llm.generate(prompts, sampling_params, lora_request=lora_requests)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    num_tokens = len(output.outputs[0].token_ids)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}, Tokens: {num_tokens}")
```

## Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enable_lora` | Enable LoRA adapter support | `False` |
| `max_loras` | Maximum number of LoRA adapters active in a batch | `1` |
| `additional_config.lora_modules` | List of `LoRAModulePath` entries pre-registering adapters by name and path | `[]` |
| `additional_config.device_group` | List of device indices to use | `[0]` |

!!! warning "Constraints"
    - AOT mode only (not supported in PYT/Eager mode)
    - LoRA adapters must be compatible with the compiled base model architecture
