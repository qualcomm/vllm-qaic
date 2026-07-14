# Multimodal Inference

QAIC supports vision-language models (VLMs) using a **kv_offload architecture** where the vision encoder and language decoder run as separate vLLM instances.

## Architecture

```text
Input (Image + Text)
        |
        v
+-----------------------+
|   Vision Encoder      |  <-- LLM instance (pooling mode)
|   (Device Group A)    |      Encodes image -> embeddings
+-----------------------+
        | embeddings
        v
+-----------------------+
|   Language Decoder    |  <-- LLM instance (generation mode)
|   (Device Group B)    |      Text + image embeddings -> tokens
+-----------------------+
        |
        v
    Generated Text
```

The vision encoder runs on one device group and the language decoder on another, allowing independent scaling and optimization.

## Supported Models

| Model | Architecture | Mode |
|-------|-------------|------|
| google/gemma-3-4b-it | Gemma3 | AOT |
| OpenGVLab/InternVL2_5-1B | InternVL | AOT |
| llava-hf/llava-1.5-7b-hf | LLaVA | AOT |
| Qwen/Qwen2.5-VL-7B-Instruct | Qwen2.5-VL | AOT / Eager |
| Qwen/Qwen2.5-VL-32B-Instruct | Qwen2.5-VL | AOT / Eager |
| Qwen/Qwen3-VL-32B-Instruct | Qwen3-VL | AOT / Eager |
| OpenGVLab/InternVL3_5-8B-Instruct | InternVL | Eager |

## Usage Example

```python
import copy
from vllm import LLM, EngineArgs, SamplingParams
from PIL import Image
import requests

# Configuration
seq_len = 128
ctx_len = 4096
decode_bsz = 4
vision_bsz = 4

# Engine args for language decoder
engine_args = {
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    "max_model_len": ctx_len,
    "long_prefill_token_threshold": seq_len,
    "max_num_seqs": decode_bsz,
    "quantization": "mxfp6",
    "kv_cache_dtype": "mxint8",
    "enable_mm_embeds": True,
    "additional_config": {
        "override_qaic_config": {
            "device_group": [1],       # Language on QID 1
            "height": [364, 512],      # Compiled resolutions
            "width": [532, 910],
        },
    },
    "limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0},
}

# Engine args for vision encoder
engine_args_vision = copy.deepcopy(engine_args)
engine_args_vision["runner"] = "pooling"
engine_args_vision["additional_config"]["device_group"] = [0]  # Vision on QID 0
engine_args_vision["max_num_seqs"] = vision_bsz
engine_args_vision["async_scheduling"] = False

# Create both instances
llm_vision = LLM(**engine_args_vision)
llm_lang = LLM(**engine_args)

# Load image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# Encode image
inputs = [{"prompt": "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n", "multi_modal_data": {"image": image}}]
embeddings = llm_vision.encode(inputs, pooling_task="embed")

# Generate with embeddings
for inp, emb in zip(inputs, embeddings):
    inp["multi_modal_data"]["image"] = emb.outputs.data

outputs = llm_lang.generate(inputs, SamplingParams(temperature=0.0, max_tokens=64))
```

## Dynamic Resolution (Qwen2.5-VL / Qwen3-VL)

For Qwen VL models, provide lists of heights and widths that the model is compiled for:

```python
additional_config={
    "override_qaic_config": {
        "height": [364, 512, 728],   # Supported input heights
        "width": [532, 910, 1280],   # Supported input widths
    },
}
```

If an input image doesn't match a compiled resolution exactly, it's resized to the best available match.

!!! warning "Constraints"
    - Vision encoder runs at batch_size=1 on device (vLLM batches preprocessing)
    - `async_scheduling=False` required for the vision encoder instance
