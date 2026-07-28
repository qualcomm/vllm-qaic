# Encoder-Decoder Models

QAIC supports encoder-decoder architectures, primarily Whisper for audio transcription.

## Whisper

OpenAI's Whisper model runs on QAIC for speech-to-text inference:

```python
import librosa
from vllm import LLM, SamplingParams

llm = LLM(
    model="openai/whisper-tiny.en",
    max_num_seqs=1,
    max_model_len=150,
    max_num_batched_tokens=1500,
    quantization="mxfp6",
    enable_prefix_caching=False,
    limit_mm_per_prompt={"audio": 1},
    hf_overrides={"max_source_positions": 1500},
    additional_config={"device_group": [0]},
)

# Load audio file (any sample rate — librosa resamples automatically)
audio = librosa.load("audio.wav", sr=None)

# For Whisper on QAIC, only prefill length (PL) = 1 is supported.
# Continuous batching is not supported.
prompt = {
    "prompt": "<|startoftranscript|>",
    "multi_modal_data": {"audio": audio},
}

sampling_params = SamplingParams(temperature=0, top_p=1.0, max_tokens=200)

outputs = llm.generate(prompt, sampling_params)

for output in outputs:
    print(f"Generated text: {output.outputs[0].text!r}")
```

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_model_len` | `ctx_len` | Decoder maximum output length in tokens |
| `max_num_batched_tokens` | `encoder_ctx_len` | Encoder context length (1500 for Whisper mel frames) |
| `max_num_seqs` | `1` | Batch size for concurrent transcriptions |
| `hf_overrides` | `{"max_source_positions": encoder_ctx_len}` | Override HF config to match compiled encoder length |
| `limit_mm_per_prompt` | `{"audio": 1}` | Limit to one audio input per prompt |
| `enable_prefix_caching` | `False` | Must be disabled for Whisper |

!!! warning "Constraints"
    - AOT mode only
    - Currently validated for Whisper family
    - Continuous batching not supported
    - Encoder context length determined at compilation time
