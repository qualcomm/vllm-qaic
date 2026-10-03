# Encoder-Decoder Models

QAIC supports encoder-decoder architectures for audio transcription.

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

## Cohere ASR

Cohere Transcribe runs through vLLM's OpenAI-compatible transcription endpoint.
This initial integration supports one transcription at a time (batch size 1).
Use `CohereLabs/cohere-transcribe-03-2026` for the multilingual model or
`CohereLabs/cohere-transcribe-arabic-07-2026` for the separately trained Arabic
model. A precompiled QPC must come from the same checkpoint that is served.

### Use an existing QPC

```bash
export QAIC_VISIBLE_DEVICES=0
export VLLM_QAIC_QPC_PATH=/path/to/cohere-asr/qpc

vllm serve CohereLabs/cohere-transcribe-03-2026 \
  --hf-overrides '{"max_source_positions":438}' \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 3504 \
  --long-prefill-token-threshold 512 \
  --limit-mm-per-prompt '{"audio":1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --additional-config '{"device_group":[0],"override_qaic_config":{"num_cores":8,"task":"transcription"}}'
```

### Build a QPC when the server starts

Automatic compilation requires a QEfficient installation with Cohere ASR
support. Until that support is released, install it from
[QEfficient PR #1275](https://github.com/quic/efficient-transformers/pull/1275).

```bash
export QAIC_VISIBLE_DEVICES=0
unset VLLM_QAIC_QPC_PATH

vllm serve CohereLabs/cohere-transcribe-03-2026 \
  --hf-overrides '{"max_source_positions":438}' \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 3504 \
  --long-prefill-token-threshold 512 \
  --limit-mm-per-prompt '{"audio":1}' \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --additional-config '{"device_group":[0],"override_qaic_config":{"num_cores":8,"task":"transcription"}}'
```

The server exports and compiles the served checkpoint before accepting requests.

Transcribe an audio file with the included client:

```bash
python examples/qaic_cohere_asr.py \
  --file-path /path/to/audio.wav \
  --language en
```

For the validated Arabic checkpoint, serve
`CohereLabs/cohere-transcribe-arabic-07-2026` and pass that same model name with
`--model` to the client together with `--language ar`. The endpoint selects the
language-specific decoder prefix; callers should not construct model prompts.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_num_seqs` | `1` | Validated static QPC batch size |
| `max_model_len` | `512` | Decoder context length compiled into the QPC |
| `max_num_batched_tokens` | `3504` | Encoder feature-frame binding compiled into the QPC |
| `hf_overrides` | `{"max_source_positions": 438}` | Cohere encoder position setting used for the QPC |
| `limit_mm_per_prompt` | `{"audio": 1}` | One audio input per transcription request |
| `VLLM_QAIC_QPC_PATH` | QPC directory | Reuses the matching precompiled QPC |
| `num_cores` | `8` | Validated QPC core count |

!!! warning "Constraints"
    - AOT mode only
    - Batch size 1 only
    - Continuous batching is not supported
    - The QPC dimensions above must match the supplied QPC
    - The served checkpoint and QPC checkpoint must match; do not compare WER across different checkpoints
