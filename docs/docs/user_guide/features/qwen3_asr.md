# Qwen3-ASR on Cloud AI 100

Qwen3-ASR is available through the QAIC speech-to-text path and the OpenAI-compatible audio workflow.

## Configuration

The reference QPC configuration is:

| Setting | Value |
| --- | ---: |
| Model | `Qwen/Qwen3-ASR-0.6B-hf` |
| Device group | One or more user-selected QAIC device IDs |
| Batch size | 1 |
| Prefill sequence length | 512 |
| Encoder context length | 3000 feature frames |
| Decoder context length | 512 tokens |
| Chunk size | 30 seconds |
| Chunk overlap | Application-controlled; 0 or 5 seconds does not require a new QPC |
| Generation | Greedy, up to 128 tokens |

Device selection is controlled by the normal QAIC settings. No device ID is reserved by the model integration.

When testing a local QEfficient checkout, set:

```bash
export VLLM_QAIC_EFFICIENT_TRANSFORMERS=/path/to/efficient-transformers
export QAIC_VISIBLE_DEVICES=20
```

For a precompiled QPC:

```bash
export VLLM_QAIC_QPC_PATH=/path/to/qpc
```

## One-shot example

```bash
python examples/qaic_qwen3_asr.py /path/to/audio.wav \
  --device-ids 0 \
  --qpc-path /path/to/qpc
```

The example loads the model once, sends one audio request, and prints the transcription and elapsed time.

## Persistent benchmark

The benchmark creates one persistent `LLM` instance, performs one warmup request, excludes that request from measurements, and sends each benchmark file once afterward:

```bash
python examples/qaic_qwen3_asr_benchmark.py \
  /path/to/audio_samples \
  --device 0 \
  --max-tokens 128 \
  --output qwen3_asr_vllm_benchmark.json
```

Audio files are grouped by parent directory and nominal filename duration (`40s`, `50s`, or `60s`). The JSON report contains per-file latency, transcription, warmup latency, and total inference wall time.

Speech requests use batch size 1. Application-level chunking should split long audio into 30-second windows and merge the resulting text; overlap is a preprocessing choice and does not change the compiled QPC shapes.
