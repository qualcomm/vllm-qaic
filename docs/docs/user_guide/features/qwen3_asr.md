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

Use the installed QEfficient package in the runtime environment. Select the
precompiled QPC and an available QAIC device with:

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
