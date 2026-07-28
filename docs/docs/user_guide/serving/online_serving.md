# Online Serving

Run an OpenAI-compatible API server on QAIC hardware.

## Starting the Server

### Docker (Recommended for Production)

For production deployments, use bind mounts to cache model downloads and compiled QPCs across container restarts:

```bash
docker run --rm -it --network host \
  --workdir /workspace \
  --device /dev/accel/ \
  --shm-size=4gb \
  --mount type=bind,source=$HOME/.cache,target=/cache \
  -e HF_HOME=/cache/huggingface \
  -e QEFF_HOME=/cache/qeff_models \
  -e HF_TOKEN=<your_hf_token> \
  ghcr.io/quic/cloud_ai_inference_vllm:1.21.2.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 2048 \
  --max-num-seq 8 \
  --max-seq-len-to-capture 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8
```

!!! tip "Cache reuse for faster restarts"
    The `--mount` binds and `-e HF_HOME`/`-e QEFF_HOME` environment variables persist model downloads and compiled QPCs in `$HOME/.cache/`. The first run will take 3-10 minutes (model download + QPC compilation), but subsequent runs reuse the cache and start in seconds.

For quick testing without cache persistence, use the minimal form from the [Quickstart](../../getting_started/quickstart.md).

### Source-Based

```bash
export QAIC_VISIBLE_DEVICES=0

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 8 \
  --max-model-len 2048 \
  --long-prefill-token-threshold 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8
```

## API Endpoints

The server exposes standard OpenAI-compatible endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/chat/completions` | Chat completions (recommended) |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | List available models |
| `GET /health` | Health check |

## Client Examples

### Chat Completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

### Streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Write a haiku about AI."}],
    "stream": true,
    "max_tokens": 64
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Explain quantum computing briefly."}],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## Server with Speculative Decoding

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 4 \
  --max-model-len 2048 \
  --long-prefill-token-threshold 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5}' \
  --additional-config '{"override_qaic_config":{"device_group":[0],"num_cores":16}}'
```

## Monitoring

The server exposes Prometheus metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

Key metrics for QAIC:
- `vllm:num_requests_running` — active requests
- `vllm:num_requests_waiting` — queued requests
- `vllm:generation_tokens_total` — total generated tokens
- `vllm:time_to_first_token_seconds` — TTFT distribution
- `vllm:time_per_output_token_seconds` — TPOT distribution
