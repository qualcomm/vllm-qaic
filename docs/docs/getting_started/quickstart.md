# Quickstart

Get vLLM running on QAIC hardware in minutes. Zero to first inference in under 5 minutes.

!!! success "What you'll have at the end"
    An OpenAI-compatible inference server running on Qualcomm Cloud AI 100,
    responding to standard `/v1/chat/completions` requests — identical to GPU-based vLLM.

## Docker (Fastest Path)

Start an OpenAI-compatible server with a pre-built container:

!!! note "First run takes several minutes"
    The first invocation will download the model from HuggingFace and compile it to a QPC binary for your hardware. This typically takes 3-10 minutes depending on model size and network speed. Subsequent runs reuse the cached artifacts and start much faster.

```bash
docker run --rm -it --network host \
  --device /dev/accel/ \
  --shm-size=2gb \
  ghcr.io/quic/cloud_ai_inference_vllm:1.21.2.0 \
  --host 127.0.0.1 \
  --port 8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-model-len 256 \
  --max-num-seq 16 \
  --max-seq-len-to-capture 128 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8
```

Test it:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain vLLM in one sentence."}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

??? example "Expected response"
    ```json
    {
      "id": "chatcmpl-abc123",
      "object": "chat.completion",
      "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "vLLM is a high-throughput inference engine for large language models that uses PagedAttention to efficiently manage memory."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 28,
        "completion_tokens": 24,
        "total_tokens": 52
      }
    }
    ```

---

## Offline Inference (Source-Based)

After [installing vllm-qaic](installation/index.md), set device visibility and run:

```bash
export QAIC_VISIBLE_DEVICES=0
```

=== "AOT Mode"

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=8,
        max_model_len=2048,
        quantization="mxfp6",
        kv_cache_dtype="mxint8",
        async_scheduling=False,
    )

    prompts = ["Hello, my name is", "The future of AI is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 50)
    ```

=== "PYT Mode"

    ```python
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_num_seqs=8,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        async_scheduling=False,
    )

    prompts = ["Hello, my name is", "The future of AI is"]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print("-" * 50)
    ```

!!! tip "Which mode should I use?"
    **AOT Mode** is recommended for production — it pre-compiles models into optimized QPCs for maximum throughput.
    **PYT Mode** is better for development and experimentation — no compilation step, immediate execution.

    See [Mode Selection](index.md) for a detailed comparison.

---

## Online Serving (Source-Based)

Start a vLLM server from your installed environment:

```bash
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-num-seqs 8 \
  --max-model-len 2048 \
  --quantization mxfp6 \
  --kv-cache-dtype mxint8
```

!!! note "Health check"
    Verify the server is ready before sending requests:
    ```bash
    curl http://localhost:8000/health
    ```

---

## Next Steps

<div class="grid cards" markdown>

-   :bar_chart: **Feature Matrix**

    See what's supported on your hardware

    [:octicons-arrow-right-24: Features](../user_guide/features/index.md)

-   :gear: **Configuration**

    Tune engine args for your workload

    [:octicons-arrow-right-24: Engine Args](../user_guide/configuration/engine_args.md)

-   :fast_forward: **Speculative Decoding**

    Accelerate decode throughput with N-gram or draft models

    [:octicons-arrow-right-24: SpD Guide](../user_guide/features/speculative_decoding.md)

-   :brain: **Supported Models**

    Full list of validated models

    [:octicons-arrow-right-24: Models](../user_guide/models/supported_models_aot.md)

</div>
