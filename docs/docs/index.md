<div style="text-align:center; margin-bottom:2rem;">
<h1 style="margin-top:1rem;">vLLM × Qualcomm Cloud AI — Inference, Reimagined.</h1>
<p style="font-size:1.1rem; color:var(--md-default-fg-color--light);">Production-Grade Inference on Qualcomm Cloud AI Accelerator.</p>
</div>

**vllm-qaic** enables LLM Serving via vLLM on Qualcomm Cloud AI accelerators. Same models, same OpenAI-compatible API, hardware-optimized execution underneath.

Install `vllm-qaic` and inference runs natively on Cloud AI accelerators. The plugin integrates through vLLM's standard platform plugin interface.

---

## Hardware Compatibility

| Hardware | NSP Cores | Features |
|----------|-----------|----------|
| Cloud AI 100 Ultra | 64 (4 QIDs × 16) | Full feature support, including multi-device parallelisms across QIDs |
| Cloud AI 80 Ultra | 32 (4 QIDs × 8) | Full feature support, including multi-device parallelisms across QIDs |
| Cloud AI 100 Standard/Pro | 16 (1 QID) | Full feature support |

**Requirements:** Qualcomm Cloud AI Apps SDK ≥ 1.22.0 · Python 3.12 · Linux (Ubuntu 22.04+, RHEL 9+)

---

## Quick Start (2 Commands)

Start an OpenAI-compatible server with a pre-built Docker container:

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

Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Explain vLLM in one sentence."}],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

---

## Two Inference Modes

`vllm-qaic` supports two execution modes:

| | AOT (Ahead-of-Time) | PYT (Eager / PyTorch) |
|---|---|---|
| **Engine** | QEfficient + QAIC Compiler | torch_qaic |
| **Best for** | Production throughput | Development flexibility |
| **Model compilation** | Pre-compiled QPCs (static graphs) | Dynamic execution |
| **Feature coverage** | Full (SpD, LoRA, disaggregated, multimodal) | VLMs (experimental) |
| **torch version** | 2.7.0+cpu | 2.10.0+cpu |

!!! warning "Mode isolation"
    The two modes cannot coexist in the same Python environment. Use separate virtual environments.

---

## Version Compatibility

| vllm-qaic | vLLM | Apps SDK | QEfficient | torch (AOT) | torch (PYT) |
|-----------|------|----------|------------|-------------|-------------|
| v0.15.0.dev0 | v0.15.0 | >= 1.22.0 | main | 2.7.0+cpu | 2.10.0+cpu |

---

## Key Capabilities

| Feature | Description |
|---------|-------------|
| **Speculative Decoding** | N-gram, suffix, and draft-model methods for decode acceleration |
| **Disaggregated Serving** | Separate prefill/decode for independent TTFT/TPOT optimization |
| **MX Quantization** | mxfp6 compute, mxint8 KV cache — hardware-native memory efficiency |
| **Multimodal** | Vision-language models via kv_offload architecture |
| **LoRA** | Adapter serving without recompilation |
| **Embedding Models** | Pooling tasks (rerank, embed, classify, score) |
| **20+ Validated Models** | Llama, Qwen, Mistral, Gemma, Phi, DeepSeek, and more |

---

<div class="grid cards" markdown>

-   :rocket:{ .lg .middle } **Quick Start**

    ---

    Get an OpenAI-compatible server running in minutes with Docker.

    [:octicons-arrow-right-24: Quickstart](getting_started/quickstart.md)

-   :bar_chart:{ .lg .middle } **Support Matrix**

    ---

    Feature availability across AOT and Eager execution modes.

    [:octicons-arrow-right-24: Support Matrix](user_guide/features/index.md)

-   :brain:{ .lg .middle } **Supported Models**

    ---

    20+ validated models across text, vision, embedding, and audio tasks.

    [:octicons-arrow-right-24: AOT Models](user_guide/models/supported_models_aot.md) · [:octicons-arrow-right-24: Eager Models](user_guide/models/supported_models_eager.md)

-   :zap:{ .lg .middle } **Features**

    ---

    SpD, quantization, disaggregated serving, multimodal, LoRA, and more.

    [:octicons-arrow-right-24: Features](user_guide/features/speculative_decoding.md)

-   :gear:{ .lg .middle } **Configuration**

    ---

    Engine args, env vars, and device management for your workload.

    [:octicons-arrow-right-24: Configuration](user_guide/configuration/engine_args.md)

-   :construction_worker:{ .lg .middle } **Developer Guide**

    ---

    Architecture, profiling, testing, and contribution guidelines.

    [:octicons-arrow-right-24: Developer Guide](developer_guide/index.md)

</div>

---

## External Resources

| Resource | Link |
|----------|------|
| QEfficient Library | [Documentation](https://quic.github.io/efficient-transformers/) |
| Cloud AI SDK | [User Guide](https://quic.github.io/cloud-ai-sdk-pages/) |
| API Reference | [Cloud AI SDK Python API](https://quic.github.io/cloud-ai-sdk-pages/latest/Python-API/index.html) |
