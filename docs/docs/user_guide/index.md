# User Guide

Operational documentation for deploying and configuring vLLM on Qualcomm Cloud AI hardware.

## Software Stack

The Qualcomm Cloud AI SDK consists of two layers:

**Platform SDK** — driver support for Cloud AI accelerators, APIs for executing model binaries, and tools for card health, monitoring, and telemetry. Includes a kernel driver, userspace runtime with Python/C++ bindings, and card firmware.

**Apps (Application) SDK** — converts trained models and prepares runtime binaries for Cloud AI platforms. Contains the graph compiler, performance tuning tools, quantization support, and integration with QEfficient.

<div style="text-align:center;">
<img src="../assets/Plat_Apps_SDK.png" alt="Qualcomm Cloud AI SDK Stack" style="max-width:100%;">
</div>

### vLLM Integration

`vllm-qaic` connects vLLM to Qualcomm Cloud AI accelerators through a layered software stack — enabling any application built on the OpenAI API (LangChain, CrewAI, curl) to serve LLMs on Cloud AI hardware.

<div style="text-align:center;">
<img src="../assets/vllm_architecture.svg" alt="vLLM Architecture on Qualcomm Cloud AI" style="width:60%;">
</div>

| Layer | Components | Description |
|---|---|---|
| **Applications** | curl, LangChain, OpenAI SDK | Any OpenAI-compatible client connects directly |
| **vLLM Server** | OpenAI-Compatible Endpoints | HTTP/gRPC serving with continuous batching and request scheduling |
| **QAIC Backend** | Prefill Engine, Decode Engine, KV Cache Manager, Sampler | vLLM backend dispatching to Cloud AI hardware |
| **QEfficient / Apps SDK** | ONNX Export, Quantization, Compiler (`qaic-compile`) | One-time AOT compilation of HuggingFace models to QPC binaries |
| **QPC** | Compiled model binary | Static-shape binary optimized for Cloud AI execution |
| **Cloud AI Accelerator** | Multi-Card (Tensor Slicing), Disaggregated Serving (Prefill / Decode) | Hardware execution across one or more Cloud AI cards |

Two execution paths are supported:

**AOT (Ahead-of-Time)** — models are exported via QEfficient, compiled to a QPC binary by the QAIC compiler, and loaded at serve time. No PyTorch at runtime.

**PYT (Eager)** — models execute dynamically via the `torch_qaic` package, which registers Cloud AI as a PyTorch backend and dispatches operators to hardware.

<div style="text-align:center;">
<img src="../assets/pytorch-stack.png" alt="PyTorch / vLLM Stack" style="max-width:100%;">
</div>

---

<div class="grid cards" markdown>

-   :bar_chart:{ .lg .middle } **Support Matrix**

    ---

    Feature availability across AOT and Eager execution modes.

    [:octicons-arrow-right-24: Support Matrix](features/index.md)

-   :brain:{ .lg .middle } **Supported Models**

    ---

    20+ validated models across text, vision, embedding, and audio.

    [:octicons-arrow-right-24: AOT Models](models/supported_models_aot.md) · [:octicons-arrow-right-24: Eager Models](models/supported_models_eager.md)

-   :globe_with_meridians:{ .lg .middle } **Serving**

    ---

    Online API server (OpenAI-compatible) and offline batch inference.

    [:octicons-arrow-right-24: Online Serving](serving/online_serving.md) · [:octicons-arrow-right-24: Offline](serving/offline_inference.md)

-   :zap:{ .lg .middle } **Features**

    ---

    Speculative decoding, quantization, multimodal, disaggregated serving, LoRA.

    [:octicons-arrow-right-24: Speculative Decoding](features/speculative_decoding.md)

-   :gear:{ .lg .middle } **Configuration**

    ---

    Engine arguments, environment variables, and device management.

    [:octicons-arrow-right-24: Engine Args](configuration/engine_args.md)

</div>
