<h1 align="center">
vLLM Qualcomm Cloud AI (QAIC) Plugin
</h1>

<p align="center">
| <a href="https://www.qualcomm.com/artificial-intelligence/data-center"><b>Qualcomm Data Center AI</b></a> | <a href="https://qualcomm.github.io/vllm-qaic/"><b>Documentation</b></a> | <a href="https://quic.github.io/cloud-ai-sdk-pages/"><b>User Guide</b></a> | <a href="docs/installation.md"><b>Installation Guide</b></a> |
</p>

---

**Qualcomm Cloud AI 100** is an AI inference accelerator designed to deliver exceptional performance and power efficiency for Large Language Models and other AI workloads. Built on Qualcomm's advanced HexNN architecture and Neural Signal Processors (NSPs), the Cloud AI 100 provides scalable, high-throughput inference capabilities optimized for enterprise and cloud deployments.

The vLLM QAIC plugin (`vllm-qaic`) is a dedicated unified backend extension that enables seamless integration of Qualcomm Cloud AI 100 Accelerators with vLLM using PyTorch and Qualcomm Cloud AI Compiler.

`vllm-qaic` supports two inference modes:

| Mode | Description |
|------|-------------|
| **Eager Mode** | Dynamic execution via `torch-qaic` |
| **Ahead-of-Time (AoT) Compiled Mode** | Static compilation via efficient-transformers and the Qualcomm Cloud AI Compiler |

> **Important:** Eager and AoT inference modes will coexist in the same environment in future releases, but for now, only one mode can be used at a time. Please ensure you are following the correct setup instructions for your selected mode of inference. Follow **only** the steps for your chosen mode.

---

For more information about Qualcomm Cloud AI 100, check out:

- 🚀 [Qualcomm Data Center AI Solutions](https://www.qualcomm.com/artificial-intelligence/data-center)
- 📚 [Cloud AI SDK User Guide](https://quic.github.io/cloud-ai-sdk-pages/)
- 🎯 [Cloud AI SDK API Reference](https://quic.github.io/cloud-ai-sdk-pages/latest/Python-API/index.html)
- ✅ [Validated Models and Configurations](https://quic.github.io/efficient-transformers/source/validate.html)

## Prerequisites

- **Hardware**: Qualcomm Cloud AI card(s) (Cloud AI 100, Cloud AI 080)
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Software**:
    - Python 3.12
    - [Qualcomm Cloud AI SDK](https://www.qualcomm.com/artificial-intelligence/data-center/cloud-ai-100-ultra#Software) >= 1.22.0
    - vLLM v0.15.0

## Getting Started

Please use the following recommended versions to get started quickly:

| vllm-qaic version | vLLM version | Apps SDK version | Branch | Release type | Doc |
|---|---|---|---|---|---|
| v0.15.0.dev0 | v0.15.0 | >= 1.22.0 | [main](https://github.com/qualcomm/vllm-qaic/tree/main) | Pre-release | See [QuickStart](#installation) and [Installation Guide](docs/installation.md) for more details |
| v0.23.0.dev0 | v0.23.0 | >= 1.22.0 | [v0.23.0](https://github.com/qualcomm/vllm-qaic/tree/v0.23.0) | Active Development | SpD and LoRaX not yet ported |

## Branches

**[main](https://github.com/qualcomm/vllm-qaic/tree/main)**: Primary development branch. Contributors should develop submissions based on this branch, and submit pull requests to this branch.

**[v0.23.0](https://github.com/qualcomm/vllm-qaic/tree/v0.23.0)**: Active development branch for plugin rebase to vLLM v0.23.0. Some features (SpD, LoRaX) are not yet ported. For production use, stay on `main`.

### Installation

For full installation instructions covering both AOT and PYT modes, scripted and manual steps, and wheel-based installs, see the **[Installation Guide](docs/installation.md)**.

**Quick start** — activate a Python 3.12 environment, then:

```bash
# AOT mode
./scripts/install.sh aot

# PYT mode
./scripts/install.sh pyt
```

Before installing, ensure your system has the Qualcomm Cloud AI SDK and drivers installed:

1. **[Install Cloud AI SDK and Drivers](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/index.html)**
2. **[Verify your installation](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/verification.html)**

> **PYT mode only:** install the Apps SDK with `--install-torch-qaic` to build `torch_qaic` wheels into `/opt/qti-aic/integrations/torch_qaic/`.

---

### Run an example

Set device visibility before running:

```bash
export QAIC_VISIBLE_DEVICES=0   # comma-separated device IDs, e.g. "0,1,2,3"
```

#### PYT (Eager) Mode

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    max_model_len=2048,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,
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

#### AOT Mode

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_num_seqs=8,
    max_model_len=2048,
    enable_prefix_caching=False,
    tensor_parallel_size=1,
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

## Models Supported

### Ahead-of-Time Compiled Mode

For the most up-to-date list of supported models and their validation status, please check our [model compatibility matrix](https://quic.github.io/efficient-transformers/source/validate.html).

### Eager Mode

We currently support the following models:

- Qwen/Qwen2.5-VL-3B-Instruct
- Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct
- Qwen/Qwen3-VL-32B-Instruct
- OpenGVLab/InternVL3_5-8B-Instruct
- OpenGVLab/InternVL2_5-38B

## Support

- **Documentation**: [Cloud AI SDK User Guide](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/)
- **API Reference**: [Cloud AI SDK API Documentation](https://quic.github.io/cloud-ai-sdk-pages/latest/Python-API/index.html#)

## Development

PLease refer to [CONTRIBUTING](CONTRIBUTING.md) on how to submit changes.

## Getting in Contact

Please report an issue or open a discussion as appropriate for your usecase.

- [Report an Issue on GitHub](../../issues)
- [Open a Discussion on GitHub](../../discussions)

## License

vllm-qaic is licensed under the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html). See [LICENSE.txt](LICENSE.txt) for the full license text.
