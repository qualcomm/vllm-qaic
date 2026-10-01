<h1 align="center">
vLLM Qualcomm Cloud AI (QAIC) Plugin
</h1>

<p align="center">
| <a href="https://www.qualcomm.com/artificial-intelligence/data-center"><b>Qualcomm Data Center AI</b></a> | <a href="https://qualcomm.github.io/vllm-qaic/"><b>Documentation</b></a> | <a href="https://quic.github.io/cloud-ai-sdk-pages/"><b>User Guide</b></a> | <a href="docs/installation.md"><b>Installation Guide</b></a> |
</p>

> **This branch is under active development — plugin rebase to vLLM v0.23.0.**
> This is not a stable release.
> For production use, please switch to the `main` branch or `release/v0.15.0`.

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
- ✅ [Validated Models and Configurations](https://qualcomm.github.io/vllm-qaic/user_guide/models/supported_models_aot/)

## Prerequisites

- **Hardware**: Qualcomm Cloud AI card(s) (Cloud AI 100, Cloud AI 080)
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Software**:
    - Python 3.12
    - [Qualcomm Cloud AI SDK](https://www.qualcomm.com/artificial-intelligence/data-center/cloud-ai-100-ultra#Software) >= 1.22.0
    - vLLM v0.23.0

### Validated Qwen3-ASR Environment

The Qwen3-ASR QAIC validation used these exact versions:

| Component | Version |
|---|---|
| QAIC Platform SDK | `AIC.1.22.0.99` |
| QAIC Apps SDK | `AIC.1.22.0.99` |
| Transformers | `5.14.1` |
| QEfficient | `1.23.0.dev0` |
| PyTorch | `2.7.0+cpu` |
| vLLM | `0.23.0` |

### Qwen3-ASR Pull Requests

The Qwen3-ASR integration is tested with these pull requests:

| Component | Pull request |
|---|---|
| efficient-transformers / QEfficient | https://github.com/quic/efficient-transformers/pull/1276 |
| vLLM-QAIC | https://github.com/qualcomm/vllm-qaic/pull/146 |

The AOT runtime dependency file is:

```text
requirements-qwen3-asr-aot.txt
```

## Getting Started

Please use the following recommended versions to get started quickly:

| vllm-qaic version | vLLM version | Apps SDK version | Branch | Release type | Doc |
|---|---|---|---|---|---|
| v0.15.0.dev0 | v0.15.0 | >= 1.22.0 | [main](https://github.com/qualcomm/vllm-qaic/tree/main)  [v0.15.0](https://github.com/qualcomm/vllm-qaic/tree/release/v0.15.0)| Pre-release | See [QuickStart](#installation) and [Installation Guide](docs/installation.md) for more details |
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

### Qwen3-ASR

Qwen3-ASR inference is supported in AOT mode with a precompiled QAIC QPC. See the [Qwen3-ASR guide](docs/docs/user_guide/features/qwen3_asr.md) for the QPC shape requirements, one-shot example, and persistent benchmark.

```bash
python examples/qaic_qwen3_asr.py /path/to/audio.wav \
  --device-ids 0 \
  --qpc-path /path/to/qpc
```

The example keeps model loading outside the timed request and accepts the QPC, device group, compiler checkout, prefill length, encoder context, and generation limit through command-line options or environment variables.

### Reproducible Qwen3-ASR Setup

The following commands fetch the exact PRs used for Qwen3-ASR. Run them on a
Linux host with Python 3.12, a compatible Qualcomm Cloud AI SDK, and an
available AI100 device.

```bash
mkdir qwen3-asr-qaic
cd qwen3-asr-qaic

git clone https://github.com/quic/efficient-transformers.git efficient-transformers
cd efficient-transformers
git fetch origin pull/1276/head:qeff-pr-1276
git checkout qeff-pr-1276
cd ..

git clone https://github.com/qualcomm/vllm-qaic.git vllm-qaic
cd vllm-qaic
git fetch origin pull/146/head:vllm-qaic-pr-146
git checkout vllm-qaic-pr-146
cd ..

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install \
  -r vllm-qaic/requirements-qwen3-asr-aot.txt

python -m pip install \
  --editable ./efficient-transformers \
  --no-deps

cd vllm-qaic
./scripts/install.sh aot
python -m pip install \
  --editable . \
  --no-build-isolation
cd ..

# The installer may install upstream QEfficient; restore the local PR.
python -m pip install \
  --editable ./efficient-transformers \
  --no-deps
```

Verify that the local PRs are imported:

```bash
python - <<'PY'
import QEfficient
import transformers
import vllm

print("QEfficient:", QEfficient.__file__)
print("Transformers:", transformers.__version__)
print("vLLM:", vllm.__file__)
PY
```

`QEfficient.__file__` must point to the local `efficient-transformers`
checkout. The vLLM import must point to the local `vllm-qaic` checkout.

Set the runtime environment:

```bash
export QAIC_VISIBLE_DEVICES=<available-ai100-device-id>
export VLLM_QAIC_EFFICIENT_TRANSFORMERS=$PWD/efficient-transformers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Compile a 30-second, zero-overlap QPC using the Qwen3-ASR compile helper
provided with the project:

```bash
python tools/compile_qwen3_asr.py \
  --model Qwen/Qwen3-ASR-0.6B-hf \
  --chunk-seconds 30 \
  --context-length 512 \
  --batch-size 1 \
  --num-cores 8 \
  --device-id "$QAIC_VISIBLE_DEVICES" \
  --output-dir "$PWD/qpc"
```

The resulting QPC directory must contain `programqpc.bin`. Set its path:

```bash
export VLLM_QAIC_QPC_PATH="$(dirname "$(find "$PWD/qpc" -name programqpc.bin -print -quit)")"
test -f "$VLLM_QAIC_QPC_PATH/programqpc.bin"
```

Run a direct test:

```bash
python vllm-qaic/examples/qaic_qwen3_asr.py \
  <audio-file> \
  --device-ids "$QAIC_VISIBLE_DEVICES" \
  --qpc-path "$VLLM_QAIC_QPC_PATH" \
  --efficient-transformers "$VLLM_QAIC_EFFICIENT_TRANSFORMERS" \
  --prefill-seq-len 512 \
  --encoder-ctx-len 3000 \
  --max-model-len 512 \
  --max-tokens 128
```

Start the OpenAI-compatible server:

```bash
vllm serve Qwen/Qwen3-ASR-0.6B-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --task transcription \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --limit-mm-per-prompt '{"audio": 1}' \
  --additional-config \
  "{\"device_group\":[${QAIC_VISIBLE_DEVICES}],\"override_qaic_config\":{\"prefill_seq_len\":512,\"encoder_ctx_len\":3000}}"
```

Send an audio request:

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@<audio-file>" \
  -F "model=Qwen/Qwen3-ASR-0.6B-hf"
```

Keep the server alive while measuring latency. The first startup includes
model and QPC loading. For long audio, split the input in the application
into 30-second chunks and merge the returned text. A 0- or 5-second overlap
is an application setting and does not require a new QPC.

## Qwen3-ASR Server

The following configuration matches the Qwen3-ASR QPC used for QAIC validation:

- Model: `Qwen/Qwen3-ASR-0.6B-hf`
- AOT mode with a precompiled QPC
- One QAIC device, selected by its QID
- `prefill_seq_len=512`
- `encoder_ctx_len=3000`
- `max_model_len=512`
- Greedy transcription with up to 128 generated tokens

Set the QPC and visible device before starting the server:

```bash
export QAIC_VISIBLE_DEVICES=0
export VLLM_QAIC_QPC_PATH=/path/to/qpc

# Optional when testing an unreleased local QEfficient checkout:
# export VLLM_QAIC_EFFICIENT_TRANSFORMERS=/path/to/efficient-transformers
```

Start the OpenAI-compatible vLLM server:

```bash
vllm serve Qwen/Qwen3-ASR-0.6B-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --task transcription \
  --max-num-seqs 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --limit-mm-per-prompt '{"audio": 1}' \
  --additional-config '{"device_group":[0],"override_qaic_config":{"prefill_seq_len":512,"encoder_ctx_len":3000}}'
```

Submit an audio file after the server is ready:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F file=@/path/to/audio.wav \
  -F model=Qwen/Qwen3-ASR-0.6B-hf
```

The first server start includes model and QPC loading. Keep the server alive when measuring request latency so initialization is not included in every request. See the [Qwen3-ASR guide](docs/docs/user_guide/features/qwen3_asr.md) for the direct one-file test and persistent benchmark.

## Models Supported

### Ahead-of-Time Compiled Mode

For the most up-to-date list of supported models and their validation status, please check [Supported Models through AoT](https://qualcomm.github.io/vllm-qaic/user_guide/models/supported_models_aot/).

### PyTorch Eager Mode

For the most up-to-date list of supported models and their validation status, please check [Supported Models through PyTorch Eager Mode](https://qualcomm.github.io/vllm-qaic/user_guide/models/supported_models_eager/).

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
