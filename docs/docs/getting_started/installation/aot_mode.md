# AOT Mode Installation

AOT (Ahead-of-Time) mode uses the QEfficient library to compile models into Qualcomm Program Containers (QPCs) for optimized static-graph inference.

!!! tip "When to use AOT mode"
    AOT mode is recommended for **production deployments** — pre-compiled models achieve maximum throughput with full feature support (SpD, LoRA, disaggregated serving, multimodal).

## Scripted Install (Recommended)

```bash
# From the vllm-qaic repo root, with Python 3.12 env activated:
./scripts/install.sh aot
```

!!! info "Configuration banner"
    Before installation starts, `install.sh` prints a full summary of every version and
    setting it will use (vllm, vllm-qaic, torch, qeff branch, target device, triton-cpu
    state). Review the banner output and override any variable before re-running.

??? example "Optional environment overrides"
    ```bash
    # Pin transformers version
    TRANSFORMERS_VERSION_AOT=4.55.3 ./scripts/install.sh aot

    # Enable triton-cpu backend (for Speculative Decoding)
    # triton-cpu is a large C++ build — requires ~10 GB of free disk space at TRITON_CPU_SRC
    TRITON_CPU=1 ./scripts/install.sh aot

    # Redirect the triton-cpu clone+build to a filesystem with more space
    # (use when $HOME has limited disk space or a user quota)
    TRITON_CPU=1 TRITON_CPU_SRC=/path/with/more/space/triton-cpu ./scripts/install.sh aot

    # Skip the disk-space pre-flight check entirely
    TRITON_CPU=1 TRITON_CPU_SKIP_DISK_CHECK=1 ./scripts/install.sh aot

    # Install from pre-built wheel
    VLLM_QAIC_INSTALL_SOURCE=wheel \
    VLLM_QAIC_SDK_PATH=/opt/qti-aic/integrations/vllm_qaic \
        ./scripts/install.sh aot
    ```

## Manual Installation

!!! warning
    Always use `python -m pip` (not `uv pip`) when installing packages with `+local`
    version labels such as `torch==2.7.0+cpu`. `uv` rejects `+local` PEP 440 labels.

```bash
# 0. Build tools
pip install "setuptools>=77.0.3,<80.0.0" setuptools-scm wheel "cmake>=3.26"

# 1. QEfficient (brings torch==2.7.0+cpu)
pip install "qefficient @ git+https://github.com/quic/efficient-transformers.git@main"

# Re-pin torch to exact AOT version
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.0+cpu"

# 2. vLLM runtime dependencies
pip install -r requirements/vllm_dependency_aot.txt

# Build vLLM from source (no C++ compilation needed)
VLLM_TARGET_DEVICE=empty pip install \
    --no-build-isolation --no-deps \
    "vllm @ git+https://github.com/vllm-project/vllm.git@v0.15.0"

# 3. vllm-qaic plugin
TORCH_QAIC_INSTALLED=0 pip install --no-build-isolation ./vllm-qaic
```

## Version Reference

| Component | Version |
|-----------|---------|
| vLLM | 0.15.0 |
| torch | 2.7.0+cpu |
| QEfficient | main branch |
| QAIC SDK | >= 1.22.0 |

!!! info "QEfficient Reference"
    QEfficient handles model export (HuggingFace → ONNX) and compilation (ONNX → QPC).
    See the [QEfficient Quick Start](https://quic.github.io/efficient-transformers/source/quick_start.html)
    for compilation parameters and workflows.
