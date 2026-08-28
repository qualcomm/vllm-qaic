# PYT Mode Installation

PYT (Eager/PyTorch) mode uses `torch_qaic` for dynamic model execution without ahead-of-time compilation.

!!! tip "When to use PYT mode"
    PYT mode is ideal for **development and experimentation** — no compilation step means instant model loading. Currently best for vision-language models (VLMs).

## Scripted Install (Recommended)

```bash
# From the vllm-qaic repo root, with Python 3.12 env activated:
./scripts/install.sh pyt
```

??? example "Optional environment overrides"
    ```bash
    # Pin transformers version
    TRANSFORMERS_VERSION_PYT=4.57.3 ./scripts/install.sh pyt

    # Install from pre-built wheel
    VLLM_QAIC_INSTALL_SOURCE=wheel \
    VLLM_QAIC_SDK_PATH=/opt/qti-aic/integrations/vllm_qaic \
        ./scripts/install.sh pyt
    ```

## Manual Installation

```bash
# 0. Build tools
pip install "setuptools>=77.0.3,<80.0.0" setuptools-scm wheel "cmake>=3.26"

# 1. CPU torch FIRST (torch_qaic requires CPU-only torch)
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.10.0+cpu" \
    "torchvision==0.25.0+cpu" \
    "torchaudio==2.10.0+cpu"

# 2. torch_qaic (from SDK installer output)
pip install /opt/qti-aic/integrations/torch_qaic/py312/torch_qaic-*.whl

# 3. vLLM runtime dependencies
pip install -r requirements/vllm_dependency_pyt.txt

# Install vllm-cpu from PyPI (no compilation needed)
pip install --no-deps "vllm-cpu==0.28.0"

# 4. vllm-qaic plugin
pip install --no-build-isolation ./vllm-qaic
```

!!! note "Apps SDK requirement"
    PYT mode requires the Apps SDK to have been installed with `--install-torch-qaic`.
    This builds `torch_qaic` wheels into `/opt/qti-aic/integrations/torch_qaic/`.

## Version Reference

| Component | Version |
|-----------|---------|
| vLLM | 0.28.0 (vllm-cpu) |
| torch | 2.10.0+cpu |
| torchvision | 0.25.0+cpu |
| torch_qaic | 0.1.0 |
| QAIC SDK | >= 1.22.0 |
