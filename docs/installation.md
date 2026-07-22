# vllm-qaic Installation Guide

This guide covers installing `vllm-qaic` in both **AOT** (Ahead-of-Time compiled) and **PYT** (Eager/PyTorch) modes, either using the provided install script or manually step by step. Both source and wheel install paths are covered.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Mode Overview](#mode-overview)
- [Install using `install.sh` (recommended)](#install-using-installsh-recommended)
  - [AOT mode](#aot-mode--installsh)
  - [PYT mode](#pyt-mode--installsh)
- [Manual Installation](#manual-installation)
  - [AOT mode](#aot-mode--manual)
  - [PYT mode](#pyt-mode--manual)
- [Wheel-based Installation](#wheel-based-installation)
  - [Build wheels](#step-1--build-wheels)
  - [Install from wheel using `install.sh`](#step-2a--install-from-wheel-using-installsh)
  - [Install from wheel manually](#step-2b--install-from-wheel-manually)
- [Verification](#verification)
- [Version Reference](#version-reference)

---

## Prerequisites

| Requirement | Value |
|---|---|
| Hardware | Qualcomm Cloud AI 100 / Cloud AI 080 |
| OS | Linux (Ubuntu 22.04+) |
| Python | 3.12 |
| QAIC Platform SDK | >= 1.22.0 |
| QAIC Apps SDK | >= 1.22.0 (PYT mode requires `--install-torch-qaic` flag) |

Install the QAIC SDK before proceeding:
- [SDK installation guide](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/index.html)
- For **PYT mode**: run the Apps SDK installer with `--install-torch-qaic` to build `torch_qaic` wheels into `/opt/qti-aic/integrations/torch_qaic/py312/`

Activate a Python 3.12 environment before running any install step:

```bash
# conda
conda create -n vllm-qaic python=3.12
conda activate vllm-qaic

# or uv venv (faster installs)
uv venv .venv --python 3.12
source .venv/bin/activate

# or plain venv
python3.12 -m venv .venv
source .venv/bin/activate
```

---

## Mode Overview

| | AOT (Ahead-of-Time) | PYT (Eager / PyTorch) |
|---|---|---|
| Inference engine | QEfficient + QAIC compiler | torch_qaic |
| torch version | `2.7.0+cpu` | `2.10.0+cpu` |
| `torch_qaic` required | No (must **not** be present) | Yes |
| vllm-qaic wheel tag | `*aot*` | `*pyt*` |

> The two modes cannot coexist in the same environment. Use a separate virtual environment for each.

---

## Install using `install.sh` (recommended)

The script handles all dependency ordering, version pinning, and `uv`/`pip` detection automatically.

> **Configuration banner:** Before installation starts, `install.sh` prints a full summary of
> every version and setting it will use (vllm, vllm-qaic, torch, qeff branch, target device,
> triton-cpu state). Review it and override any variable before re-running.

### AOT mode — `install.sh`

```bash
# From the vllm-qaic repo root, with your env activated:
./scripts/install.sh aot
```

### PYT mode — `install.sh`

```bash
./scripts/install.sh pyt
```

**Optional environment overrides:**

```bash
# Pin transformers to a specific version after QEfficient/torch_qaic install
TRANSFORMERS_VERSION_AOT=4.55.3 ./scripts/install.sh aot
TRANSFORMERS_VERSION_PYT=4.57.3 ./scripts/install.sh pyt

# Enable triton-cpu backend for AOT Speculative Decoding
# triton-cpu is a large C++ build — requires ~10 GB of free disk space at TRITON_CPU_SRC
TRITON_CPU=1 ./scripts/install.sh aot

# Redirect the triton-cpu clone+build to a filesystem with more space
# (use when $HOME has limited disk space or a user quota)
TRITON_CPU=1 TRITON_CPU_SRC=/path/with/more/space/triton-cpu ./scripts/install.sh aot

# Skip the disk-space pre-flight check entirely
TRITON_CPU=1 TRITON_CPU_SKIP_DISK_CHECK=1 ./scripts/install.sh aot

# Force wheel install from a custom SDK path
VLLM_QAIC_INSTALL_SOURCE=wheel VLLM_QAIC_SDK_PATH=/path/to/sdk ./scripts/install.sh pyt
```

---

## Manual Installation

Use these steps if you prefer not to run `install.sh`.

> **Note:** Always use `python -m pip` (not `uv pip`) when installing packages with `+local` version labels such as `torch==2.7.0+cpu`. `uv` rejects `+local` PEP 440 labels from remote indexes.

### AOT mode — manual

```bash
# 0. Build tools
pip install "setuptools>=77.0.3,<80.0.0" setuptools-scm wheel "cmake>=3.26"

# [Optional] Remove torch_qaic if present — it must not coexist with AOT mode
pip uninstall -y torch-qaic

# 1. QEfficient — brings torch==2.7.0+cpu as a dependency
pip install "qefficient @ git+https://github.com/quic/efficient-transformers.git@main"

# 1 (cont.) Re-pin torch to exact AOT version
#   (required for uv venvs: uv skips +local labels from remote indexes)
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.7.0+cpu"

# [Optional] Pin transformers if QEfficient's version conflicts with your model
# pip install "transformers==<version>"

# 2. vllm runtime dependencies (torch excluded — already installed above)
pip install -r requirements/vllm_dependency_aot.txt

# 2 (cont.) vllm from public GitHub tag — built with VLLM_TARGET_DEVICE=empty
#   (empty target: no C++ compilation, no torch in wheel METADATA)
VLLM_TARGET_DEVICE=empty pip install \
    --no-build-isolation --no-deps \
    "vllm @ git+https://github.com/vllm-project/vllm.git@v0.15.0"

# 3. vllm-qaic from source
TORCH_QAIC_INSTALLED=0 pip install --no-build-isolation ./vllm-qaic
```

### PYT mode — manual

```bash
# 0. Build tools
pip install "setuptools>=77.0.3,<80.0.0" setuptools-scm wheel "cmake>=3.26"

# 1. CPU torch FIRST — torch_qaic validates at import that torch is CPU-only
#    and will error if a CUDA torch is present when it imports.
python -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.10.0+cpu" \
    "torchvision==0.25.0+cpu" \
    "torchaudio==2.10.0+cpu"

# 1b. torch_qaic AFTER torch is confirmed CPU-only
#     Replace py312 with your Python version (py310, py311, py312)
pip install /opt/qti-aic/integrations/torch_qaic/py312/torch_qaic-*.whl

# [Optional] Pin transformers if torch_qaic's version conflicts with your model
# pip install "transformers==<version>"

# 2. vllm runtime dependencies (torch excluded — already installed above)
pip install -r requirements/vllm_dependency_pyt.txt

# 2 (cont.) vllm — choose based on VLLM_TARGET_DEVICE_PYT:

# Option A (default): install vllm-cpu from PyPI — no C++ compilation needed
pip install --no-deps "vllm-cpu==0.15.0"

# Option B: build from public GitHub tag with VLLM_TARGET_DEVICE=empty
#   (no C++ compilation, no torch in wheel METADATA — uv-safe)
# VLLM_TARGET_DEVICE=empty pip install \
#     --no-build-isolation --no-deps \
#     "vllm @ git+https://github.com/vllm-project/vllm.git@v0.15.0"

# 3. vllm-qaic from source
pip install --no-build-isolation ./vllm-qaic
```

---

## Wheel-based Installation

Use this path when distributing a pre-built `vllm-qaic` wheel (e.g., from the QAIC SDK or a CI artifact) instead of building from source.

### Step 1 — Build wheels

Wheels are always built inside a docker container — run `build_wheels.sh` from the repo root with docker available (no venv activation needed; the script builds/reuses the vllm-qaic base image and does the build inside an ephemeral container):

```bash
# Build both AOT and PYT wheels into ./dist/
./scripts/build_wheels.sh both --outdir ./dist

# Or build individually
./scripts/build_wheels.sh aot --outdir ./dist
./scripts/build_wheels.sh pyt --outdir ./dist
```

Output locations:

| Mode | Wheel path |
|---|---|
| AOT | `dist/aot/vllm_qaic-*aot*-py3-none-any.whl` |
| PYT | `dist/pyt/py312/vllm_qaic-*pyt*-cp312-cp312-linux_x86_64.whl` |

### Step 2a — Install from wheel using `install.sh`

Point `VLLM_QAIC_SDK_PATH` to the directory **above** the `py312/` subdirectory for PYT, or directly to the AOT wheel directory:

```bash
# AOT wheel install
VLLM_QAIC_INSTALL_SOURCE=wheel \
VLLM_QAIC_SDK_PATH=/path/to/dist/aot \
    ./scripts/install.sh aot

# PYT wheel install
VLLM_QAIC_INSTALL_SOURCE=wheel \
VLLM_QAIC_SDK_PATH=/path/to/dist/pyt \
    ./scripts/install.sh pyt
```

The script installs all upstream dependencies (QEfficient/torch_qaic, vllm deps, vllm) exactly as in the source install, then drops in the pre-built `vllm-qaic` wheel for Step 3.

### Step 2b — Install from wheel manually

After completing the manual steps above through Step 2 (vllm installed), replace Step 3 with:

```bash
# AOT wheel
pip install /path/to/dist/aot/vllm_qaic-*aot*.whl

# PYT wheel (--no-deps because vllm-qaic runtime deps are already installed)
pip install --no-deps /path/to/dist/pyt/py312/vllm_qaic-*pyt*.whl
```

---

## Verification

After installation, verify the environment is clean:

```bash
# Check torch is CPU-only (no CUDA suffix)
python -c "import torch; print(torch.__version__)"
# AOT expected: 2.7.0+cpu
# PYT expected: 2.10.0+cpu

# Verify vllm-qaic loads and registers the QAIC platform plugin
python -c "import vllm_qaic; print('vllm_qaic OK')"

# PYT only: verify torch_qaic
python -c "import torch_qaic; print('torch_qaic OK')"

# Confirm no nvidia/CUDA packages are installed
pip list | grep -i "nvidia\|cuda-toolkit\|cuda-bin"
# (should return no output)
```

---

## Version Reference

All version constants are defined in [`scripts/utility.sh`](../scripts/utility.sh). Update that file when bumping versions.

| Constant | Value | Description |
|---|---|---|
| `VLLM_VERSION` | `0.15.0` | vLLM release tag |
| `TORCH_VERSION_AOT` | `2.7.0+cpu` | CPU torch for AOT (matches QEfficient exact pin) |
| `TORCH_VERSION_PYT` | `2.10.0+cpu` | CPU torch for PYT |
| `TORCHVISION_VERSION_PYT` | `0.25.0+cpu` | torchvision for PYT (keep in sync with torch) |
| `TORCHAUDIO_VERSION_PYT` | `2.10.0+cpu` | torchaudio for PYT (keep in sync with torch) |
| `QEFF_BRANCH` | `main` | QEfficient branch/tag |
| `TORCH_QAIC_VERSION` | `0.1.0` | torch_qaic wheel version |
| `VLLM_TARGET_DEVICE_AOT` | `empty` | vLLM build target for AOT mode |
| `VLLM_TARGET_DEVICE_PYT` | `cpu` | vLLM build target for PYT mode |
| `TRITON_CPU` | `0` | Set to `1` to enable triton-cpu backend (AOT SpD) |
| `TRITON_CPU_COMMIT` | `e60f448f...` | Pinned triton-cpu commit hash |
| `TRITON_CPU_SRC` | `$HOME/triton-cpu` | Clone destination for triton-cpu source |
| `TRITON_CPU_COMPILE_MAX_JOBS` | `4` | Parallel build jobs for triton-cpu compilation |
| `TRITON_CPU_SKIP_DISK_CHECK` | `0` | Set to `1` to skip the 10 GB disk-space pre-flight check |
| `TORCH_QAIC_BASE_PATH` | `/opt/qti-aic/integrations/torch_qaic` | SDK path for torch_qaic wheels |
| `VLLM_QAIC_SDK_PATH` | `/opt/qti-aic/integrations/vllm_qaic` | SDK path for pre-built vllm-qaic wheels |
