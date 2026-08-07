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
- [Docker-based Installation](#docker-based-installation)
    - [Build targets](#build-targets)
    - [Build commands](#build-commands)
    - [Entrypoint script (dev only)](#entrypoint-script-dev-only)
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
| torch version | `2.7.0+cpu` | `2.11.0+cpu` |
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
    "torch==2.11.0+cpu" \
    "torchvision==0.26.0+cpu" \
    "torchaudio==2.11.0+cpu"

# 1b. torch_qaic AFTER torch is confirmed CPU-only
#     Replace py312 with your Python version (py310, py311, py312)
pip install /opt/qti-aic/integrations/torch_qaic/py312/torch_qaic-*.whl

# [Optional] Pin transformers if torch_qaic's version conflicts with your model
# pip install "transformers==<version>"

# 2. vllm runtime dependencies (torch excluded — already installed above)
pip install -r requirements/vllm_dependency_pyt.txt

# 2 (cont.) vllm — build from public GitHub tag with VLLM_TARGET_DEVICE=empty
#   (no C++ compilation, no torch in wheel METADATA — uv-safe)
VLLM_TARGET_DEVICE=empty pip install \
    --no-build-isolation --no-deps \
    "vllm @ git+https://github.com/vllm-project/vllm.git@v0.23.0"

# 3. vllm-qaic from source
pip install --no-build-isolation ./vllm-qaic
```

---

## Docker-based Installation

`docker/Dockerfile.aot` and `docker/Dockerfile.pyt` each build vllm-qaic inside a container, layered on a shared `aot-base`/`pyt-base` stage (system packages, uv venv, torch, vllm deps + vllm). Both Dockerfiles expose the same four BuildKit targets on top of that base.

### Build targets

| Target | vllm-qaic source | Install mode | Use case |
|---|---|---|---|
| `release` | Published git ref (`VLLM_QAIC_GIT_REF` tag/branch) | Non-editable | Self-contained image, reproducible from ARG pins alone — no build context needed for vllm-qaic itself |
| `ci` | Build-context checkout (default), or `VLLM_QAIC_PR`/`VLLM_QAIC_BRANCH` | Non-editable | Test a specific PR/branch/checkout without touching a dev environment |
| `dev` | Build-context checkout (always) | Editable (`pip install -e`) | Interactive iteration — source is bind-mounted at `/src/vllm-qaic` and kept in the image |
| `wheel` | Build-context checkout | N/A — builds and exports a wheel, no install | Produce a distributable `vllm_qaic-*.whl` via BuildKit's local exporter (see [Wheel-based Installation](#wheel-based-installation)) |

`release`/`ci` both install non-editably and differ only in *where* vllm-qaic's source comes from. `dev` is the only target with an editable install and the only one that ships source directories (`/src/vllm-qaic`, plus `/src/qefficient` for AOT) and a `sudo`+`entrypoint.sh` layer for interactive use — `release`/`ci` are meant to run as immutable images under whatever user the orchestrator picks, not as UID-mapped interactive containers.

`PYTHON_VERSION` (default `3.12`; also `3.10`/`3.11`) is threaded through every target via `aot-base`/`pyt-base`, provisioned with `uv python install` rather than apt so non-default versions work regardless of the base image's own Python.

### Build commands

```bash
# Release (default pins)
docker build --target release -f docker/Dockerfile.aot -t vllm-qaic-aot:1.22 .
docker build --target release -f docker/Dockerfile.pyt -t vllm-qaic-pyt:1.22 .

# Release (override version / device arch)
docker build --target release -f docker/Dockerfile.aot \
  --build-arg VLLM_QAIC_GIT_REF=v1.23 --build-arg QEFF_BRANCH=release/v1.23.0 \
  -t vllm-qaic-aot:1.23 .
docker build --target release -f docker/Dockerfile.pyt \
  --build-arg QAIC_DEVICE_ARCH=v81 -t vllm-qaic-pyt:1.22-v81 .

# CI (current checkout)
docker build --target ci -f docker/Dockerfile.aot -t vllm-qaic-aot:ci .
docker build --target ci -f docker/Dockerfile.pyt -t vllm-qaic-pyt:ci .

# CI (specific PR or branch)
docker build --target ci -f docker/Dockerfile.aot --build-arg VLLM_QAIC_PR=42 -t vllm-qaic-aot:ci-pr-42 .
docker build --target ci -f docker/Dockerfile.pyt --build-arg VLLM_QAIC_BRANCH=feature/my-branch -t vllm-qaic-pyt:ci-branch .

# Dev (editable install; AOT also supports overriding the QEfficient ref)
docker build --target dev -f docker/Dockerfile.aot -t vllm-qaic-aot:dev .
docker build --target dev -f docker/Dockerfile.aot --build-arg QEFF_PR=456 -t vllm-qaic-aot:dev .
docker build --target dev -f docker/Dockerfile.pyt -t vllm-qaic-pyt:dev .
```

The `BASE_IMAGE` used by any target must have the QAIC Platform and Apps SDKs installed (`/opt/qti-aic/` present) so vllm-qaic can load at runtime; override with `--build-arg BASE_IMAGE=...`.

### Entrypoint script (dev only)

`dev` images use `docker/entrypoint.sh` to map the container process to the host's UID/GID on `docker run`, so bind-mounted source stays host-owned and the QAIC device is accessible without running as root:

```bash
docker run -it --rm \
  -e USER_UID=$(id -u) -e USER_GID=$(id -g) \
  -e QAIC_GID=$(getent group qaic | cut -d: -f3) \
  --device /dev/accel/ -v $(pwd):/src/vllm-qaic \
  vllm-qaic-pyt:dev
```

- If `USER_UID`/`USER_GID` are unset, the entrypoint is a no-op passthrough (`exec "$@"`) and the command runs as root.
- Otherwise it creates/reuses a matching-GID/UID user, grants it passwordless sudo, joins it to a group literally named `qaic` at `QAIC_GID` (created if missing) so it can access `/dev/accel`, `chown`s its home directory, then execs the container command as that user via `runuser`.
- `--device /dev/accel/` and `QAIC_GID` are independent requirements — passing one without the other leaves the container either without device nodes at all, or with device nodes it lacks permission to open. Both are needed for `torch_qaic`/QEfficient to enumerate live devices.
- To add other setup logic (extra packages, env vars) that should run once per container start rather than at build time, extend `docker/entrypoint.sh` before its final `exec runuser -u "${USERNAME}" -- "$@"` line — anything after that line only runs as root and is skipped once the exec replaces the shell.

---

## Wheel-based Installation

Use this path when distributing a pre-built `vllm-qaic` wheel (e.g., from the QAIC SDK or a CI artifact) instead of building from source.

### Step 1 — Build wheels

Wheels are always built inside a docker container — run `build_wheels.sh` from the repo root with `docker buildx` available (no venv activation needed; the script invokes each Dockerfile's `wheel` BuildKit target and exports the result via BuildKit's local exporter, no bind mount or container run required):

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
# PYT expected: 2.11.0+cpu

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
| `TORCH_VERSION_PYT` | `2.11.0+cpu` | CPU torch for PYT |
| `TORCHVISION_VERSION_PYT` | `0.26.0+cpu` | torchvision for PYT (keep in sync with torch) |
| `TORCHAUDIO_VERSION_PYT` | `2.11.0+cpu` | torchaudio for PYT (keep in sync with torch) |
| `QEFF_BRANCH` | `main` | QEfficient branch/tag |
| `TORCH_QAIC_VERSION` | `0.1.0` | torch_qaic wheel version |
| `VLLM_TARGET_DEVICE_AOT` | `empty` | vLLM build target for AOT mode |
| `VLLM_TARGET_DEVICE_PYT` | `empty` | vLLM build target for PYT mode |
| `TRITON_CPU` | `0` | Set to `1` to enable triton-cpu backend (AOT SpD) |
| `TRITON_CPU_COMMIT` | `e60f448f...` | Pinned triton-cpu commit hash |
| `TRITON_CPU_SRC` | `$HOME/triton-cpu` | Clone destination for triton-cpu source |
| `TRITON_CPU_COMPILE_MAX_JOBS` | `4` | Parallel build jobs for triton-cpu compilation |
| `TRITON_CPU_SKIP_DISK_CHECK` | `0` | Set to `1` to skip the 10 GB disk-space pre-flight check |
| `TORCH_QAIC_BASE_PATH` | `/opt/qti-aic/integrations/torch_qaic` | SDK path for torch_qaic wheels |
| `VLLM_QAIC_SDK_PATH` | `/opt/qti-aic/integrations/vllm_qaic` | SDK path for pre-built vllm-qaic wheels |
