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
    - [Build args](#build-args)
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
> triton-cpu state, rust frontend state). Review it and override any variable before re-running.

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

# Build vllm's experimental Rust OpenAI-compatible frontend (rust/, build_rust.sh)
# from source instead of installing vllm from the published git tag.
# Requires a Rust toolchain (cargo) on PATH — see "Installing Rust" below.
# Applies to both aot and pyt; clones/reuses a vllm checkout as a sibling of the
# vllm-qaic repo (`../vllm`) and checks out the pinned VLLM_VERSION tag.
VLLM_BUILD_RUST=1 ./scripts/install.sh aot

# Force wheel install from a custom SDK path
VLLM_QAIC_INSTALL_SOURCE=wheel VLLM_QAIC_SDK_PATH=/path/to/sdk ./scripts/install.sh pyt
```

**Installing Rust:** `VLLM_BUILD_RUST=1` requires `cargo` on `PATH` — `install.sh` checks for it up front and exits with an error if it's missing, rather than installing it for you. Install via [rustup](https://rustup.rs):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
cargo --version   # confirm it's on PATH before re-running install.sh
```

Or via conda-forge, if you'd rather keep it inside an existing conda env:

```bash
conda install -c conda-forge rust
```

**Using the Rust frontend after installing with `VLLM_BUILD_RUST=1`:** it's opt-in at *serve* time too — the standard `vllm serve` entrypoint still runs the Python frontend unless `VLLM_USE_RUST_FRONTEND=1` is set:

```bash
VLLM_USE_RUST_FRONTEND=1 vllm serve <model> [...same flags as usual...]
```

> **Status:** this is an experimental, unfinished component of upstream vLLM (not vllm-qaic specific). It does not support every OpenAI-compatible request field yet

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
    "vllm @ git+https://github.com/vllm-project/vllm.git@v0.23.0"

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

`ci` and `dev` additionally install `requirements/test.txt` (pytest, ruff, mypy, etc.) so tests can run directly inside the image; `release`/`wheel` don't, since they're meant to be lean runtime/distribution artifacts.

`PYTHON_VERSION` (default `3.12`; also `3.10`/`3.11`) is threaded through every target via `aot-base`/`pyt-base`, provisioned with `uv python install` rather than apt so non-default versions work regardless of the base image's own Python.

### Build args

All `ARG`s are global (declared before the first `FROM`) and re-declared inside every stage that needs them — BuildKit requires that re-declaration for a global `ARG`'s value to be visible inside a stage's `RUN`/`COPY` instructions. Override any of them with `--build-arg NAME=value`.

**`docker/Dockerfile.aot`**

| ARG | Default | Description |
|---|---|---|
| `BASE_IMAGE` | `ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0` | Must have the QAIC Platform and Apps SDKs installed (`/opt/qti-aic/` present) |
| `VENV` | `/opt/venv-aot` | Path to the venv created inside the image |
| `UV_VERSION` | `0.11.29` | Pinned `uv` binary version, pulled via `COPY --from` |
| `PYTHON_VERSION` | `3.12` | Python version (`3.10`/`3.11`/`3.12`), provisioned via `uv python install` |
| `RUST_VERSION` | `1.90` | Pinned Rust toolchain image tag (`rust:<ver>-slim`); only used when `VLLM_BUILD_RUST=1` |
| `VLLM_VERSION` | `0.23.0` | vLLM release tag to install |
| `VLLM_PR` | *(empty)* | Any target: vLLM PR number to fetch (takes priority over `VLLM_BRANCH` and `VLLM_VERSION`) |
| `VLLM_BRANCH` | *(empty)* | Any target: vLLM branch to clone instead of the pinned `VLLM_VERSION` tag |
| `VLLM_QAIC_VERSION` | `1.22` | vllm-qaic SDK/version tag (wheel tag/version suffix) |
| `QEFF_BRANCH` | `release/v1.22.0` | QEfficient branch/tag to install |
| `TORCH_VERSION_AOT` | `2.7.0+cpu` | CPU torch version for AOT |
| `TORCHVISION_VERSION_AOT` | `0.22.0+cpu` | torchvision version for AOT |
| `TRITON_CPU` | `1` | Set to `1` to build the triton-cpu backend (AOT SpD); Docker defaults ON, unlike `install.sh`'s default OFF |
| `TRITON_CPU_COMMIT` | `e60f448f8f197073b75d6d3e77347414a5db3ee7` | Pinned triton-cpu commit hash |
| `TRITON_CPU_COMPILE_MAX_JOBS` | `4` | Parallel build jobs for triton-cpu compilation |
| `VLLM_BUILD_RUST` | `1` | Set to `1` to build vLLM's experimental Rust OpenAI frontend (`vllm-rs`) |
| `VLLM_QAIC_GIT_REF` | `v0.23.0` | `release` target: vllm-qaic git tag/branch to clone |
| `VLLM_QAIC_PR` | *(empty)* | `ci` target: PR number to fetch (takes priority over `VLLM_QAIC_BRANCH`) |
| `VLLM_QAIC_BRANCH` | *(empty)* | `ci` target: branch to fetch |
| `QEFF_PR` | *(empty)* | `dev` target: QEfficient PR to install editable (overrides `QEFF_BRANCH`) |

**`docker/Dockerfile.pyt`**

| ARG | Default | Description |
|---|---|---|
| `BASE_IMAGE` | `ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0` | Must have the QAIC Platform and Apps SDKs installed (`/opt/qti-aic/` present with `torch_qaic` wheels) |
| `VENV` | `/opt/venv-pyt` | Path to the venv created inside the image |
| `UV_VERSION` | `0.11.29` | Pinned `uv` binary version, pulled via `COPY --from` |
| `PYTHON_VERSION` | `3.12` | Python version (`3.10`/`3.11`/`3.12`), provisioned via `uv python install` |
| `RUST_VERSION` | `1.90` | Pinned Rust toolchain image tag (`rust:<ver>-slim`); only used when `VLLM_BUILD_RUST=1` |
| `VLLM_VERSION` | `0.23.0` | vLLM release tag to install |
| `VLLM_PR` | *(empty)* | Any target: vLLM PR number to fetch (takes priority over `VLLM_BRANCH` and `VLLM_VERSION`) |
| `VLLM_BRANCH` | *(empty)* | Any target: vLLM branch to clone instead of the pinned `VLLM_VERSION` tag |
| `VLLM_QAIC_VERSION` | `1.22` | vllm-qaic SDK/version tag (wheel tag/version suffix) |
| `TORCH_VERSION_PYT` | `2.11.0+cpu` | CPU torch version for PYT |
| `TORCHVISION_VERSION_PYT` | `0.26.0+cpu` | torchvision version for PYT |
| `TORCHAUDIO_VERSION_PYT` | `2.11.0+cpu` | torchaudio version for PYT |
| `VLLM_TARGET_DEVICE_PYT` | `empty` | vLLM build target device (`empty` = no C++ compilation) |
| `TORCH_QAIC_BASE_PATH` | `/opt/qti-aic/integrations/torch_qaic` | SDK path containing `torch_qaic` wheels inside `BASE_IMAGE` |
| `QAIC_DEVICE_ARCH` | `v68` | `v68` = AI 100 series, `v81` = AI 200 series (includes BF16 kernels); controls which Hexagon kernel C++ sources compile |
| `VLLM_BUILD_RUST` | `1` | Set to `1` to build vLLM's experimental Rust OpenAI frontend (`vllm-rs`) |
| `VLLM_QAIC_GIT_REF` | `v0.23.0` | `release` target: vllm-qaic git tag/branch to clone |
| `VLLM_QAIC_PR` | *(empty)* | `ci` target: PR number to fetch (takes priority over `VLLM_QAIC_BRANCH`) |
| `VLLM_QAIC_BRANCH` | *(empty)* | `ci` target: branch to fetch |

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

# Any target — install vllm itself from a PR or branch instead of the pinned tag
docker build --target release -f docker/Dockerfile.aot --build-arg VLLM_PR=12345 -t vllm-qaic-aot:vllm-pr-12345 .
docker build --target dev -f docker/Dockerfile.pyt --build-arg VLLM_BRANCH=some-feature-branch -t vllm-qaic-pyt:dev .

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
| `VLLM_VERSION` | `0.23.0` | vLLM release tag |
| `VLLM_QAIC_VERSION` | `0.23.0` | vllm-qaic SDK/version tag (used in wheel tag and version suffix) |
| `TORCH_VERSION_AOT` | `2.7.0+cpu` | CPU torch for AOT (matches QEfficient exact pin) |
| `TORCHVISION_VERSION_AOT` | `0.22.0+cpu` | torchvision for AOT (keep in sync with torch) |
| `TORCH_VERSION_PYT` | `2.11.0+cpu` | CPU torch for PYT |
| `TORCHVISION_VERSION_PYT` | `0.26.0+cpu` | torchvision for PYT (keep in sync with torch) |
| `TORCHAUDIO_VERSION_PYT` | `2.11.0+cpu` | torchaudio for PYT (keep in sync with torch) |
| `QEFF_BRANCH` | `main` | QEfficient branch/tag |
| `TORCH_QAIC_VERSION` | `0.1.0` | torch_qaic wheel version |
| `VLLM_TARGET_DEVICE_AOT` | `empty` | vLLM build target for AOT mode |
| `VLLM_TARGET_DEVICE_PYT` | `empty` | vLLM build target for PYT mode |
| `TRITON_CPU` | `0` | Set to `1` to enable triton-cpu backend (AOT SpD) |
| `TRITON_CPU_COMMIT` | `e60f448f...` | Pinned triton-cpu commit hash |
| `TRITON_CPU_SRC` | `<repo>/.build/triton-cpu` | Clone destination for triton-cpu source |
| `TRITON_CPU_COMPILE_MAX_JOBS` | `4` | Parallel build jobs for triton-cpu compilation |
| `TRITON_CPU_SKIP_DISK_CHECK`¹ | `0` | Set to `1` to skip the 10 GB disk-space pre-flight check |
| `TORCH_QAIC_BASE_PATH` | `/opt/qti-aic/integrations/torch_qaic` | SDK path for torch_qaic wheels |
| `VLLM_QAIC_SDK_PATH` | `/opt/qti-aic/integrations/vllm_qaic` | SDK path for pre-built vllm-qaic wheels |

¹ Defined in `scripts/install_triton_cpu.sh`, not `utility.sh`.
