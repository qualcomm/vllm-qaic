# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Dockerfile.pyt — PYT (PyTorch Eager) mode for vllm-qaic.
#
# Stage overview
# --------------
#   pyt-base   Shared foundation: system packages, uv venv, build tools,
#              torch + torch_qaic (SDK wheel), vllm deps + vllm.
#              Can be built and pushed as a standalone image.
#
#   release    FROM pyt-base. Installs vllm-qaic non-editable from a
#              published git ref (tag or branch). Fully self-contained.
#
#   ci         FROM pyt-base. Installs vllm-qaic non-editable from the
#              current build-context checkout, a branch, or a PR number.
#
#   dev        FROM pyt-base. Installs vllm-qaic editable (pip install -e)
#              for fast in-container iteration.
#
#   wheel      FROM pyt-base. Builds a vllm-qaic PYT wheel (compiling the
#              Hexagon kernel extension) from the build-context checkout
#              and exports it via BuildKit's local exporter.
#
# Key differences from Dockerfile.aot
# ------------------------------------
#   - torch_qaic comes from the SDK wheel at /opt/qti-aic (not from git)
#   - vllm is installed from source with target device "empty" (no C++ build)
#   - vllm_qaic itself compiles Hexagon kernel C++ extensions — build-essential
#     and cmake are required; QAIC_DEVICE_ARCH controls which sources compile
#   - No triton-cpu step (PYT mode has no equivalent)
#   - TORCH_QAIC_INSTALLED is not set (auto-detected = 1 via torch_qaic import)
#
# Build commands
# --------------
#   # Standalone base:
#   docker build --target pyt-base -t vllm-qaic-pyt-base:1.22 .
#
#   # Release (default pins):
#   docker build --target release -f docker/Dockerfile.pyt -t vllm-qaic-pyt:1.22 .
#
#   # Release (AI200 device):
#   docker build --target release -f docker/Dockerfile.pyt \
#     --build-arg QAIC_DEVICE_ARCH=v81 -t vllm-qaic-pyt:1.22-v81 .
#
#   # CI (current checkout):
#   docker build --target ci -f docker/Dockerfile.pyt -t vllm-qaic-pyt:ci .
#
#   # CI (specific PR):
#   docker build --target ci -f docker/Dockerfile.pyt \
#     --build-arg VLLM_QAIC_PR=42 -t vllm-qaic-pyt:ci-pr-42 .
#
#   # Dev (editable vllm-qaic):
#   docker build --target dev -f docker/Dockerfile.pyt -t vllm-qaic-pyt:dev .
#
#   # Wheel (specific python version + device arch, export to ./dist/pyt/py311):
#   docker buildx build --target wheel -f docker/Dockerfile.pyt \
#     --build-arg PYTHON_VERSION=3.11 --build-arg QAIC_DEVICE_ARCH=v81 \
#     --output type=local,dest=./dist/pyt/py311 .
#
# The BASE_IMAGE must have the QAIC Platform and Apps SDKs installed
# (i.e. /opt/qti-aic/ present with torch_qaic wheels).
#
# Run dev container with host-matching UID/GID and qaic group access:
#   docker run -it --rm \
#     -e USER_UID=$(id -u) -e USER_GID=$(id -g) \
#     -e QAIC_GID=$(getent group qaic | cut -d: -f3) \
#     --device /dev/accel/ -v $(pwd):/src/vllm-qaic \
#     vllm-qaic-pyt:dev
#
# Verify any target:
#   docker run --rm <image> python -c "import vllm_qaic; print('OK')"
# ------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Infrastructure ARGs
# ---------------------------------------------------------------------------
ARG BASE_IMAGE="ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0"
ARG VENV="/opt/venv-pyt"
ARG UV_VERSION="0.11.29"
ARG PYTHON_VERSION="3.12"

# ---------------------------------------------------------------------------
# Shared stack version pins — defaults mirror scripts/utility.sh.
# ---------------------------------------------------------------------------
ARG VLLM_VERSION="0.23.0"
ARG VLLM_QAIC_VERSION="1.22"
ARG TORCH_VERSION_PYT="2.11.0+cpu"
ARG TORCHVISION_VERSION_PYT="0.26.0+cpu"
ARG TORCHAUDIO_VERSION_PYT="2.11.0+cpu"
ARG VLLM_TARGET_DEVICE_PYT="empty"
ARG TORCH_QAIC_BASE_PATH="/opt/qti-aic/integrations/torch_qaic"

# ---------------------------------------------------------------------------
# Device architecture for Hexagon kernel C++ extension compile.
#   v68 = AI 100 series (BF16 kernels excluded)
#   v81 = AI 200 series (includes BF16 kernels)
# ---------------------------------------------------------------------------
ARG QAIC_DEVICE_ARCH="v68"

# ---------------------------------------------------------------------------
# Release-specific
# ---------------------------------------------------------------------------
ARG VLLM_QAIC_GIT_REF="v0.23.0"

# ---------------------------------------------------------------------------
# CI-specific (both empty → use build-context COPY)
# ---------------------------------------------------------------------------
ARG VLLM_QAIC_PR=""
ARG VLLM_QAIC_BRANCH=""

# ---------------------------------------------------------------------------
# Pinned uv binary — FROM supports ARG substitution, COPY --from does not.
# ---------------------------------------------------------------------------
# hadolint ignore=DL3006
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ===========================================================================
# pyt-base — shared foundation for all three targets.
#
# Layer structure (each RUN is an independent cache layer):
#   1. infra      : apt + uv venv + build-deps (merged)
#   2. step1      : torch + torchvision + torchaudio (python -m pip)
#   3. step2      : torch_qaic SDK wheel from /opt/qti-aic
#   4. step3      : vllm_dependency_pyt.txt + build-deps re-pin + vllm deps + vllm
# ===========================================================================
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS pyt-base

COPY --from=uv /uv /usr/local/bin/uv

# UV environment — set once, inherited by all downstream stages.
# UV_PYTHON_DOWNLOADS=manual: never auto-download a runtime implicitly; the
# infra layer below explicitly installs PYTHON_VERSION via `uv python install`.
# UV_PYTHON_INSTALL_DIR: pinned to a known path (uv's default is
# $HOME/.local/share/uv/python) so it can be COPY'd alongside VENV into
# release/ci/dev — `uv venv --seed` symlinks the venv's python binary to this
# external store rather than copying it in, so any final stage that only
# copies VENV out of its builder is left with a dangling symlink.
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=manual \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_NO_PROGRESS=1 \
    UV_HTTP_TIMEOUT=500 \
    UV_INDEX_STRATEGY=unsafe-best-match \
    UV_CACHE_DIR=/var/cache/uv

# Use bash for all RUN steps.
SHELL ["/bin/bash", "-c"]

# Declare all ARGs after the last ENV block — ARGs declared before an ENV
# instruction can lose their RUN-scope binding in Docker BuildKit.
ARG VENV="/opt/venv-pyt"
ARG PYTHON_VERSION="3.12"
ARG VLLM_VERSION="0.23.0"
ARG VLLM_QAIC_VERSION="1.22"
ARG TORCH_VERSION_PYT="2.11.0+cpu"
ARG TORCHVISION_VERSION_PYT="0.26.0+cpu"
ARG TORCHAUDIO_VERSION_PYT="2.11.0+cpu"
ARG VLLM_TARGET_DEVICE_PYT="empty"
ARG TORCH_QAIC_BASE_PATH="/opt/qti-aic/integrations/torch_qaic"
ARG QAIC_DEVICE_ARCH="v68"

# ---------------------------------------------------------------------------
# Layer 1 — infra: system packages + uv-managed python + build tools (merged)
# build-essential + cmake: needed to compile vllm_qaic's Hexagon kernel C++
# extensions from csrc/ during pip install. Python comes from uv (not apt)
# so PYTHON_VERSION can be any of 3.10/3.11/3.12 regardless of what the base
# image's apt repos carry — uv's standalone CPython builds ship their own
# headers, so no python3.X-dev package is needed either.
# NOTE: the uv-installed Python must NOT sit under a cache mount — uv venv
# symlinks into it, and a cache mount's contents vanish when the RUN ends,
# leaving a dangling symlink in the venv for every later layer.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    --mount=type=cache,sharing=locked,target=/var/cache/uv \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git && \
    uv python install ${PYTHON_VERSION} && \
    uv venv ${VENV} --python ${PYTHON_VERSION} --seed && \
    PATH="${VENV}/bin:${PATH}" uv pip install \
        "setuptools>=77.0.3,<80.0.0" \
        setuptools-scm \
        setuptools-rust \
        wheel \
        "cmake>=3.26"

ENV PATH="${VENV}/bin:${PATH}" \
    VIRTUAL_ENV="${VENV}"

# ---------------------------------------------------------------------------
# Layer 2 — step1: torch + torchvision + torchaudio
# Must use python -m pip (uv refuses +local version labels like 2.10.0+cpu).
# torch_qaic validates at import that torch is CPU-only; install torch first.
# ---------------------------------------------------------------------------
COPY scripts/utility.sh /src/vllm-qaic/scripts/utility.sh
RUN python -m pip install --quiet \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION_PYT}" \
        "torchvision==${TORCHVISION_VERSION_PYT}" \
        "torchaudio==${TORCHAUDIO_VERSION_PYT}"

# ---------------------------------------------------------------------------
# Layer 3 — step2: torch_qaic from SDK wheel
# The wheel lives at ${TORCH_QAIC_BASE_PATH}/py<PYTHON_VERSION>/torch_qaic-*.whl
# inside the BASE_IMAGE. It links against Ubuntu's OpenSSL 3 and libQAic.so.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,sharing=locked,target=/var/cache/uv \
    PYVER="py$(echo "${PYTHON_VERSION}" | tr -d '.')" && \
    uv pip install "${TORCH_QAIC_BASE_PATH}/${PYVER}"/torch_qaic-*.whl

# ---------------------------------------------------------------------------
# Layer 4 — step3: vllm dependencies + vllm
# Re-pin build deps after the dependency install which may downgrade them.
# vllm-cpu is a prebuilt PyPI wheel — no C++ compilation needed here.
# TORCH_DEVICE_BACKEND_AUTOLOAD=0: vllm's setup.py does `import torch` at
# module load; torch auto-loads torch_qaic as a registered backend extension,
# which enumerates live QAIC devices and SIGABRTs when none are present in
# this build environment. Scoped to this RUN only — real containers still
# want torch_qaic auto-loaded at runtime.
# ---------------------------------------------------------------------------
COPY requirements/vllm_dependency_pyt.txt /src/vllm-qaic/requirements/vllm_dependency_pyt.txt
RUN --mount=type=cache,sharing=locked,target=/var/cache/uv \
    uv pip install -r /src/vllm-qaic/requirements/vllm_dependency_pyt.txt && \
    uv pip install \
        "setuptools>=77.0.3,<80.0.0" setuptools-scm setuptools-rust wheel "cmake>=3.26" && \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE_PYT}" uv pip install \
        --no-build-isolation --no-deps \
        "vllm @ git+https://github.com/vllm-project/vllm.git@v${VLLM_VERSION}"

ENV VLLM_PLUGINS="qaic"

# ===========================================================================
# RELEASE — non-editable install from a published git ref.
# Clones the ref, overlays our fixed setup.py, then installs.
# ===========================================================================
FROM pyt-base AS release-builder

ARG VENV="/opt/venv-pyt"
ARG VLLM_VERSION="0.23.0"
ARG VLLM_QAIC_VERSION="1.22"
ARG VLLM_QAIC_GIT_REF="v0.23.0"
ARG QAIC_DEVICE_ARCH="v68"

COPY setup.py /src/vllm-qaic-setup.py
RUN --mount=type=cache,sharing=locked,target=/var/cache/uv \
    git clone --branch "${VLLM_QAIC_GIT_REF}" --depth 1 \
        https://github.com/qualcomm/vllm-qaic.git /src/vllm-qaic-release && \
    cp /src/vllm-qaic-setup.py /src/vllm-qaic-release/setup.py && \
    QAIC_DEVICE_ARCH="${QAIC_DEVICE_ARCH}" \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION}+pyt${VLLM_QAIC_VERSION}" \
    uv pip install --no-build-isolation /src/vllm-qaic-release

# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS release

ARG VENV="/opt/venv-pyt"

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=release-builder ${VENV} ${VENV}
COPY --from=release-builder /opt/uv-python /opt/uv-python

ENV PATH="${VENV}/bin:${PATH}" \
    VIRTUAL_ENV="${VENV}" \
    VLLM_PLUGINS="qaic"

# ===========================================================================
# CI — non-editable install from build-context checkout, branch, or PR.
# Priority: VLLM_QAIC_PR > VLLM_QAIC_BRANCH > build-context COPY (default).
# ===========================================================================
FROM pyt-base AS ci-builder

ARG VENV="/opt/venv-pyt"
ARG VLLM_VERSION="0.23.0"
ARG VLLM_QAIC_VERSION="1.22"
ARG VLLM_QAIC_PR=""
ARG VLLM_QAIC_BRANCH=""
ARG QAIC_DEVICE_ARCH="v68"

COPY . /src/vllm-qaic
RUN --mount=type=cache,sharing=locked,target=/var/cache/uv \
    if [ -n "${VLLM_QAIC_PR}" ]; then \
        echo "=== CI: vllm-qaic from PR #${VLLM_QAIC_PR} ===" && \
        git clone https://github.com/qualcomm/vllm-qaic.git /src/vllm-qaic-ci && \
        git -C /src/vllm-qaic-ci fetch origin "refs/pull/${VLLM_QAIC_PR}/head" && \
        git -C /src/vllm-qaic-ci checkout FETCH_HEAD && \
        SRC=/src/vllm-qaic-ci; \
    elif [ -n "${VLLM_QAIC_BRANCH}" ]; then \
        echo "=== CI: vllm-qaic from branch ${VLLM_QAIC_BRANCH} ===" && \
        git clone --branch "${VLLM_QAIC_BRANCH}" --depth 1 \
            https://github.com/qualcomm/vllm-qaic.git /src/vllm-qaic-ci && \
        SRC=/src/vllm-qaic-ci; \
    else \
        echo "=== CI: vllm-qaic from build context ===" && \
        SRC=/src/vllm-qaic; \
    fi && \
    QAIC_DEVICE_ARCH="${QAIC_DEVICE_ARCH}" \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION}+pyt${VLLM_QAIC_VERSION}" \
    uv pip install --no-build-isolation "${SRC}"

# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS ci

ARG VENV="/opt/venv-pyt"

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=ci-builder ${VENV} ${VENV}
COPY --from=ci-builder /opt/uv-python /opt/uv-python

ENV PATH="${VENV}/bin:${PATH}" \
    VIRTUAL_ENV="${VENV}" \
    VLLM_PLUGINS="qaic"

# ===========================================================================
# DEV — editable install of vllm-qaic for fast iteration.
# torch_qaic is a binary SDK wheel — no editable override needed.
# Source dir is kept in the final image (editable install requires it).
# ===========================================================================
FROM pyt-base AS dev-builder

ARG VENV="/opt/venv-pyt"
ARG VLLM_VERSION="0.23.0"
ARG VLLM_QAIC_VERSION="1.22"
ARG QAIC_DEVICE_ARCH="v68"

COPY . /src/vllm-qaic
RUN --mount=type=cache,sharing=locked,target=/var/cache/uv \
    QAIC_DEVICE_ARCH="${QAIC_DEVICE_ARCH}" \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION}+pyt${VLLM_QAIC_VERSION}" \
    uv pip install --no-build-isolation -e /src/vllm-qaic

# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS dev

ARG VENV="/opt/venv-pyt"

# sudo + entrypoint.sh: map the container user to the host's uid/gid (and the
# host's qaic device group) on `docker run`, so bind-mounted files keep host
# ownership and /dev/accel stays accessible without running as root.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends sudo

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=dev-builder ${VENV} ${VENV}
COPY --from=dev-builder /opt/uv-python /opt/uv-python
COPY --from=dev-builder /src/vllm-qaic /src/vllm-qaic
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV PATH="${VENV}/bin:${PATH}" \
    VIRTUAL_ENV="${VENV}" \
    VLLM_PLUGINS="qaic" \
    VENV="${VENV}"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

# ===========================================================================
# WHEEL — build a vllm-qaic PYT wheel from the build-context checkout and
# export it via BuildKit's local exporter (no container run / bind mount
# needed). PYT wheels are cp<pyver>-linux_x86_64 (compiles Hexagon kernels).
#
#   docker buildx build --target wheel -f docker/Dockerfile.pyt \
#     --build-arg PYTHON_VERSION=3.11 --build-arg QAIC_DEVICE_ARCH=v81 \
#     --output type=local,dest=./dist/pyt/py311 .
# ===========================================================================
FROM pyt-base AS wheel-builder

ARG VLLM_VERSION="0.23.0"
ARG VLLM_QAIC_VERSION="1.22"
ARG QAIC_DEVICE_ARCH="v68"

COPY . /src/vllm-qaic
RUN --mount=type=cache,sharing=locked,target=/var/cache/uv \
    QAIC_DEVICE_ARCH="${QAIC_DEVICE_ARCH}" \
    VLLM_VERSION_OVERRIDE="${VLLM_VERSION}+pyt${VLLM_QAIC_VERSION}" \
    uv build --wheel --no-build-isolation --out-dir /out /src/vllm-qaic

FROM scratch AS wheel
COPY --from=wheel-builder /out /
