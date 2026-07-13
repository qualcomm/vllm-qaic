# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Standalone Dockerfile for vllm-qaic in PYT (PyTorch Eager) mode.
#
# PYT/Eager mode uses torch_qaic (from the QAIC Apps SDK) as the
# inference engine. It requires torch==2.10.0+cpu and must NOT have
# qefficient installed in the same environment.
#
# The BASE_IMAGE must have the QAIC Apps SDK installed at /opt/qti-aic
# (both Apps and Platform SDKs). The builder stage runs directly on top
# of BASE_IMAGE so that torch_qaic (linked against Ubuntu's OpenSSL 3)
# and the QAIC SDK libs are natively available without a bind-mount.
#
# Build (run from the vllm-qaic repo root):
#   docker build -f docker/Dockerfile.pyt \
#     -t vllm-qaic-pyt:latest .
#
# Override the base image if needed:
#   docker build -f docker/Dockerfile.pyt \
#     --build-arg BASE_IMAGE=ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0 \
#     -t vllm-qaic-pyt:latest .
# ------------------------------------------------------------------

ARG BASE_IMAGE="ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0"
ARG VENV="/opt/venv-pyt"
# Target device architecture — controls which C++ kernel sources are compiled:
#   v81 = AI 200 series (includes BF16 kernels)
#   v68 = AI 100 series (BF16 kernels excluded)
# Pass at build time: --build-arg QAIC_DEVICE_ARCH=v81
ARG QAIC_DEVICE_ARCH="v68"

# ---------------------------------------------------------------------------
# Stage 1: Build the full PYT Python environment on top of the SDK base image.
#   - Using BASE_IMAGE (Ubuntu 24.04) rather than manylinux ensures OpenSSL 3
#     is present, which torch_qaic._C requires at import time.
#   - /opt/qti-aic is already in the image: no bind-mount needed.
#   - Build tools (g++, cmake) are installed via apt then removed after use.
#   - Only the finished venv is retained in the final stage.
# ---------------------------------------------------------------------------
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS pyt_venv_builder

ARG VENV
ARG QAIC_DEVICE_ARCH
ARG PIP_CACHE_DIR="/var/cache/pip"

# 1a. C++ build tools needed to compile the vllm_qaic Hexagon kernel
#     extensions from csrc/ during the vllm-qaic install step.
RUN --mount=type=cache,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,sharing=locked,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3.12-dev \
        python3.12-venv

# 1b. Build-time Python tools required for --no-build-isolation installs.
RUN --mount=type=cache,sharing=locked,target=${PIP_CACHE_DIR} \
    pip install \
        "setuptools>=77.0.3,<80.0.0" \
        setuptools-scm \
        wheel \
        "cmake>=3.26"

# Create the venv.
RUN python3.12 -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# 1c–1f. Delegate the full PYT install sequence to install.sh, which handles:
#   torch==2.10.0+cpu → torch_qaic → vllm deps → vllm-cpu → vllm-qaic.
#   /opt/qti-aic is already present in the image (from BASE_IMAGE), so
#   torch_qaic, libQAic.so, and qaic-util are all directly accessible.
#   VLLM_VERSION_OVERRIDE bypasses get_qaic_sdk_version() to prevent
#   libQAic.so's stdout output from corrupting pip's version capture.
#   QAIC_DEVICE_ARCH bypasses _get_device_arch() which requires live devices.
COPY . /src/vllm-qaic
RUN --mount=type=cache,sharing=locked,target=${PIP_CACHE_DIR} \
    VIRTUAL_ENV="${VENV}" \
    VLLM_VERSION_OVERRIDE="0.15.0+pyt1.22" \
    QAIC_DEVICE_ARCH="${QAIC_DEVICE_ARCH}" \
    bash /src/vllm-qaic/scripts/install.sh pyt

# ---------------------------------------------------------------------------
# Stage 2: Runtime image.
#   Start fresh from BASE_IMAGE and drop in only the finished venv,
#   leaving behind build tools and intermediate build artefacts.
# ---------------------------------------------------------------------------
# hadolint ignore=DL3006
FROM ${BASE_IMAGE} AS final

ARG VENV

COPY --from=pyt_venv_builder ${VENV} ${VENV}

ENV PATH="${VENV}/bin:${PATH}"

# Tell vLLM to load the QAIC platform plugin on startup.
ENV VLLM_PLUGINS="qaic"
