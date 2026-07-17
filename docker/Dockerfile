# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Base Dockerfile for vllm-qaic.
#
# This image only sets up base OS and Python dependencies — it does NOT
# install vllm-qaic, torch, or vllm. That happens later, inside a running
# container, via docker/install_vllm_qaic_in_docker.sh (which creates a
# "vllm-env" venv and runs scripts/install.sh in aot or pyt mode).
#
# The BASE_IMAGE must have the QAIC Platform and Apps SDKs installed
# (i.e. /opt/qti-aic/ present) so that vllm-qaic can load at runtime.
#
# Build (use docker/build_docker_image.sh, or directly from the repo root):
#   docker build -f docker/Dockerfile \
#     --build-arg PYTHON_VERSION=3.12 \
#     -t vllm-qaic:latest docker/
#
# Override the base image if needed:
#   docker build -f docker/Dockerfile \
#     --build-arg BASE_IMAGE=ghcr.io/quic/cloud_ai_inference_ubuntu24:1.21.6.0 \
#     -t vllm-qaic:latest docker/
# ------------------------------------------------------------------

ARG BASE_IMAGE="docker-registry.qualcomm.com/qraniumtest/qranium:1.22.1.19-ubuntu22-x86_64"
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}

ARG PYTHON_VERSION="3.12"

# Base OS + Python deps only. python3-venv lets install_vllm_qaic_in_docker.sh
# create a "vllm-env" venv at install time; build-essential/cmake are needed
# because vllm-qaic's csrc/ Hexagon kernels compile during its pip install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cmake \
        sudo \
    && rm -rf /var/lib/apt/lists/*

# Tell vLLM to load the QAIC platform plugin on startup.
ENV VLLM_PLUGINS="qaic"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
