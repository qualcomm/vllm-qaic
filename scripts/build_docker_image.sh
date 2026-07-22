#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build the vllm-qaic base image (docker/Dockerfile).
#
# This image only sets up base OS/Python dependencies. To install
# vllm-qaic (aot or pyt mode) into a running container, use
# scripts/install_vllm_qaic_in_docker.sh afterwards.
#
# Usage:
#   ./build_docker_image.sh [ --python <version> ]
#                            [ --tag <image_tag> ]
#                            [ --base-image <image:tag> ]
#                            [ --force ]
#                            [ --dry-run ]

set -euo pipefail

usage() {
  cat << EOM
Usage: build_docker_image.sh [ --python <version> ]
                              [ --tag <image_tag> ]
                              [ --base-image <image:tag> ]
                              [ --force ]
                              [ --dry-run ]

--python      Python version to install in the image (default: ${DEFAULT_PYTHON_VERSION}).
--tag         Docker image tag to build (default: vllm-qaic-<base_image_slug>:latest).
--base-image  Override the QAIC SDK base image (default: ${DEFAULT_BASE_IMAGE}).
--force       Rebuild even if an image with this tag already exists.
--dry-run     Print the docker build command without running it.
EOM
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DOCKER_DIR="${REPO_ROOT}/docker"

source "${SCRIPT_DIR}/utility.sh"

DEFAULT_PYTHON_VERSION="3.12"

PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
IMAGE_TAG=""
BASE_IMAGE=""
FORCE_REBUILD="OFF"
DRY_RUN="OFF"

while [ $# -gt 0 ]; do
    if [ "z$1" == "z--python" ]; then
        PYTHON_VERSION="$2"
        shift 2
    elif [ "z$1" == "z--tag" ]; then
        IMAGE_TAG="$2"
        shift 2
    elif [ "z$1" == "z--base-image" ]; then
        BASE_IMAGE="$2"
        shift 2
    elif [ "z$1" == "z--force" ]; then
        FORCE_REBUILD="ON"
        shift
    elif [ "z$1" == "z--dry-run" ]; then
        DRY_RUN="ON"
        shift
    elif [ "z$1" == "z-h" ] || [ "z$1" == "z--help" ]; then
        usage
        exit 1
    else
        echo "Invalid option: $1" >&2
        usage
        exit 1
    fi
done

EFFECTIVE_BASE_IMAGE="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}"

if [ -z "${IMAGE_TAG}" ]; then
    IMAGE_TAG="vllm-qaic-$(base_image_tag_slug "${EFFECTIVE_BASE_IMAGE}"):latest"
fi

if [ "${FORCE_REBUILD}" == "OFF" ] && docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_TAG}$"; then
    echo "Image ${IMAGE_TAG} already exists. Skipping build (use --force to rebuild)."
    exit 0
fi

echo "================================================================"
echo "Build Configuration"
echo "----------------------------------------------------------------"
echo "PYTHON_VERSION : ${PYTHON_VERSION}"
echo "IMAGE_TAG      : ${IMAGE_TAG}"
echo "BASE_IMAGE     : ${EFFECTIVE_BASE_IMAGE}"
echo "================================================================"

CMD=(docker build
     --build-arg "PYTHON_VERSION=${PYTHON_VERSION}")

if [ -n "${BASE_IMAGE}" ]; then
    CMD+=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

CMD+=(-f "${DOCKER_DIR}/Dockerfile"
      -t "${IMAGE_TAG}"
      "${DOCKER_DIR}")

if [ "${DRY_RUN}" == "ON" ]; then
    printf '[DRY-RUN] '; printf '%q ' "${CMD[@]}"; echo
else
    "${CMD[@]}"
    echo "Built docker image: ${IMAGE_TAG}"
fi
