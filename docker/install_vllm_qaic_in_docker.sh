#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Host-side script that installs vllm-qaic (aot or pyt mode) into a
# vllm-qaic docker container.
#
# It starts a long-lived container from the base image (docker/Dockerfile)
# if one isn't already running, bind-mounting the repo, then execs into it
# as a non-root user (matching the host's uid/gid, joined to the host's
# qaic group) to create a "vllm-env" venv and run scripts/install.sh.
#
# Usage:
#   ./install_vllm_qaic_in_docker.sh --mode aot|pyt
#                                     [ --python 3.10|3.11|3.12 ]
#                                     [ --image <image_tag> ]
#                                     [ --container-name <name> ]
#                                     [ --device-arch v68|v81 ]
#                                     [ --dry-run ]

set -euo pipefail

usage() {
  cat << EOM
Usage: install_vllm_qaic_in_docker.sh --mode aot|pyt
                                       [ --python 3.10|3.11|3.12 ]
                                       [ --image <image_tag> ]
                                       [ --container-name <name> ]
                                       [ --device-arch v68|v81 ]
                                       [ --dry-run ]

--mode            Install mode: aot or pyt (required).
--python          Python version to create the venv with: 3.10, 3.11, or
                   3.12 (default: ${DEFAULT_PYTHON_VERSION}). Also used to
                   derive the default container name.
--image           Docker image to run if the container isn't already up
                   (default: ${DEFAULT_IMAGE_TAG}).
--container-name  Name of the long-lived container to start/reuse
                   (default: vllm-qaic-<mode>-py<XYZ>-dev).
--device-arch     QAIC device arch for PYT builds: v68 (AI100) or v81
                   (AI200). Bypasses setup.py's live-device probe, needed
                   when devices aren't accessible at install time
                   (default: ${DEFAULT_DEVICE_ARCH}). Ignored for --mode aot.
--dry-run         Print the docker commands without running them.
EOM
}

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${DOCKER_DIR}")"

DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_IMAGE_TAG="vllm-qaic:latest"
DEFAULT_DEVICE_ARCH="v68"

MODE=""
PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
IMAGE_TAG="${DEFAULT_IMAGE_TAG}"
CONTAINER_NAME=""
DEVICE_ARCH="${DEFAULT_DEVICE_ARCH}"
DRY_RUN="OFF"

while [ $# -gt 0 ]; do
    if [ "z$1" == "z--mode" ]; then
        MODE="$2"
        shift 2
    elif [ "z$1" == "z--python" ]; then
        PYTHON_VERSION="$2"
        shift 2
    elif [ "z$1" == "z--image" ]; then
        IMAGE_TAG="$2"
        shift 2
    elif [ "z$1" == "z--container-name" ]; then
        CONTAINER_NAME="$2"
        shift 2
    elif [ "z$1" == "z--device-arch" ]; then
        DEVICE_ARCH="$2"
        shift 2
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

if [ "${MODE}" != "aot" ] && [ "${MODE}" != "pyt" ]; then
    echo "ERROR: --mode must be 'aot' or 'pyt'" >&2
    usage
    exit 1
fi

if [ "${PYTHON_VERSION}" != "3.10" ] && [ "${PYTHON_VERSION}" != "3.11" ] && [ "${PYTHON_VERSION}" != "3.12" ]; then
    echo "ERROR: --python must be 3.10, 3.11, or 3.12" >&2
    usage
    exit 1
fi

if [ "${DEVICE_ARCH}" != "v68" ] && [ "${DEVICE_ARCH}" != "v81" ]; then
    echo "ERROR: --device-arch must be 'v68' or 'v81'" >&2
    usage
    exit 1
fi

PYVER_TAG="py${PYTHON_VERSION//./}"

if [ -z "${CONTAINER_NAME}" ]; then
    CONTAINER_NAME="vllm-qaic-${MODE}-${PYVER_TAG}-dev"
fi

run_echo() {
    if [ "${DRY_RUN}" == "ON" ]; then
        printf '[DRY-RUN] '; printf '%q ' "$@"; echo
    else
        "$@"
    fi
}

USER_UID="$(id -u)"
USER_GID="$(id -g)"
QAIC_GID="$(getent group qaic | cut -d: -f3 || true)"

CONTAINER_RUNNING="$(docker ps --filter "name=^${CONTAINER_NAME}$" --filter "status=running" --format '{{.Names}}' || true)"

if [ -z "${CONTAINER_RUNNING}" ]; then
    echo "Starting container '${CONTAINER_NAME}' from image '${IMAGE_TAG}'..."
    run_echo docker run -d --name "${CONTAINER_NAME}" \
        -e USER_UID="${USER_UID}" -e USER_GID="${USER_GID}" -e QAIC_GID="${QAIC_GID}" \
        --device /dev/accel/ \
        --volume "${REPO_ROOT}:${REPO_ROOT}" \
        --workdir "${REPO_ROOT}" \
        "${IMAGE_TAG}" sleep infinity
else
    echo "Reusing already-running container '${CONTAINER_NAME}'."
fi

VENV_INSTALL_CMD="VENV=\"\${HOME}/vllm-env\"; \
[ -d \"\${VENV}\" ] || python${PYTHON_VERSION} -m venv \"\${VENV}\"; \
QAIC_VISIBLE_DEVICES=0"
if [ "${MODE}" == "pyt" ]; then
    VENV_INSTALL_CMD="${VENV_INSTALL_CMD} QAIC_DEVICE_ARCH=\"${DEVICE_ARCH}\""
fi
VENV_INSTALL_CMD="${VENV_INSTALL_CMD} VIRTUAL_ENV=\"\${VENV}\" bash scripts/install.sh ${MODE}; \
unset QAIC_VISIBLE_DEVICES QAIC_DEVICE_ARCH"

echo "Installing vllm-qaic (${MODE} mode) into '${CONTAINER_NAME}'..."
run_echo docker exec --user "${USER_UID}" --workdir "${REPO_ROOT}" "${CONTAINER_NAME}" \
    bash -c "${VENV_INSTALL_CMD}"

echo ""
echo "================================================================"
echo "  vllm-qaic (${MODE}) installed in container '${CONTAINER_NAME}'"
echo "  venv: ~/vllm-env"
echo "  Activate with: docker exec --user ${USER_UID} -it ${CONTAINER_NAME} bash -c 'source ~/vllm-env/bin/activate && bash'"
echo "================================================================"
