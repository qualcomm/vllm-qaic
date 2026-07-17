#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build vllm_qaic wheel(s) inside a docker container.
#
# Builds (or reuses) the base vllm-qaic image for the requested Python
# version via build_docker_image.sh, then runs an ephemeral container that
# creates a venv and invokes scripts/build_wheels.sh for the requested
# mode(s). The repo is bind-mounted, so wheels land directly on the host
# under --outdir (default: <repo_root>/dist), same layout as running
# build_wheels.sh natively.
#
# Usage:
#   ./build_vllm_qaic_whl.sh --python 3.10|3.11|3.12
#                             --mode aot|pyt|both
#                             [ --image <image_tag> ]
#                             [ --outdir <dir> ]
#                             [ --device-arch v68|v81 ]
#                             [ --force ]
#                             [ --dry-run ]

set -euo pipefail

usage() {
  cat << EOM
Usage: build_vllm_qaic_whl.sh --python 3.10|3.11|3.12
                               --mode aot|pyt|both
                               [ --image <image_tag> ]
                               [ --outdir <dir> ]
                               [ --device-arch v68|v81 ]
                               [ --force ]
                               [ --dry-run ]

--python       Python version to build with: 3.10, 3.11, or 3.12 (required).
--mode         Wheel(s) to build: aot, pyt, or both (required).
--image        Docker image to build/reuse (default: vllm-qaic-<mode>-py<XYZ>:latest).
--outdir       Wheel output directory (default: <repo_root>/dist).
--device-arch  QAIC device arch for PYT builds: v68 (AI100) or v81 (AI200).
               Bypasses setup.py's live-device probe, needed when devices
               aren't accessible at build time (default: ${DEFAULT_DEVICE_ARCH}).
               Ignored for --mode aot.
--force        Force a rebuild of the docker image even if it already exists.
--dry-run      Print the docker commands without running them.
EOM
}

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${DOCKER_DIR}")"

DEFAULT_DEVICE_ARCH="v68"

PYTHON_VERSION=""
MODE=""
IMAGE_TAG=""
OUT_DIR=""
DEVICE_ARCH="${DEFAULT_DEVICE_ARCH}"
FORCE_REBUILD="OFF"
DRY_RUN="OFF"

while [ $# -gt 0 ]; do
    if [ "z$1" == "z--python" ]; then
        PYTHON_VERSION="$2"
        shift 2
    elif [ "z$1" == "z--mode" ]; then
        MODE="$2"
        shift 2
    elif [ "z$1" == "z--image" ]; then
        IMAGE_TAG="$2"
        shift 2
    elif [ "z$1" == "z--outdir" ]; then
        OUT_DIR="$2"
        shift 2
    elif [ "z$1" == "z--device-arch" ]; then
        DEVICE_ARCH="$2"
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

if [ "${PYTHON_VERSION}" != "3.10" ] && [ "${PYTHON_VERSION}" != "3.11" ] && [ "${PYTHON_VERSION}" != "3.12" ]; then
    echo "ERROR: --python must be 3.10, 3.11, or 3.12" >&2
    usage
    exit 1
fi

if [ "${MODE}" != "aot" ] && [ "${MODE}" != "pyt" ] && [ "${MODE}" != "both" ]; then
    echo "ERROR: --mode must be 'aot', 'pyt', or 'both'" >&2
    usage
    exit 1
fi

if [ "${DEVICE_ARCH}" != "v68" ] && [ "${DEVICE_ARCH}" != "v81" ]; then
    echo "ERROR: --device-arch must be 'v68' or 'v81'" >&2
    usage
    exit 1
fi

PYVER_TAG="py${PYTHON_VERSION//./}"

if [ -z "${IMAGE_TAG}" ]; then
    IMAGE_TAG="vllm-qaic-${MODE}-${PYVER_TAG}:latest"
fi

if [ -z "${OUT_DIR}" ]; then
    OUT_DIR="${REPO_ROOT}/dist"
fi

run_echo() {
    if [ "${DRY_RUN}" == "ON" ]; then
        printf '[DRY-RUN] '; printf '%q ' "$@"; echo
    else
        "$@"
    fi
}

echo "================================================================"
echo "Build Configuration"
echo "----------------------------------------------------------------"
echo "PYTHON_VERSION : ${PYTHON_VERSION}"
echo "MODE           : ${MODE}"
echo "IMAGE_TAG      : ${IMAGE_TAG}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "================================================================"

BUILD_IMAGE_ARGS=(--python "${PYTHON_VERSION}" --tag "${IMAGE_TAG}")
if [ "${FORCE_REBUILD}" == "ON" ]; then
    BUILD_IMAGE_ARGS+=(--force)
fi
if [ "${DRY_RUN}" == "ON" ]; then
    BUILD_IMAGE_ARGS+=(--dry-run)
fi

run_echo bash "${DOCKER_DIR}/build_docker_image.sh" "${BUILD_IMAGE_ARGS[@]}"

USER_UID="$(id -u)"
USER_GID="$(id -g)"
QAIC_GID="$(getent group qaic | cut -d: -f3 || true)"

BUILD_CMD="VENV=\"\${HOME}/wheel-build-env\"; \
python${PYTHON_VERSION} -m venv \"\${VENV}\"; \
VIRTUAL_ENV=\"\${VENV}\" QAIC_VISIBLE_DEVICES=0"
if [ "${MODE}" == "pyt" ] || [ "${MODE}" == "both" ]; then
    BUILD_CMD="${BUILD_CMD} QAIC_DEVICE_ARCH=\"${DEVICE_ARCH}\""
fi
BUILD_CMD="${BUILD_CMD} \
    bash scripts/build_wheels.sh ${MODE} --pyver ${PYTHON_VERSION} --outdir ${OUT_DIR}; \
unset QAIC_VISIBLE_DEVICES QAIC_DEVICE_ARCH"

echo "Building vllm_qaic wheel(s) [${MODE}] for python ${PYTHON_VERSION}..."
RUN_STATUS=0
run_echo docker run --rm \
    -e USER_UID="${USER_UID}" -e USER_GID="${USER_GID}" -e QAIC_GID="${QAIC_GID}" \
    --device /dev/accel/ \
    --volume "${REPO_ROOT}:${REPO_ROOT}" \
    --workdir "${REPO_ROOT}" \
    "${IMAGE_TAG}" bash -c "${BUILD_CMD}" || RUN_STATUS=$?

echo ""
echo "================================================================"
echo "  Wheel build results"
echo "----------------------------------------------------------------"

report_wheel() {
    local label="$1"
    local pattern="$2"
    local whl
    whl=$(ls ${pattern} 2>/dev/null | head -1)
    if [ -n "${whl}" ]; then
        echo "  ${label}: FOUND (${whl})"
    else
        echo "  ${label}: MISSING (expected ${pattern})"
        RUN_STATUS=1
    fi
}

if [ "${DRY_RUN}" == "ON" ]; then
    echo "  (dry-run: no wheels were actually built)"
elif [ "${MODE}" == "aot" ] || [ "${MODE}" == "both" ]; then
    report_wheel "aot" "${OUT_DIR}/aot/vllm_qaic-*aot*.whl"
fi
if [ "${DRY_RUN}" == "OFF" ] && { [ "${MODE}" == "pyt" ] || [ "${MODE}" == "both" ]; }; then
    report_wheel "pyt" "${OUT_DIR}/pyt/${PYVER_TAG}/vllm_qaic-*pyt*.whl"
fi

echo "================================================================"

exit "${RUN_STATUS}"
