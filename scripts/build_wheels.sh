#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build vllm-qaic wheels for AOT and/or PYT modes.
#
# Thin wrapper around `docker buildx build --target wheel`. Each Dockerfile
# (docker/Dockerfile.aot, docker/Dockerfile.pyt) exposes a `wheel` BuildKit
# target that builds the wheel inside its own base stage and exports it via
# BuildKit's local exporter — no ephemeral container, bind mount, or
# host-uid/gid mapping needed. Wheels land on the host under --outdir
# (default: <repo_root>/dist), same layout as before.
#
# Usage:
#   ./scripts/build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
#                              [--outdir <dir>] [--device-arch v68|v81]
#                              [--base-image <image:tag>] [--dry-run]

set -euo pipefail

DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_DEVICE_ARCH="v68"

usage() {
  cat << EOM
Usage: build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
                        [--outdir <dir>] [--device-arch v68|v81]
                        [--base-image <image:tag>] [--dry-run]

aot|pyt|both   Wheel(s) to build (default: both).
--pyver        Python version to build with: 3.10, 3.11, or 3.12
               (default: ${DEFAULT_PYTHON_VERSION}).
--outdir       Wheel output directory (default: <repo_root>/dist).
--device-arch  QAIC device arch for PYT kernel builds: v68 (AI100) or v81 (AI200).
               Bypasses setup.py's live-device probe, needed when devices
               aren't accessible at build time (default: ${DEFAULT_DEVICE_ARCH}).
               Ignored for aot mode.
--base-image   Override the QAIC SDK base image (passed through as the
               BASE_IMAGE build-arg; default: each Dockerfile's own ARG
               default).
--dry-run      Print the docker buildx commands without running them.
EOM
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DOCKER_DIR="${REPO_ROOT}/docker"

# Defaults
BUILD_TARGET="both"
PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
OUT_DIR="${REPO_ROOT}/dist"
DEVICE_ARCH="${DEFAULT_DEVICE_ARCH}"
BASE_IMAGE=""
DRY_RUN="OFF"

if [[ $# -gt 0 && "$1" != --* ]]; then
    BUILD_TARGET="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyver) PYTHON_VERSION="$2"; shift 2 ;;
        --outdir) OUT_DIR="$2"; shift 2 ;;
        --device-arch) DEVICE_ARCH="$2"; shift 2 ;;
        --base-image) BASE_IMAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN="ON"; shift ;;
        -h|--help) usage; exit 1 ;;
        *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ "${BUILD_TARGET}" != "aot" && "${BUILD_TARGET}" != "pyt" && "${BUILD_TARGET}" != "both" ]]; then
    echo "ERROR: unknown target '${BUILD_TARGET}'. Use aot|pyt|both" >&2
    exit 1
fi

if [[ "${PYTHON_VERSION}" != "3.10" && "${PYTHON_VERSION}" != "3.11" && "${PYTHON_VERSION}" != "3.12" ]]; then
    echo "ERROR: --pyver must be 3.10, 3.11, or 3.12" >&2
    exit 1
fi

if [[ "${DEVICE_ARCH}" != "v68" && "${DEVICE_ARCH}" != "v81" ]]; then
    echo "ERROR: --device-arch must be 'v68' or 'v81'" >&2
    exit 1
fi

PYVER_TAG="py${PYTHON_VERSION//./}"

run_echo() {
    if [[ "${DRY_RUN}" == "ON" ]]; then
        printf '[DRY-RUN] '; printf '%q ' "$@"; echo
    else
        "$@"
    fi
}

echo "================================================================"
echo "Build Configuration"
echo "----------------------------------------------------------------"
echo "PYTHON_VERSION : ${PYTHON_VERSION}"
echo "MODE           : ${BUILD_TARGET}"
echo "DEVICE_ARCH    : ${DEVICE_ARCH} (pyt only)"
echo "BASE_IMAGE     : ${BASE_IMAGE:-<Dockerfile default>}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "================================================================"

BASE_IMAGE_ARGS=()
if [ -n "${BASE_IMAGE}" ]; then
    BASE_IMAGE_ARGS=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

build_aot_wheel() {
    echo ""
    echo "=== Building AOT wheel (pure Python, py3-none-any — pyver-independent) ==="
    run_echo docker buildx build --target wheel -f "${DOCKER_DIR}/Dockerfile.aot" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        "${BASE_IMAGE_ARGS[@]}" \
        --output "type=local,dest=${OUT_DIR}/aot" \
        "${REPO_ROOT}"
}

build_pyt_wheel() {
    echo ""
    echo "=== Building PYT wheel (Hexagon compiled) ==="
    run_echo docker buildx build --target wheel -f "${DOCKER_DIR}/Dockerfile.pyt" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg QAIC_DEVICE_ARCH="${DEVICE_ARCH}" \
        "${BASE_IMAGE_ARGS[@]}" \
        --output "type=local,dest=${OUT_DIR}/pyt/${PYVER_TAG}" \
        "${REPO_ROOT}"
}

case "${BUILD_TARGET}" in
    aot)  build_aot_wheel ;;
    pyt)  build_pyt_wheel ;;
    both) build_aot_wheel; build_pyt_wheel ;;
esac

echo ""
echo "================================================================"
echo "  Wheel build results"
echo "----------------------------------------------------------------"

RUN_STATUS=0

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
elif [ "${BUILD_TARGET}" == "aot" ] || [ "${BUILD_TARGET}" == "both" ]; then
    report_wheel "aot" "${OUT_DIR}/aot/vllm_qaic-*aot*.whl"
fi
if [ "${DRY_RUN}" == "OFF" ] && { [ "${BUILD_TARGET}" == "pyt" ] || [ "${BUILD_TARGET}" == "both" ]; }; then
    report_wheel "pyt" "${OUT_DIR}/pyt/${PYVER_TAG}/vllm_qaic-*pyt*.whl"
fi

echo "================================================================"

exit "${RUN_STATUS}"
