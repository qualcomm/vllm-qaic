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
#                              [--base-image <image:tag>]
#                              [--wheel-name <name.whl>] [--dry-run]

set -euo pipefail

DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_DEVICE_ARCH="v68"

usage() {
  cat << EOM
Usage: build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
                        [--outdir <dir>] [--device-arch v68|v81]
                        [--base-image <image:tag>]
                        [--wheel-name <name.whl>] [--dry-run]

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
--wheel-name   Override the generated wheel's filename (passed through as the
               WHEEL_NAME build-arg; the Dockerfile's wheel stage renames the
               wheel before exporting it). Bare filename ending in .whl — the
               directory is still controlled by --outdir. Requires an explicit
               aot or pyt target; not usable with 'both' (one name cannot
               apply to two wheels). Default: uv build's own name,
               vllm_qaic-<version>-<py>-<abi>-<platform>.whl.
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
WHEEL_NAME=""
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
        --wheel-name) WHEEL_NAME="$2"; shift 2 ;;
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

# --wheel-name names exactly one wheel file, so it can't cover a 'both' build
# (which produces an aot and a pyt wheel). 'both' is also the default target,
# so an override with no explicit target lands here too.
if [[ -n "${WHEEL_NAME}" && "${BUILD_TARGET}" == "both" ]]; then
    echo "ERROR: --wheel-name cannot be used with target 'both' — one filename" >&2
    echo "       cannot name two wheels. Pass an explicit 'aot' or 'pyt' target," >&2
    echo "       e.g. ./scripts/build_wheels.sh aot --wheel-name ${WHEEL_NAME}" >&2
    exit 1
fi

# Restrict to the character set legal in a wheel filename (PEP 427): rejects
# path separators (--outdir owns the directory) and anything that could break
# out of the quoting in the Dockerfile's rename step.
if [[ -n "${WHEEL_NAME}" && ! "${WHEEL_NAME}" =~ ^[A-Za-z0-9._+-]+\.whl$ ]]; then
    echo "ERROR: --wheel-name must be a bare filename ending in .whl, using only" >&2
    echo "       letters, digits, '.', '_', '+' and '-' (got '${WHEEL_NAME}')" >&2
    exit 1
fi

# Renaming touches only the filename, not the .dist-info inside the wheel. pip
# reads the distribution, version and compatibility tags off the filename and
# requires the distribution to match that metadata — so a name that isn't a
# PEP 427 wheel filename, or whose distribution part isn't vllm_qaic, still
# builds and exports fine but cannot be pip-installed. Warn, don't fail: a
# custom name is legitimate for archiving or republishing the artifact.
if [[ -n "${WHEEL_NAME}" ]]; then
    # <distribution>-<version>[-<build>]-<pytag>-<abitag>-<plattag>.whl — no
    # field may contain '-', and a build tag must start with a digit.
    WHEEL_NAME_RE="^[A-Za-z0-9_.]+-[A-Za-z0-9_.!+]+(-[0-9][A-Za-z0-9_.]*)?"
    WHEEL_NAME_RE+="-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+\.whl$"

    if [[ ! "${WHEEL_NAME}" =~ ${WHEEL_NAME_RE} ]]; then
        echo "WARNING: '${WHEEL_NAME}' is not a valid wheel filename (expected" >&2
        echo "         <distribution>-<version>[-<build>]-<pytag>-<abitag>-<plattag>.whl," >&2
        echo "         e.g. vllm_qaic-1.22.0-py3-none-any.whl). It will be built and" >&2
        echo "         exported under that name, but pip cannot parse or install it." >&2
    elif [[ "${WHEEL_NAME}" != vllm_qaic-* ]]; then
        echo "WARNING: '${WHEEL_NAME}' does not name distribution 'vllm_qaic'. The" >&2
        echo "         wheel's internal metadata still says vllm_qaic, and pip rejects" >&2
        echo "         a wheel whose filename disagrees with it. Fine for archiving" >&2
        echo "         under a custom name; 'pip install' of it will fail." >&2
    fi
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
echo "WHEEL_NAME     : ${WHEEL_NAME:-<uv build default>}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "================================================================"

BASE_IMAGE_ARGS=()
if [ -n "${BASE_IMAGE}" ]; then
    BASE_IMAGE_ARGS=(--build-arg "BASE_IMAGE=${BASE_IMAGE}")
fi

# Left empty unless overridden, so the wheel stage keeps uv build's own name.
WHEEL_NAME_ARGS=()
if [ -n "${WHEEL_NAME}" ]; then
    WHEEL_NAME_ARGS=(--build-arg "WHEEL_NAME=${WHEEL_NAME}")
fi

build_aot_wheel() {
    echo ""
    echo "=== Building AOT wheel (pure Python, py3-none-any — pyver-independent) ==="
    run_echo docker buildx build --target wheel -f "${DOCKER_DIR}/Dockerfile.aot" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        "${BASE_IMAGE_ARGS[@]}" \
        "${WHEEL_NAME_ARGS[@]}" \
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
        "${WHEEL_NAME_ARGS[@]}" \
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

# An override replaces the whole filename, so the result check looks for that
# exact name rather than the default glob. Safe to set both patterns — the
# validation above rejects --wheel-name for anything but a single target.
AOT_WHEEL_NAME="vllm_qaic-*aot*.whl"
PYT_WHEEL_NAME="vllm_qaic-*pyt*.whl"
if [ -n "${WHEEL_NAME}" ]; then
    AOT_WHEEL_NAME="${WHEEL_NAME}"
    PYT_WHEEL_NAME="${WHEEL_NAME}"
fi

if [ "${DRY_RUN}" == "ON" ]; then
    echo "  (dry-run: no wheels were actually built)"
elif [ "${BUILD_TARGET}" == "aot" ] || [ "${BUILD_TARGET}" == "both" ]; then
    report_wheel "aot" "${OUT_DIR}/aot/${AOT_WHEEL_NAME}"
fi
if [ "${DRY_RUN}" == "OFF" ] && { [ "${BUILD_TARGET}" == "pyt" ] || [ "${BUILD_TARGET}" == "both" ]; }; then
    report_wheel "pyt" "${OUT_DIR}/pyt/${PYVER_TAG}/${PYT_WHEEL_NAME}"
fi

echo "================================================================"

exit "${RUN_STATUS}"
