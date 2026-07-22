#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Build vllm-qaic wheels for AOT and/or PYT modes.
#
# Wheels are ALWAYS built inside a docker container — there is no native
# build path. Running this script builds/reuses the vllm-qaic base image
# (see scripts/build_docker_image.sh), launches an ephemeral container
# with the repo bind-mounted, and re-invokes itself inside that container
# to do the actual build. Wheels land on the host under --outdir (default:
# <repo_root>/dist), same layout either way.
#
# Usage:
#   ./scripts/build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
#                              [--outdir <dir>] [--image <image_tag>]
#                              [--base-image <image:tag>]
#                              [--device-arch v68|v81] [--force] [--dry-run]

set -euo pipefail

usage() {
  cat << EOM
Usage: build_wheels.sh [aot|pyt|both] [--pyver 3.10|3.11|3.12]
                        [--outdir <dir>] [--image <image_tag>]
                        [--base-image <image:tag>]
                        [--device-arch v68|v81] [--force] [--dry-run]

aot|pyt|both   Wheel(s) to build (default: both).
--pyver        Python version to build with: 3.10, 3.11, or 3.12
               (default: ${DEFAULT_PYTHON_VERSION}).
--outdir       Wheel output directory (default: <repo_root>/dist).
--image        Docker image to build/reuse (default: vllm-qaic-<mode>-py<XYZ>-<base_image_slug>:latest).
--base-image   Override the QAIC SDK base image (passed through to
               build_docker_image.sh's --base-image; default: ${DEFAULT_BASE_IMAGE}).
--device-arch  QAIC device arch for PYT kernel builds: v68 (AI100) or v81 (AI200).
               Bypasses setup.py's live-device probe, needed when devices
               aren't accessible at build time (default: ${DEFAULT_DEVICE_ARCH}).
               Ignored for aot mode.
--force        Force a rebuild of the docker image even if it already exists.
--dry-run      Print the docker commands without running them.
EOM
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${SCRIPT_DIR}/utility.sh"

DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_DEVICE_ARCH="v68"

# Defaults
BUILD_TARGET="both"
PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
OUT_DIR="${REPO_ROOT}/dist"
IMAGE_TAG=""
BASE_IMAGE=""
DEVICE_ARCH="${DEFAULT_DEVICE_ARCH}"
FORCE_REBUILD="OFF"
DRY_RUN="OFF"

if [[ $# -gt 0 && "$1" != --* ]]; then
    BUILD_TARGET="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pyver) PYTHON_VERSION="$2"; shift 2 ;;
        --outdir) OUT_DIR="$2"; shift 2 ;;
        --image) IMAGE_TAG="$2"; shift 2 ;;
        --base-image) BASE_IMAGE="$2"; shift 2 ;;
        --device-arch) DEVICE_ARCH="$2"; shift 2 ;;
        --force) FORCE_REBUILD="ON"; shift ;;
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

EFFECTIVE_BASE_IMAGE="${BASE_IMAGE:-${DEFAULT_BASE_IMAGE}}"

if [[ -z "${IMAGE_TAG}" ]]; then
    IMAGE_TAG="vllm-qaic-${BUILD_TARGET}-${PYVER_TAG}-$(base_image_tag_slug "${EFFECTIVE_BASE_IMAGE}"):latest"
fi

run_echo() {
    if [[ "${DRY_RUN}" == "ON" ]]; then
        printf '[DRY-RUN] '; printf '%q ' "$@"; echo
    else
        "$@"
    fi
}

# ------------------------------------------------------------------
# Inner mode: runs INSIDE the container (set by the outer docker run below).
# Does the actual wheel build in the active Python environment.
# ------------------------------------------------------------------
run_build_in_container() {
    # Prefer the active venv's python so that packages installed by PIP (e.g. 'build')
    # are importable when running 'python -m build'. Fall back to the system pythonX.Y.
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYTHON="${VIRTUAL_ENV}/bin/python"
    else
        PYTHON="python${PYTHON_VERSION}"
        if ! command -v "${PYTHON}" &>/dev/null; then
            echo "ERROR: ${PYTHON} not found in PATH. Install Python ${PYTHON_VERSION} first." >&2
            exit 1
        fi
    fi

    # Use uv pip if available and the active venv was created by uv (faster installs).
    # uv venvs are identified by a 'uv = ' line in pyvenv.cfg.
    UV=$(command -v uv 2>/dev/null || true)
    PIP="${PYTHON} -m pip"
    USE_UV=0
    if [ -n "${UV}" ] && [ -n "${VIRTUAL_ENV:-}" ] && \
       grep -q "^uv = " "${VIRTUAL_ENV}/pyvenv.cfg" 2>/dev/null; then
        PIP="${UV} pip"
        USE_UV=1
        echo "INFO: uv venv detected — using uv pip for faster installs"
    fi

    echo "========================================================"
    echo "  vllm-qaic wheel build"
    echo "  target   : ${BUILD_TARGET}"
    echo "  python   : ${PYTHON_VERSION}"
    echo "  outdir   : ${OUT_DIR}"
    echo "========================================================"

    # Install build system deps needed for --no-isolation builds.
    # Mirrors the build-deps step in install.sh.
    setup_build_tools() {
        ${PIP} install "setuptools>=77.0.3,<80.0.0" setuptools-scm wheel build "cmake>=3.26"
    }

    # Install PYT build prerequisites: CPU torch first, then torch_qaic.
    # Mirrors install.sh Step 1 (PYT) — CPU torch must precede torch_qaic because
    # torch_qaic validates at import that torch is CPU-only.
    # torch must use python -m pip because uv refuses +local version labels
    # (e.g. torch==2.10.0+cpu) from remote indexes.
    setup_pyt_build_deps() {
        local pyver_tag="py${PYTHON_VERSION/./}"
        echo "=== Installing PYT build deps ==="
        ${PIP} install pip
        ${PYTHON} -m pip install --quiet \
            --index-url https://download.pytorch.org/whl/cpu \
            "torch==${TORCH_VERSION_PYT}" \
            "torchvision==${TORCHVISION_VERSION_PYT}" \
            "torchaudio==${TORCHAUDIO_VERSION_PYT}"
        ${PIP} install "${TORCH_QAIC_BASE_PATH}/${pyver_tag}"/torch_qaic-*.whl
    }

    build_aot_wheel() {
        echo ""
        echo "=== Building AOT wheel (pure Python, py3-none-any — pyver-independent) ==="
        setup_build_tools
        local aot_out="${OUT_DIR}/aot"
        mkdir -p "${aot_out}"

        cd "${REPO_ROOT}"
        # Force AOT mode: TORCH_QAIC_INSTALLED=0 overrides auto-detection
        TORCH_QAIC_INSTALLED=0 \
            ${PYTHON} -m build --wheel --no-isolation \
            --outdir "${aot_out}"

        local whl
        whl=$(ls "${aot_out}"/vllm_qaic-*aot*.whl 2>/dev/null | head -1)
        if [ -z "${whl}" ]; then
            echo "ERROR: AOT wheel not found in ${aot_out}" >&2
            exit 1
        fi
        echo "AOT wheel: ${whl}"
    }

    build_pyt_wheel() {
        echo ""
        echo "=== Building PYT wheel (Hexagon compiled) ==="
        setup_build_tools

        # Install torch_qaic + torch if not already present
        if ! ${PYTHON} -c "import torch_qaic" 2>/dev/null; then
            setup_pyt_build_deps
        fi

        local pyver_tag="py${PYTHON_VERSION/./}"
        local pyt_out="${OUT_DIR}/pyt/${pyver_tag}"
        mkdir -p "${pyt_out}"

        cd "${REPO_ROOT}"
        # torch_qaic present → auto-detected as PYT mode
        ${PYTHON} -m build --wheel --no-isolation \
            --outdir "${pyt_out}"

        local whl
        whl=$(ls "${pyt_out}"/vllm_qaic-*pyt*.whl 2>/dev/null | head -1)
        if [ -z "${whl}" ]; then
            echo "ERROR: PYT wheel not found in ${pyt_out}" >&2
            exit 1
        fi
        echo "PYT wheel: ${whl}"
    }

    case "${BUILD_TARGET}" in
        aot)  build_aot_wheel ;;
        pyt)  build_pyt_wheel ;;
        both) build_aot_wheel; build_pyt_wheel ;;
    esac

    echo ""
    echo "========================================================"
    echo "  Build complete. Wheels in ${OUT_DIR}/"
    echo "========================================================"
}

if [[ "${IN_VLLM_QAIC_CONTAINER:-0}" == "1" ]]; then
    run_build_in_container
    exit 0
fi

# ------------------------------------------------------------------
# Outer mode: builds/reuses the docker image, then re-invokes this
# script inside an ephemeral container to do the actual build.
# ------------------------------------------------------------------
echo "================================================================"
echo "Build Configuration"
echo "----------------------------------------------------------------"
echo "PYTHON_VERSION : ${PYTHON_VERSION}"
echo "MODE           : ${BUILD_TARGET}"
echo "IMAGE_TAG      : ${IMAGE_TAG}"
echo "BASE_IMAGE     : ${EFFECTIVE_BASE_IMAGE}"
echo "OUT_DIR        : ${OUT_DIR}"
echo "================================================================"

BUILD_IMAGE_ARGS=(--python "${PYTHON_VERSION}" --tag "${IMAGE_TAG}")
if [ -n "${BASE_IMAGE}" ]; then
    BUILD_IMAGE_ARGS+=(--base-image "${BASE_IMAGE}")
fi
if [ "${FORCE_REBUILD}" == "ON" ]; then
    BUILD_IMAGE_ARGS+=(--force)
fi
if [ "${DRY_RUN}" == "ON" ]; then
    BUILD_IMAGE_ARGS+=(--dry-run)
fi

run_echo bash "${SCRIPT_DIR}/build_docker_image.sh" "${BUILD_IMAGE_ARGS[@]}"

USER_UID="$(id -u)"
USER_GID="$(id -g)"
QAIC_GID="$(getent group qaic | cut -d: -f3 || true)"

BUILD_CMD="VENV=\"\${HOME}/wheel-build-env\"; \
python${PYTHON_VERSION} -m venv \"\${VENV}\"; \
IN_VLLM_QAIC_CONTAINER=1 VIRTUAL_ENV=\"\${VENV}\" QAIC_VISIBLE_DEVICES=0"
if [ "${BUILD_TARGET}" == "pyt" ] || [ "${BUILD_TARGET}" == "both" ]; then
    BUILD_CMD="${BUILD_CMD} QAIC_DEVICE_ARCH=\"${DEVICE_ARCH}\""
fi
BUILD_CMD="${BUILD_CMD} \
    bash scripts/build_wheels.sh ${BUILD_TARGET} --pyver ${PYTHON_VERSION} --outdir ${OUT_DIR}; \
unset QAIC_VISIBLE_DEVICES QAIC_DEVICE_ARCH"

echo "Building vllm_qaic wheel(s) [${BUILD_TARGET}] for python ${PYTHON_VERSION}..."
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
elif [ "${BUILD_TARGET}" == "aot" ] || [ "${BUILD_TARGET}" == "both" ]; then
    report_wheel "aot" "${OUT_DIR}/aot/vllm_qaic-*aot*.whl"
fi
if [ "${DRY_RUN}" == "OFF" ] && { [ "${BUILD_TARGET}" == "pyt" ] || [ "${BUILD_TARGET}" == "both" ]; }; then
    report_wheel "pyt" "${OUT_DIR}/pyt/${PYVER_TAG}/vllm_qaic-*pyt*.whl"
fi

echo "================================================================"

exit "${RUN_STATUS}"
