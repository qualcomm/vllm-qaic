#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Install vllm-qaic in AOT or PYT mode.
#
# Prerequisites: activate your target environment before running this script.
#   conda activate myenv
#   # or
#   source /path/to/venv/bin/activate
#
# Usage:
#   ./scripts/install.sh aot
#   ./scripts/install.sh pyt
#
# Install source detection:
#   - Run from vllm-qaic repo (GitHub/Gerrit clone):
#       setup.py found at repo root → installs from source via pip install .
#   - Run from SDK (/opt/qti-aic/integrations/vllm_qaic/):
#       no setup.py → installs pre-built wheel from /opt/qti-aic/integrations/vllm_qaic/pyXXX/
#
# Environment overrides:
#   TRANSFORMERS_VERSION_AOT   If set, pins transformers version after qefficient install
#   TRANSFORMERS_VERSION_PYT   If set, pins transformers version after torch_qaic install

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utility.sh"

MODE="${1:-aot}"
shift || true
if [[ $# -gt 0 ]]; then
    echo "Unknown arg: $1"; exit 1
fi

if [[ "${MODE}" != "aot" && "${MODE}" != "pyt" ]]; then
    echo "ERROR: mode must be 'aot' or 'pyt'" >&2
    exit 1
fi

# Require an active Python environment.
# Prefer ${VIRTUAL_ENV}/bin/python when a venv is active — command -v python can
# resolve to a shadowing conda python when conda's bin appears in PATH before the
# venv's bin (e.g. PATH=miniconda/bin:$PATH prepended after source activate).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
else
    PYTHON=$(command -v python 2>/dev/null || true)
fi
if [ -z "${PYTHON}" ]; then
    echo "ERROR: no python found in PATH. Activate a conda env or venv first." >&2
    exit 1
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

PYVER="py$(${PYTHON} -c 'import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))')"

# Detect install source: repo (source) vs SDK (wheel)
# Override with VLLM_QAIC_INSTALL_SOURCE=wheel to force wheel install from
# a custom path: VLLM_QAIC_SDK_PATH=<dir> VLLM_QAIC_INSTALL_SOURCE=wheel install.sh pyt
# AOT wheel is py3-none-any (pyver-independent) → no PYVER subdir.
# PYT wheel is pyver-specific → lives under ${VLLM_QAIC_SDK_PATH}/${PYVER}/.
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
if [ "${VLLM_QAIC_INSTALL_SOURCE:-}" = "wheel" ] || [ ! -f "${REPO_ROOT}/setup.py" ]; then
    INSTALL_SOURCE="wheel"
else
    INSTALL_SOURCE="source"
fi
if [ "${INSTALL_SOURCE}" = "wheel" ]; then
    if [ "${MODE}" = "aot" ]; then
        SDK_WHEEL_DIR="${VLLM_QAIC_SDK_PATH}"
    else
        SDK_WHEEL_DIR="${VLLM_QAIC_SDK_PATH}/${PYVER}"
    fi
fi

TORCH_QAIC_WHEEL_DIR="${TORCH_QAIC_BASE_PATH}/${PYVER}"

echo ""
echo "========================================================"
echo "  vllm-qaic installer"
echo "  mode     : ${MODE}"
echo "  source   : ${INSTALL_SOURCE}$([ "${INSTALL_SOURCE}" = "source" ] && echo " (${REPO_ROOT})" || echo "")"
echo "  python   : $(${PYTHON} --version)"
echo "  env      : $(${PYTHON} -c 'import sys; print(sys.prefix)')"
echo "  --------------------------------------------------------"
if [ "${MODE}" = "aot" ]; then
    echo "  vllm           : ${VLLM_VERSION}       vllm-qaic: ${VLLM_QAIC_VERSION}"
    echo "  torch (AOT)    : ${TORCH_VERSION_AOT}  torchvision: ${TORCHVISION_VERSION_AOT}"
    echo "  qeff branch    : ${QEFF_BRANCH}"
    echo "  target device  : ${VLLM_TARGET_DEVICE_AOT}"
    if [ "${TRITON_CPU}" = "1" ]; then
        echo "  triton-cpu     : enabled"
        echo "    src          : ${TRITON_CPU_SRC}"
        echo "    commit       : ${TRITON_CPU_COMMIT}"
        echo "    max jobs     : ${TRITON_CPU_COMPILE_MAX_JOBS}"
    else
        echo "  triton-cpu     : disabled  (set TRITON_CPU=1 to enable)"
    fi
else
    echo "  vllm           : ${VLLM_VERSION}       vllm-qaic: ${VLLM_QAIC_VERSION}"
    echo "  torch (PYT)    : ${TORCH_VERSION_PYT}  torchvision: ${TORCHVISION_VERSION_PYT}"
    echo "  torchaudio     : ${TORCHAUDIO_VERSION_PYT}"
    echo "  torch-qaic     : ${TORCH_QAIC_VERSION}"
    echo "  target device  : ${VLLM_TARGET_DEVICE_PYT}"
fi
echo "  --------------------------------------------------------"
echo "  Override any variable before running, e.g.:"
echo "    TRITON_CPU=1 TRITON_CPU_SRC=/data/triton-cpu ./scripts/install.sh aot"
echo "========================================================"
echo ""

# vllm build deps — always needed for --no-build-isolation
${PIP} install "setuptools>=77.0.3,<80.0.0" setuptools-scm setuptools-rust wheel "cmake>=3.26"

if [ "${MODE}" = "aot" ]; then
    if ${PYTHON} -c "import importlib.metadata; importlib.metadata.version('torch-qaic')" 2>/dev/null; then
        echo "INFO: torch_qaic detected in AOT env — uninstalling..."
        if [ "${USE_UV}" = "1" ]; then
            ${PIP} uninstall torch-qaic || true
        else
            ${PIP} uninstall -y torch-qaic || true
        fi
    fi

    echo "=== Step 1: QEfficient (brings torch ${TORCH_VERSION_AOT}) ==="
    ${PIP} install "qefficient @ git+https://github.com/quic/efficient-transformers.git@${QEFF_BRANCH}"
    # uv skips +local labels from remote indexes — explicitly pin torch to AOT version.
    # Ensure pip is present first (uv venvs have no pip by default).
    if [ "${USE_UV}" = "1" ]; then
        ${PIP} install pip
    fi
    ${PYTHON} -m pip install --quiet \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION_AOT}" \
        "torchvision==${TORCHVISION_VERSION_AOT}"

    # Optional transformers pin
    if [ -n "${TRANSFORMERS_VERSION_AOT:-}" ]; then
        echo "=== Step 1b: pinning transformers==${TRANSFORMERS_VERSION_AOT} ==="
        ${PIP} install "transformers==${TRANSFORMERS_VERSION_AOT}"
    fi

    echo "=== Step 2: vllm deps + vllm (VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE_AOT}) ==="
    ${PIP} install -r "${SCRIPT_DIR}/../requirements/vllm_dependency_aot.txt"
    VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE_AOT}" ${PIP} install \
        --no-build-isolation --no-deps \
        "vllm @ git+https://github.com/vllm-project/vllm.git@v${VLLM_VERSION}"

    echo "=== Step 3: vllm-qaic-aot ==="
    if [ "${INSTALL_SOURCE}" = "source" ]; then
        TORCH_QAIC_INSTALLED=0 ${PIP} install --no-build-isolation "${REPO_ROOT}"
    else
        ${PIP} install "${SDK_WHEEL_DIR}"/vllm_qaic-*aot*.whl
    fi

    # Step 4 (optional): triton-cpu backend for AOT SpD
    if [ "${TRITON_CPU}" = "1" ]; then
        echo "=== Step 4: triton-cpu backend ==="
        bash "${SCRIPT_DIR}/install_triton_cpu.sh"
    fi

elif [ "${MODE}" = "pyt" ]; then
    echo "=== Step 1: torch==${TORCH_VERSION_PYT} (CPU) ==="
    # Install CPU torch BEFORE torch_qaic — torch_qaic validates at import that torch is
    # CPU-only and will error if CUDA torch is present.
    # Always use python -m pip for torch: uv refuses +local version labels (e.g.
    # torch==2.10.0+cpu) from remote indexes. Ensure pip is present first (uv venvs have
    # no pip by default; conda/venv always have it so this is a no-op for them).
    if [ "${USE_UV}" = "1" ]; then
        ${PIP} install pip
    fi
    ${PYTHON} -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION_PYT}" \
        "torchvision==${TORCHVISION_VERSION_PYT}" \
        "torchaudio==${TORCHAUDIO_VERSION_PYT}"

    echo "=== Step 1b: torch_qaic ==="
    ${PIP} install "${TORCH_QAIC_WHEEL_DIR}"/torch_qaic-*.whl

    # Optional transformers pin
    if [ -n "${TRANSFORMERS_VERSION_PYT:-}" ]; then
        echo "=== Step 1c: pinning transformers==${TRANSFORMERS_VERSION_PYT} ==="
        ${PIP} install "transformers==${TRANSFORMERS_VERSION_PYT}"
    fi

    echo "=== Step 2: vllm deps + vllm (VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE_PYT}) ==="
    ${PIP} install -r "${SCRIPT_DIR}/../requirements/vllm_dependency_pyt.txt"
    if [ "${VLLM_TARGET_DEVICE_PYT}" = "cpu" ]; then
        # PyPI vllm-cpu wheel: no C++ compilation, no CUDA deps.
        # QAIC functionality comes from torch_qaic (Step 1b) loaded as a vllm platform plugin.
        ${PIP} install --no-deps "vllm-cpu==${VLLM_VERSION}"
    else
        # VLLM_TARGET_DEVICE=empty: build from public GitHub tag, avoids C++ compilation
        # and produces no torch Requires-Dist in METADATA so uv will not upgrade torch.
        VLLM_TARGET_DEVICE=empty ${PIP} install \
            --no-build-isolation --no-deps \
            "vllm @ git+https://github.com/vllm-project/vllm.git@v${VLLM_VERSION}"
    fi

    echo "=== Step 3: vllm-qaic-pyt ==="
    if [ "${INSTALL_SOURCE}" = "source" ]; then
        ${PIP} install --no-build-isolation "${REPO_ROOT}"
    else
        ${PIP} install --no-deps "${SDK_WHEEL_DIR}"/vllm_qaic-*pyt*.whl
    fi
fi

echo ""
echo "========================================================"
echo "  vllm-qaic (${MODE}) installed [${INSTALL_SOURCE}]"
echo "========================================================"
