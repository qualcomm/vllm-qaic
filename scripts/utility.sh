# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Version constants for vllm-qaic build and install scripts.
# Source this file from other scripts: source "$(dirname "$0")/utility.sh"

# === Common ===
VLLM_VERSION="0.23.0"
VLLM_QAIC_VERSION="1.22"

# === AOT stack ===
TORCH_VERSION_AOT="2.7.0+cpu"   # CPU-only torch for AOT; qefficient pins to exactly 2.7.0+cpu
TORCHVISION_VERSION_AOT="0.22.0+cpu"  # torchvision for AOT; needed by transformers (gemma3n → timm)
QEFF_BRANCH="release/v1.22.0"               # or tag e.g. "v0.3.2"
# TRANSFORMERS_VERSION_AOT="4.55.3"  # optional: uncomment to pin if qefficient's version conflicts
VLLM_TARGET_DEVICE_AOT="${VLLM_TARGET_DEVICE_AOT:-empty}"
TRITON_CPU="${TRITON_CPU:-0}"                          # set to 1 to enable
TRITON_CPU_COMMIT="${TRITON_CPU_COMMIT:-e60f448f8f197073b75d6d3e77347414a5db3ee7}"
TRITON_CPU_SRC="${TRITON_CPU_SRC:-${HOME}/triton-cpu}"                 # clone destination
TRITON_CPU_COMPILE_MAX_JOBS="${TRITON_CPU_COMPILE_MAX_JOBS:-4}"        # parallel build jobs

# === PYT stack ===
TORCH_VERSION_PYT="2.10.0+cpu"
TORCH_QAIC_VERSION="0.1.0"
# TRANSFORMERS_VERSION_PYT="4.57.3"  # optional: uncomment to pin if torch_qaic's version conflicts
VLLM_TARGET_DEVICE_PYT="${VLLM_TARGET_DEVICE_PYT:-cpu}"
TORCHVISION_VERSION_PYT="0.25.0+cpu"
TORCHAUDIO_VERSION_PYT="2.10.0+cpu"

# === Paths ===
TORCH_QAIC_BASE_PATH="/opt/qti-aic/integrations/torch_qaic"
VLLM_QAIC_SDK_PATH="${VLLM_QAIC_SDK_PATH:-/opt/qti-aic/integrations/vllm_qaic}"
