#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
set -e

echo "-------------------Paged Attention Test Suite-------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}"

declare -i device_id=0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Test PA with different prompt lengths: a) prompt_len < block_size
#    b) prompt_len = block_size  c) prompt_len > block_size
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Validate Paged Attention outputs with different prompt lengths"
pytest --disable-warnings -s -v "${TEST_DIR}/test_paged_attention.py::test_with_different_prompt_lengths" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --decode-bsz 4 \
    --dtype "mxfp6" \
    --kv-dtype "mxint8"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Test with different dbsz, num_kv_blocks and ctx_len - also validates batch invariance
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Validate Paged Attention outputs with different configurations"
pytest --disable-warnings -s -v "${TEST_DIR}/test_paged_attention.py::test_pa_model_outputs" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --dtype "mxfp6" \
    --kv-dtype "mxint8"

# ─────────────────────────────────────────────────────────────────────────────
# 3. Prefix caching multichat test
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Prefix Caching Test"
pytest --disable-warnings -s -v "${TEST_DIR}/../test_multichat.py" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --seq-len 32 \
    --ctx-len 1024 \
    --decode-bsz 16 \
    --dtype "mxfp6" \
    --kv-dtype "mxint8" \
    --device-id "${device_id}"

# ─────────────────────────────────────────────────────────────────────────────
# 4. Performance gains on prefix caching: work with 8K CL
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Paged Attention Performance Test"
pytest --disable-warnings -s -v "${TEST_DIR}/test_paged_attention.py::test_prefix_caching_performance" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --seq-len 128 \
    --ctx-len 8192 \
    --decode-bsz 4 \
    --dtype "mxfp6" \
    --kv-dtype "mxint8" \
    --device-id "${device_id}, ${device_id+1}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. T9 accuracy test for paged attention (T9 support for PA)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "Model Accuracy Test with Prefix Caching"
pytest --disable-warnings -s -v "${TEST_DIR}/../test_accuracy.py" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --seq-len 128 \
    --ctx-len 2048 \
    --decode-bsz 4 \
    --dtype "mxfp6" \
    --kv-dtype "mxint8" \
    --device-group 1 \
    --device-id "${device_id}" \
    --prefix-cache "True"

echo "Model Accuracy Test without Prefix Caching"
pytest --disable-warnings -s -v "${TEST_DIR}/../test_accuracy.py" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --seq-len 128 \
    --ctx-len 2048 \
    --decode-bsz 4 \
    --dtype "mxfp6" \
    --kv-dtype "mxint8" \
    --device-group 1 \
    --device-id "${device_id}" \
    --prefix-cache "False"
# ─────────────────────────────────────────────────────────────────────────────
# 6. MultiModal Test With Paged Attention
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Paged Attention: MultiModal (Qwen2.5-VL) ==="
pytest --disable-warnings -s -v "${TEST_DIR}/../multimodal/test_multimodal.py::test_paged_attention_single_image" \
    --model-name "Qwen/Qwen2.5-VL-3B-Instruct" \
    --seq-len 128 \
    --ctx-len 4096 \
    --decode-bsz 1 \
    --dtype "auto" \
    --quantization "mxfp6" \
    --kv-dtype "mxint8" \
    --device-group 1 \
    --device-id "${device_id},${device_id+1}" \
    --max-end-tokens 40
