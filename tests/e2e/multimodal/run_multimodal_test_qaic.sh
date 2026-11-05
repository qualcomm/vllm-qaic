#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
set -e

echo "---------Multimodal Test Suite-------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}"

# Pool of QAIC device ids available to this run (override with env var if needed).
# Vision tests need 2 devices (pooling/encoder LLM + generation LLM); the
# whisper/audio tests need 1.
DEVICE_ID_POOL="${DEVICE_ID_POOL:-0,1}"

# ─────────────────────────────────────────────────────────────────────────────
# Vision tests
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Multimodal: Qwen3-VL ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::TestQwen3VL"

echo ""
echo "=== Multimodal: Qwen2.5-VL ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::TestQwen25VL"

echo ""
echo "=== Multimodal: InternVL ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::TestInternVL"

echo ""
echo "=== Multimodal: Llava ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::TestLlava"

echo ""
echo "=== Multimodal: Gemma3 ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::TestGemma3"

echo ""
echo "=== Multimodal: Granite ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::TestGranite"

# ─────────────────────────────────────────────────────────────────────────────
# Whisper tests (audio transcription).
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Multimodal: Whisper ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal.py::test_whisper"

echo ""
echo "=== Multimodal: Audio Transcription (OpenAI API Server) ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_multimodal_openai.py::test_audio"

echo ""
echo "---------Multimodal Test Suite Complete-------------------"
