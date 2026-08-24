#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
set -e

echo "---------LoRA Test Suite-------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}"

# Pool of QAIC device ids available to this run (override with env var if needed).
# Single-LLM tests need 1 device; the disaggregated prefill/decode test needs 2.
DEVICE_ID_POOL="${DEVICE_ID_POOL:-0,1}"

# ─────────────────────────────────────────────────────────────────────────────
# LoRA tests: adapter cache search / adaptername-to-id mapping (no QAIC device)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== LoRA: Adapter Cache Search ==="
pytest --disable-warnings -s -vv \
    "${TEST_DIR}/test_qaic_lora_adapter_cache.py::test_qaic_search_adapters_in_cache"

echo ""
echo "=== LoRA: Adaptername-to-ID Mapping ==="
pytest --disable-warnings -s -vv \
    "${TEST_DIR}/test_qaic_lora_adapter_cache.py::test_qaic_get_qaic_model_dump_adaptername_to_id"

# ─────────────────────────────────────────────────────────────────────────────
# LoRA tests: max adapter load, offline init caching, output consistency
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== LoRA: Max Adapter Load ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_qaic_lora.py::test_llm_lora_max_adapter_load"

echo ""
echo "=== LoRA: Offline Init Caching ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_qaic_lora.py::test_llm_lora_offline_init_caching"

echo ""
echo "=== LoRA: Output Consistency ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_qaic_lora.py::test_llm_lora_consistency"

# ─────────────────────────────────────────────────────────────────────────────
# LoRA tests: online server initialization (OpenAI-compatible + minimal API)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== LoRA: Online Init (OpenAI API Server) ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_qaic_lora.py::test_llm_lora_online_openai_init"

# ─────────────────────────────────────────────────────────────────────────────
# LoRA tests: multiple concurrent LoRA requests
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== LoRA: Multiple LoRA Requests ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_qaic_generate_multiple_loras.py::test_multiple_lora_requests"
