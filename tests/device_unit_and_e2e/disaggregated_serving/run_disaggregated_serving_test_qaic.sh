#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
set -e

echo "---------Disaggregated Serving Test Suite-------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}"

# Pool of QAIC device ids available to this run (override with env var if needed).
# Need atleast 4 devices
DEVICE_ID_POOL="${DEVICE_ID_POOL:-0,1,2,3}"

echo ""
echo "=== Disaggregated Serving: Basic Suite ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_qaic_disagg.py::TestDisaggregatedServingBasic"

echo ""
echo "=== Disaggregated Serving: Prefill Streaming Benchmark ==="
pytest --disable-warnings -s -vv \
    --device-id "${DEVICE_ID_POOL}" \
    "${TEST_DIR}/test_prefill_streaming.py::test_prefill_streaming_metrics"

echo ""
echo "---------Disaggregated Serving Test Suite Complete-------------------"
