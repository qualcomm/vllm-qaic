#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trap 'rm -rf "$SCRIPT_DIR/scheduler_output"' EXIT

cd "$SCRIPT_DIR/.." || exit

if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ ! -f vllm_qaic/hexagon_kernels.so ]]; then
    echo "[ci_fasttest_qaic_pyt] Building source-tree QAIC custom kernels"
    if ! python3 setup.py build_ext --inplace; then
        echo "ERROR: failed to build source-tree QAIC custom kernels" >&2
        exit 1
    fi
fi

HOST="localhost"
PORT="8080"
TIMEOUT="1800"

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --device-ids)
            DEVICE_IDS_ARG="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

_discover_free_device_ids() {
    /opt/qti-aic/tools/qaic-util -q | awk '
        /^QID / { qid=$2; status=""; nsp_total=-1; nsp_free=-1 }
        /Status:/ { status=$0; sub(/^[ \t]*Status:/, "", status) }
        /Nsp Total:/ { nsp_total=$0; sub(/^[ \t]*Nsp Total:/, "", nsp_total); nsp_total+=0 }
        /Nsp Free:/ {
            nsp_free=$0; sub(/^[ \t]*Nsp Free:/, "", nsp_free); nsp_free+=0
            if (status == "Ready" && nsp_total > 0 && nsp_free == nsp_total) print qid
        }
    ' | paste -sd, -
}

DEVICE_IDS="${DEVICE_IDS_ARG:-${DEVICE_IDS:-$(_discover_free_device_ids)}}"
if [[ -z "$DEVICE_IDS" ]]; then
    echo "ERROR: no QAIC devices are Status:Ready with Nsp Free" >&2
    exit 1
fi

OUTPUT_DIR="$SCRIPT_DIR/scheduler_output/pyt_fasttest"
mkdir -p "$OUTPUT_DIR"
JOBS_FILE="$OUTPUT_DIR/jobs.json"
echo "[ci_fasttest_qaic_pyt] Collecting eligible PyTorch jobs from: tests/e2e"
DEVICE_POOL_SIZE=$(($(tr -cd ',' <<< "$DEVICE_IDS" | wc -c) + 1))
python3 "$SCRIPT_DIR/collect_jobs.py" tests/e2e --output "$JOBS_FILE" -- \
    --ignore=tests/e2e/lora/test_qaic_lora.py \
    --device-pool-size "$DEVICE_POOL_SIZE" \
    --host "$HOST" \
    --port "$PORT"
COLLECT_STATUS=$?
if [[ $COLLECT_STATUS -ne 0 ]]; then
    echo "ERROR: collect_jobs.py failed (exit $COLLECT_STATUS)" >&2
    exit 1
fi

echo "[ci_fasttest_qaic_pyt] Dispatching against device pool: $DEVICE_IDS"
python3 "$SCRIPT_DIR/scheduler.py" "$JOBS_FILE" \
    --device-ids "$DEVICE_IDS" \
    --output-dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" \
    --set-qaic-visible-devices
