#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

#################### ci_fasttest_qaic.sh ##############################
# goal: To quickly run through vllm tests to check if system works
# May customize this script for your own test cases
# $ bash -x ci_fasttest_qaic.sh
# $ bash -x ci_fasttest_qaic.sh --host localhost --port 8080
# $ bash -x ci_fasttest_qaic.sh --device-ids 0,1,2,3,4
# $ bash -x ci_fasttest_qaic.sh --timeout 900
########################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

trap 'rm -rf "$SCRIPT_DIR/scheduler_output"' EXIT

cd "$SCRIPT_DIR/.." || exit

HOST="localhost"
PORT="8080"
TIMEOUT="1800"

# Parse arguments
POSITIONAL=()
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
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"  # restore positional parameters

# Device pool available to this run. Resolution order:
#   1. --device-ids CLI arg (explicit, highest precedence)
#   2. DEVICE_IDS env var (e.g. to pin a fixed subset on a GHA runner)
#   3. auto-discovered QIDs currently Status:Ready with Nsp Free == Nsp Total,
#      per `qaic-util -q`
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

OUTPUT_DIR="$SCRIPT_DIR/scheduler_output/fasttest"
mkdir -p "$OUTPUT_DIR"
JOBS_FILE="$OUTPUT_DIR/jobs.json"

echo "[ci_fasttest_qaic] Collecting jobs from: tests/e2e"
DEVICE_POOL_SIZE=$(($(tr -cd ',' <<< "$DEVICE_IDS" | wc -c) + 1))
python3 "$SCRIPT_DIR/collect_jobs.py" tests/e2e --output "$JOBS_FILE" -- \
    --ignore=tests/e2e/multimodal/test_multimodal.py \
    --device-pool-size "$DEVICE_POOL_SIZE" \
    --host "$HOST" \
    --port "$PORT"
COLLECT_STATUS=$?
if [[ $COLLECT_STATUS -ne 0 ]]; then
    echo "ERROR: collect_jobs.py failed (exit $COLLECT_STATUS)" >&2
    exit 1
fi

echo "[ci_fasttest_qaic] Dispatching against device pool: $DEVICE_IDS"
SCHED_STATUS=0
python3 "$SCRIPT_DIR/scheduler.py" "$JOBS_FILE" \
    --device-ids "$DEVICE_IDS" \
    --output-dir "$OUTPUT_DIR" \
    --timeout "$TIMEOUT" || SCHED_STATUS=$?

if [[ $SCHED_STATUS -ne 0 ]]; then
    exit 1
fi
