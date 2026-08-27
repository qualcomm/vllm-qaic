# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

#!/usr/bin/env bash

# run_all_tests.sh — Run all vllm-qaic unit and integration tests
#
# Auto-discovers every test_*.py file under this directory so the script
# never goes stale when new test files are added.
#
# USAGE:
#   # Run all tests (hardware tests auto-skip when --model-name is not provided)
#   bash run_all_tests.sh
#
#   # Run a specific feature only
#   bash run_all_tests.sh --feature lora
#   bash run_all_tests.sh --feature spd
#   bash run_all_tests.sh --feature generic
#   bash run_all_tests.sh --feature custom_ops
#
#   # Run with on-device tests enabled
#   bash run_all_tests.sh \
#       --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --embed-model BAAI/bge-base-en-v1.5 \
#       --seq-len 128 --ctx-len 256 --decode-bsz 4 \
#       --dtype mxfp6 --kv-dtype mxint8 \
#       --device-group 1 --device-id 0
#
#   # Run all suites concurrently instead of sequentially (see collect_jobs.py /
#   # scheduler.py — mirrors the job-dispatch pattern ci_scripts/scheduler.py
#   # uses for device_unit_and_e2e/, minus the device-pool logic, since unit/
#   # tests are pure-Python and never contend for a real QAIC device pool)
#   bash run_all_tests.sh --parallel
#   bash run_all_tests.sh --parallel --workers 4
#
# MODEL SELECTION (each feature uses the right model; on-device tests
# auto-skip unless the corresponding flag is provided):
#   --model-name       Generic inference tests  (no default — must be passed)
#   --embed-model      Embedding tests          (no default — must be passed)
#   --lora-model-name  LoRA tests               (default: mistralai/Mistral-7B-v0.1)
#                      The default LoRA adapter (predibase/gsm8k) is for Mistral-7B.
#                      It is auto-downloaded from HuggingFace Hub — no --lora-path needed.
#   --lora-path        Override LoRA adapter path (optional, auto-downloads if not set)
#
# FEATURE FILTER:
#   --feature FEATURE  Run only tests for the given feature. FEATURE must be a
#                      subdirectory name. Omit to run all features.
#
#   Available features:
#     accuracy          benchmark         consistency       custom_ops
#     disaggregated     embedding         generic           lora
#     multimodal        on_device_sampling  prefixcaching
#     samplers          spd
#
#   Notes:
#     --feature generic    runs ALL tests in generic/ including:
#                          test_envs, test_platform, test_patch_config, test_patch_utils,
#                          test_quant_config, test_qaic_utils, test_chunked_prefill,
#                          test_error_handling, test_inference_correctness,
#                          test_model_loading, test_sampling_params
#     --feature custom_ops runs ALL tests in custom_ops/ including:
#                          test_ops_topk_router, test_rms_norm_kernel
#                          (test_grouped_topk_qaic.py was a duplicate of
#                          device_unit_and_e2e/test_unit_qaic_grouped_topk.py —
#                          all 4 of its tests required real QAIC hardware, so
#                          it has been moved to tests/_deprecated/)
#
# PARALLEL EXECUTION:
#   --parallel         Run each test_*.py file as an independent job via
#                      collect_jobs.py + scheduler.py instead of the default
#                      sequential loop. Every other flag (--feature,
#                      --model-name, etc.) still applies — --parallel only
#                      changes how jobs are dispatched, not which jobs run or
#                      with what args.
#   --workers N        Max concurrent jobs when --parallel is set (default: 8).
#                      Ignored without --parallel.
#
# ESTIMATED RUNTIME:
#   Without hardware:  ~1 minute  (on-device tests auto-skip)
#   With hardware:     ~20-30 minutes (depends on model load time)
#
# ---------------------------------------------------------------------------------------

set -uo pipefail
# Note: -e is intentionally omitted so the script always runs all suites
# and reports a full result even when individual suites fail.

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL_NAME=""
EMBED_MODEL=""
LORA_MODEL_NAME="mistralai/Mistral-7B-v0.1"  # predibase/gsm8k adapter is for Mistral-7B
SEQ_LEN=128
CTX_LEN=256
DECODE_BSZ=4
DTYPE="mxfp6"
KV_DTYPE="mxint8"
DEVICE_GROUP=1
DEVICE_ID=0
LORA_PATH=""   # Optional — auto-downloads predibase/gsm8k from HuggingFace if not set
VERBOSE=0
FEATURE=""     # Empty = run all features; set to a feature name to run only that feature
PARALLEL=0
WORKERS=8

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)       MODEL_NAME="$2";       shift 2 ;;
        --embed-model)      EMBED_MODEL="$2";      shift 2 ;;
        --lora-model-name)  LORA_MODEL_NAME="$2";  shift 2 ;;
        --seq-len)          SEQ_LEN="$2";          shift 2 ;;
        --ctx-len)          CTX_LEN="$2";          shift 2 ;;
        --decode-bsz)       DECODE_BSZ="$2";       shift 2 ;;
        --dtype)            DTYPE="$2";            shift 2 ;;
        --kv-dtype)         KV_DTYPE="$2";         shift 2 ;;
        --device-group)     DEVICE_GROUP="$2";     shift 2 ;;
        --device-id)        DEVICE_ID="$2";        shift 2 ;;
        --lora-path)        LORA_PATH="$2";        shift 2 ;;
        --feature)          FEATURE="$2";          shift 2 ;;
        --parallel)         PARALLEL=1;            shift ;;
        --workers)          WORKERS="$2";          shift 2 ;;
        -v|--verbose)       VERBOSE=1;             shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASS=0
FAIL=0
SKIP=0
RESULTS=()

# Build pytest flags as an array to avoid word-splitting/quoting issues
PYTEST_FLAGS=(--tb=short -ra)
if [[ $VERBOSE -eq 1 ]]; then
    PYTEST_FLAGS+=(-v)
fi

# On-device flags (also as arrays to preserve argument boundaries)
DEVICE_FLAGS=()
if [[ -n "$MODEL_NAME" ]]; then
    DEVICE_FLAGS=(
        --model-name   "$MODEL_NAME"
        --seq-len      "$SEQ_LEN"
        --ctx-len      "$CTX_LEN"
        --decode-bsz   "$DECODE_BSZ"
        --dtype        "$DTYPE"
        --kv-dtype     "$KV_DTYPE"
        --device-group "$DEVICE_GROUP"
        --device-id    "$DEVICE_ID"
    )
fi

EMBED_FLAGS=()
if [[ -n "$EMBED_MODEL" ]]; then
    EMBED_FLAGS=(
        --model-name   "$EMBED_MODEL"
        --ctx-len      "$CTX_LEN"
        --decode-bsz   "$DECODE_BSZ"
        --device-group "$DEVICE_GROUP"
        --device-id    "$DEVICE_ID"
    )
elif [[ ${#DEVICE_FLAGS[@]} -gt 0 ]]; then
    EMBED_FLAGS=("${DEVICE_FLAGS[@]}")
fi

# LoRA flags — use lora_model_name (Mistral-7B) instead of model_name (TinyLlama)
# because the default adapter (predibase/gsm8k) is for Mistral-7B.
# The lora_path fixture auto-downloads predibase/gsm8k if --lora-path is not provided.
# Note: --model-name is intentionally NOT set here. The lora tests create their own
# LLM instances via _make_lora_llm() using --lora-model-name. The shared llm fixture
# is not used by lora tests (test_base_model_generates_output was removed since
# generic inference is already covered by generic/test_inference_correctness.py).
LORA_FLAGS=()
if [[ ${#DEVICE_FLAGS[@]} -gt 0 ]]; then
    LORA_FLAGS=(
        --lora-model-name "$LORA_MODEL_NAME"
        --seq-len         "$SEQ_LEN"
        --ctx-len         "$CTX_LEN"
        --decode-bsz      "$DECODE_BSZ"
        --dtype           "$DTYPE"
        --kv-dtype        "$KV_DTYPE"
        --device-group    "$DEVICE_GROUP"
        --device-id       "$DEVICE_ID"
    )
    # Only add --lora-path if explicitly provided; otherwise the fixture auto-downloads
    if [[ -n "$LORA_PATH" ]]; then
        LORA_FLAGS+=(--lora-path "$LORA_PATH")
    fi
fi

# ---------------------------------------------------------------------------
# Feature filter helper
#
# Returns 0 (true) if rel_path belongs to the requested FEATURE, 1 otherwise.
# Matching rule: rel_path starts with "FEATURE/"
#   e.g. --feature lora    matches  lora/test_lora_integration.py
#   e.g. --feature generic matches  generic/test_platform.py
# ---------------------------------------------------------------------------
feature_matches() {
    local rel_path="$1"
    [[ -z "$FEATURE" ]] && return 0          # no filter — always match
    [[ "$rel_path" == "${FEATURE}/"* ]] && return 0   # subdirectory match
    return 1
}

# ---------------------------------------------------------------------------
# Helper: run_suite NAME REL_PATH [EXTRA_ARGS...]
#   NAME      — display name for the suite
#   REL_PATH  — path relative to SCRIPT_DIR (file or directory)
#   EXTRA_ARGS — additional pytest args (e.g. device flags)
# ---------------------------------------------------------------------------
run_suite() {
    local name="$1"
    local rel_path="$2"
    shift 2
    local extra_args=("$@")

    local full_path="${SCRIPT_DIR}/${rel_path}"

    # Skip gracefully if the path does not exist
    if [[ ! -e "$full_path" ]]; then
        echo ""
        echo "  ⚠️  SKIPPED (not found): ${rel_path}"
        RESULTS+=("SKIP | ${name} | 0s")
        SKIP=$((SKIP + 1))
        return 0
    fi

    local start_time
    start_time=$(date +%s)

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: ${name}"
    echo "  Path:    ${rel_path}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    python3 -m pytest "$full_path" "${PYTEST_FLAGS[@]}" "${extra_args[@]}"
    local rc=$?

    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # rc=5 means pytest collected 0 tests — treat as pass
    if [[ $rc -eq 0 || $rc -eq 5 ]]; then
        echo "  ✅ PASSED in ${elapsed}s"
        RESULTS+=("PASS | ${name} | ${elapsed}s")
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAILED in ${elapsed}s (exit code ${rc})"
        RESULTS+=("FAIL | ${name} | ${elapsed}s")
        FAIL=$((FAIL + 1))
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Auto-discover all test files relative to SCRIPT_DIR, sorted
# ---------------------------------------------------------------------------
discover_test_files() {
    find "$SCRIPT_DIR" -name "test_*.py" -type f \
        | sed "s|^${SCRIPT_DIR}/||" \
        | sort
}

# ---------------------------------------------------------------------------
# Helper: array_to_json ARR...
#   Converts a bash array of strings to a JSON array string, for handing
#   flag arrays to collect_jobs.py.
# ---------------------------------------------------------------------------
array_to_json() {
    if [[ $# -eq 0 ]]; then
        echo "[]"
        return 0
    fi
    printf '%s\n' "$@" | jq -R . | jq -s -c .
}

# ---------------------------------------------------------------------------
# Run all tests (or the selected feature)
# On-device tests auto-skip when --model-name is not provided (via llm fixture)
# ---------------------------------------------------------------------------
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  Running vllm-qaic unit tests                                               ║"
if [[ -n "$FEATURE" ]]; then
echo "║  Feature filter: ${FEATURE}$(printf '%*s' $((60 - ${#FEATURE})) '')║"
fi
if [[ -n "$MODEL_NAME" ]]; then
echo "║  On-device tests ENABLED  (model: ${MODEL_NAME:0:44})$(printf '%*s' $((44 - ${#MODEL_NAME} > 0 ? 44 - ${#MODEL_NAME} : 0)) '')║"
else
echo "║  On-device tests will auto-skip (no --model-name provided)                  ║"
fi
if [[ $PARALLEL -eq 1 ]]; then
echo "║  Parallel mode: up to ${WORKERS} concurrent job(s)$(printf '%*s' $((37 - ${#WORKERS})) '')║"
fi
echo "╚══════════════════════════════════════════════════════════════════════════════╝"

if [[ $PARALLEL -eq 1 ]]; then
    # -------------------------------------------------------------------
    # Parallel dispatch: collect_jobs.py resolves the exact same per-file
    # flag routing as the sequential loop below (embed/lora/device/none),
    # then scheduler.py runs all jobs concurrently and prints each job's
    # full captured output as one atomic block as it finishes.
    # -------------------------------------------------------------------
    JOBS_FILE="$(mktemp /tmp/vllm_qaic_unit_jobs.XXXXXX.json)"
    OUTPUT_DIR="$(mktemp -d /tmp/vllm_qaic_unit_parallel.XXXXXX)"

    python3 "$SCRIPT_DIR/collect_jobs.py" \
        --output "$JOBS_FILE" \
        --feature "$FEATURE" \
        --device-flags "$(array_to_json "${DEVICE_FLAGS[@]}")" \
        --embed-flags "$(array_to_json "${EMBED_FLAGS[@]}")" \
        --lora-flags "$(array_to_json "${LORA_FLAGS[@]}")"
    COLLECT_RC=$?
    if [[ $COLLECT_RC -ne 0 ]]; then
        echo "  ❌ collect_jobs.py failed (exit code ${COLLECT_RC})"
        rm -f "$JOBS_FILE"
        exit 1
    fi

    python3 "$SCRIPT_DIR/scheduler.py" "$JOBS_FILE" \
        --script-dir "$SCRIPT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --workers "$WORKERS"
    SCHED_RC=$?

    rm -f "$JOBS_FILE"
    echo ""
    echo "  Per-job logs kept at: ${OUTPUT_DIR}"
    exit $SCHED_RC
fi

while IFS= read -r rel_path; do
    # Apply feature filter — skip suites that don't match
    feature_matches "$rel_path" || continue

    suite_name="${rel_path%.py}"

    # Route to the correct flags based on test type
    if [[ "$rel_path" == embedding/* ]] && [[ ${#EMBED_FLAGS[@]} -gt 0 ]]; then
        # Embedding tests use embed model (BAAI/bge-base-en-v1.5)
        run_suite "$suite_name" "$rel_path" "${EMBED_FLAGS[@]}"
    elif [[ "$rel_path" == lora/* ]] && [[ ${#LORA_FLAGS[@]} -gt 0 ]]; then
        # LoRA tests use lora model (Mistral-7B by default)
        run_suite "$suite_name" "$rel_path" "${LORA_FLAGS[@]}"
    elif [[ ${#DEVICE_FLAGS[@]} -gt 0 ]]; then
        # All other on-device tests use the generic model flags
        run_suite "$suite_name" "$rel_path" "${DEVICE_FLAGS[@]}"
    else
        # No device flags — run without on-device args (hardware tests auto-skip)
        run_suite "$suite_name" "$rel_path"
    fi
done < <(discover_test_files)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL + SKIP))
echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║  TEST SUMMARY                                                                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
for result in "${RESULTS[@]}"; do
    echo "  ${result}"
done
echo ""
if [[ -n "$FEATURE" ]]; then
    echo "  Feature: ${FEATURE}  |  Total suites: ${TOTAL}  |  Passed: ${PASS}  |  Failed: ${FAIL}  |  Skipped: ${SKIP}"
else
    echo "  Total suites: ${TOTAL}  |  Passed: ${PASS}  |  Failed: ${FAIL}  |  Skipped: ${SKIP}"
fi
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo "  ❌ Some test suites FAILED"
    exit 1
else
    echo "  ✅ All test suites PASSED"
    exit 0
fi