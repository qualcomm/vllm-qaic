#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

#################### ci_fallback_ops.sh ##############################
# CI script for collecting fallback ops across model types.
#
# Usage:
#   bash ci_fallback_ops.sh [OPTIONS]
#
# Options:
#   --type       llm|vlm|all    Model type to run (default: all)
#   --model      <model_id>     Run a single model directly (requires --type llm|vlm)
#   --priority   P0|P1|P2       Priority tier filter (default: all)
#   --family     <name>         Run only models matching this family name
#   --tp-size    <N>            Default tensor parallel size (default: 4). A model
#                               with a pinned tp_size in model_configs_<type>.py
#                               overrides this, and its log is named with the TP it
#                               actually ran at.
#   --logs-dir   <dir>          Log output directory (default: ./ci_logs)
#   --models-dir <dir>          Directory containing scraped JSON files
#   --ref        <git_ref>      vLLM ref whose model list to use, i.e.
#                               <type>_models_<ref>.json as written by
#                               scrape_models.py (default: newest found)
#   --delete-hf-checkpoint      Delete each model's cached HF checkpoint once its
#                               run finishes, to bound disk use across a sweep
#                               (default: keep checkpoints)
#
# Examples:
#   bash ci_fallback_ops.sh --type llm --priority P0
#   bash ci_fallback_ops.sh --type vlm --family qwen --tp-size 4
#   bash ci_fallback_ops.sh --type llm --model allenai/OLMo-1B-hf
#   bash ci_fallback_ops.sh --type vlm --model Qwen/Qwen-VL
#   bash ci_fallback_ops.sh --type llm --delete-hf-checkpoint
#   bash ci_fallback_ops.sh --type llm --ref v0.23.0
#
# To add a new model type (e.g. embedding):
#   1. Add entry to RUNNER_MAP, JSON_PREFIX, RUNNER_EXTRA_ARGS below
#   2. Add run_<type>.py in the eager/ dir
#   3. Run scrape_models.py with the new type and store its JSON
######################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGER_DIR="${SCRIPT_DIR}/eager"

# User-configurable defaults
TYPE="all"
MODEL=""
PRIORITY=""
FAMILY=""
TP_SIZE="4"
QAIC_DEBUG="1"
LOGS_DIR="${PWD}/ci_logs"
MODELS_DIR="${EAGER_DIR}"
PRIORITY_CFG="${EAGER_DIR}/priority_config.json"
# Empty means "auto-discover": pick the newest <type>_models_<ref>.json present.
REF=""
# Off by default: a sweep must never evict a checkpoint unless explicitly asked,
# since the HF cache is usually shared between engineers.
DELETE_HF_CHECKPOINT="0"
SLEEP_BETWEEN=10

# QAIC runtime environment
export QAIC_FORCE_PLATFORM_QCCL=1
export QAIC_QCCL_ALGO=tree
export QAIC_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model type registry 
# To support a new model type:
#   1. Add its name to ALL_TYPES (controls run order for --type all)
#   2. Add one entry to each of RUNNER_MAP, JSON_PREFIX, RUNNER_EXTRA_ARGS
#   3. Create the corresponding run_<type>.py in eager/
ALL_TYPES=("llm" "vlm")

#python scripts
declare -A RUNNER_MAP=(
    ["llm"]="run_llms.py"
    ["vlm"]="run_vlms.py"
    # ["embedding"]="run_embeddings.py"
    # ["spd"]="run_spd.py"
)
#model list
declare -A JSON_PREFIX=(
    ["llm"]="llm_models"
    ["vlm"]="vlm_models"
    # ["embedding"]="embedding_models"
    # ["spd"]="spd_models"
)
#extra args for runner script
declare -A RUNNER_EXTRA_ARGS=(
    ["llm"]=""
    ["vlm"]="--model-impl vllm"
    # ["embedding"]=""
    # ["spd"]=""
)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)        TYPE="$2";        shift 2 ;;
        --model)       MODEL="$2";       shift 2 ;;
        --priority)    PRIORITY="$2";    shift 2 ;;
        --family)      FAMILY="$2";      shift 2 ;;
        --tp-size)     TP_SIZE="$2";     shift 2 ;;
        --logs-dir)    LOGS_DIR="$2";    shift 2 ;;
        --models-dir)  MODELS_DIR="$2";  shift 2 ;;
        --ref)         REF="$2";         shift 2 ;;
        # Boolean flag: takes no value, so shift 1
        --delete-hf-checkpoint) DELETE_HF_CHECKPOINT=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# --model requires a specific type (not "all")
if [ -n "$MODEL" ] && [ "$TYPE" = "all" ]; then
    echo "Error: --model requires --type llm|vlm (not 'all')"
    exit 1
fi

mkdir -p "${LOGS_DIR}"

# Python helper: filter model list from JSON
get_models() {
    local json_file="$1"
    python - "$json_file" "$PRIORITY_CFG" "$PRIORITY" "$FAMILY" << 'PYEOF'
import json, sys

json_file, priority_cfg_file, priority_filter, family_filter = sys.argv[1:]

with open(json_file) as f:
    models = json.load(f)

if priority_filter:
    with open(priority_cfg_file) as f:
        cfg = json.load(f)
    p0 = cfg.get("P0", [])
    p1 = cfg.get("P1", "*")

    def get_priority(family):
        if any(p in family for p in p0):
            return "P0"
        if p1 == "*" or (isinstance(p1, list) and any(p in family for p in p1)):
            return "P1"
        return "P2"

    models = [m for m in models if get_priority(m.get("family", "")) == priority_filter]

if family_filter:
    models = [m for m in models if family_filter in m.get("family", "")]

for m in models:
    print(m["model"])
PYEOF
}

# Mirror scrape_models.py's ref_suffix(): a ref may contain '/'
# (e.g. 'release/1.2'), which is not filename-safe.
ref_suffix() {
    echo "${1//\//_}"
}

# Locate the scraped model list for one type, setting RESOLVED_JSON/RESOLVED_REF.
# scrape_models.py writes one list per vLLM ref -- <prefix>_<ref>.json, e.g.
# llm_models_v0.23.0.json -- and several refs can sit side by side in MODELS_DIR,
# so a bare <prefix>.json lookup finds nothing.
#   --ref given : use exactly that ref's list, no guessing.
#   no --ref    : use the newest list found, and say which, because silently
#                 running a stale model list is worse than being noisy.
# Returns non-zero (with a message) when the type has no list to run.
resolve_json() {
    local model_type="$1"
    local prefix="${JSON_PREFIX[$model_type]}"
    RESOLVED_JSON=""
    RESOLVED_REF=""

    if [ -n "$REF" ]; then
        local candidate="${MODELS_DIR}/${prefix}_$(ref_suffix "$REF").json"
        if [ ! -f "$candidate" ]; then
            echo "Warning: model list not found: ${candidate} — skipping ${model_type}"
            echo "         create it with: python ${EAGER_DIR}/scrape_models.py --ref ${REF} --type ${model_type} --output-dir ${MODELS_DIR}"
            return 1
        fi
        RESOLVED_JSON="$candidate"
        RESOLVED_REF="$REF"
        echo "Using ${model_type} list: ${RESOLVED_JSON##*/} (ref ${RESOLVED_REF})"
        return 0
    fi

    local -a matches=()
    local f
    for f in "${MODELS_DIR}/${prefix}_"*.json; do
        [ -f "$f" ] && matches+=("$f")
    done

    if [ ${#matches[@]} -eq 0 ]; then
        echo "Warning: no ${prefix}_<ref>.json in ${MODELS_DIR} — skipping ${model_type}"
        echo "         create one with: python ${EAGER_DIR}/scrape_models.py --type ${model_type} --output-dir ${MODELS_DIR}"
        return 1
    fi

    # -V sorts version-ish names naturally, so v0.23.0 beats v0.9.1
    RESOLVED_JSON=$(printf '%s\n' "${matches[@]}" | sort -V | tail -1)
    # <prefix>_<ref>.json -> <ref>
    local base="${RESOLVED_JSON##*/}"
    base="${base%.json}"
    RESOLVED_REF="${base#${prefix}_}"

    if [ ${#matches[@]} -gt 1 ]; then
        echo "Note: ${#matches[@]} ${prefix} lists in ${MODELS_DIR}: ${matches[*]##*/}"
    fi
    echo "Using ${model_type} list: ${RESOLVED_JSON##*/} (ref ${RESOLVED_REF})"
    return 0
}

# Resolve the TP a model will actually run with.
# Falls back to the sweep default, with a warning, when the type has no
# model_configs_<type>.py, so a newly added model type still runs.
effective_tp() {
    local model_type="$1"
    local model_name="$2"
    local resolved

    resolved=$(python - "$EAGER_DIR" "$model_type" "$model_name" "$TP_SIZE" <<'PYEOF'
import importlib
import sys

eager_dir, model_type, model_name, default_tp = sys.argv[1:]
sys.path.insert(0, eager_dir)
mod = importlib.import_module(f"model_configs_{model_type}")
# Mirror run_llms.py, which un-escapes '--' before looking up the config.
print(mod.get_tp_size(model_name.replace("--", "/"), int(default_tp)))
PYEOF
    ) || resolved=""

    if [[ ! "$resolved" =~ ^[0-9]+$ ]]; then
        echo "Warning: could not resolve effective TP for ${model_name} (${model_type})," \
             "falling back to --tp-size ${TP_SIZE}" >&2
        resolved="$TP_SIZE"
    fi
    echo "$resolved"
}

# Run one model
run_model() {
    local model_type="$1"
    local model_name="$2"
    local runner="${RUNNER_MAP[$model_type]}"
    local extra_args="${RUNNER_EXTRA_ARGS[$model_type]}"

    local m_name="${model_name//\//_}"
    m_name="${m_name// /_}"

    # Set or clear QAIC debug env vars before running the model
    if [ "$QAIC_DEBUG" -eq 1 ]; then
        source "${EAGER_DIR}/export_qaic_debug.sh"
    else
        source "${EAGER_DIR}/unset_qaic_debug.sh"
    fi

    # TP the runner will actually use. May differ from TP_SIZE when
    # model_configs_<type>.py pins tp_size for this model.
    local run_tp
    run_tp="$(effective_tp "$model_type" "$model_name")"
    if [ "$run_tp" != "$TP_SIZE" ]; then
        echo "Note: ${model_name} pins TP=${run_tp} (sweep default ${TP_SIZE})"
    fi

    # Build log file paths: log_<debug>_<model>_tp<N>.log and parse_<debug>_<model>_tp<N>.log
    # <N> is the effective TP, which is what parse_logs_to_excel.py reports.
    local logname="${LOGS_DIR}/log_${QAIC_DEBUG}_${m_name}_tp${run_tp}.log"
    local parse_logname="${LOGS_DIR}/parse_${QAIC_DEBUG}_${m_name}_tp${run_tp}.log"

    local cleanup_flag=()
    if [ "$DELETE_HF_CHECKPOINT" -eq 1 ]; then
        cleanup_flag=(--delete-hf-checkpoint)
    fi

    logsave "${logname}" python "${EAGER_DIR}/${runner}" \
        --model-name "${model_name}" \
        --tp-size "${TP_SIZE}" \
        ${extra_args} \
        "${cleanup_flag[@]}"

    # Parse the log to extract fallback ops and write to parse log
    log_path="${logname}"
    logsave "${parse_logname}" python "${EAGER_DIR}/fallback_parser.py" \
        --file-name "${log_path}"

    echo "sleeping for ${SLEEP_BETWEEN}s.."
    sleep "${SLEEP_BETWEEN}"
    echo "awakening.."
}

# Determine which types to run
# Expand "all" to the ordered list in ALL_TYPES; otherwise use the single type given
if [ "$TYPE" = "all" ]; then
    TYPES=("${ALL_TYPES[@]}")
else
    TYPES=("$TYPE")
fi

# Main loop
# For each type: find its ref's model list, filter by priority/family, run each one
# EXCEL_REF is what gets handed to parse_logs_to_excel.py: a single ref when every
# type resolved to the same one, empty (= merge whatever is found) otherwise.
EXCEL_REF="$REF"
EXCEL_REF_SET=0

for model_type in "${TYPES[@]}"; do

    # Single-model mode: bypass the model list entirely and run the given model
    if [ -n "$MODEL" ]; then
        run_model "$model_type" "$MODEL"
        continue
    fi

    # Locate <type>_models_<ref>.json; skips this type if it has no list
    resolve_json "$model_type" || continue
    json_file="$RESOLVED_JSON"

    # Remember the ref actually used, so the Excel step reads the same lists
    if [ -z "$REF" ] && [ -n "$RESOLVED_REF" ]; then
        if [ "$EXCEL_REF_SET" -eq 0 ]; then
            EXCEL_REF="$RESOLVED_REF"
            EXCEL_REF_SET=1
        elif [ "$EXCEL_REF" != "$RESOLVED_REF" ]; then
            EXCEL_REF=""
        fi
    fi

    while IFS= read -r model_name; do
        [ -z "$model_name" ] && continue
        run_model "$model_type" "$model_name"
    done < <(get_models "$json_file")
done

# Aggregate all parse logs into a per-type Excel sheet once all models are done
python "${EAGER_DIR}/parse_logs_to_excel.py" \
    --log-dir "${LOGS_DIR}" \
    --models-dir "${MODELS_DIR}" \
    ${EXCEL_REF:+--ref "${EXCEL_REF}"}
