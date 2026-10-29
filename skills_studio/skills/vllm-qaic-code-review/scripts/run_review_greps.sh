#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Mechanical pre-pass for the vllm-qaic-code-review skill.
#
# Usage: run_review_greps.sh <diff-file> [<repo-root>]
#
# Prints candidate hits per section as `path:line: <added line>`, where `line`
# is the line number in the patched file. Each section header names the rule
# ID the hits are candidates for. A hit is a lead for a role agent to judge,
# not a finding: the agent still applies the rule, its carve-outs and the
# anchoring gate. Empty sections print nothing under the header.

set -u

DIFF="${1:-}"
REPO_ROOT="${2:-.}"

if [[ -z "$DIFF" || ! -f "$DIFF" ]]; then
    echo "usage: $0 <diff-file> [<repo-root>]" >&2
    exit 2
fi

ADDED="$(mktemp)"
FILES="$(mktemp)"
trap 'rm -f "$ADDED" "$FILES"' EXIT

# One record per added line: <path>\t<new-line-number>\t<text without the '+'>.
awk '
    /^\+\+\+ / { path = substr($0, 5); sub(/^b\//, "", path); next }
    /^@@ / {
        split($3, a, ","); line = substr(a[1], 2) + 0; next
    }
    /^\+/ { printf "%s\t%d\t%s\n", path, line, substr($0, 2); line++; next }
    /^ / { line++ }
' "$DIFF" > "$ADDED"

# Changed paths with their git status letter (A/M/D/R).
awk '
    /^diff --git / { path = $4; sub(/^b\//, "", path); status = "M" }
    /^new file mode/ { status = "A" }
    /^deleted file mode/ { status = "D" }
    /^rename to / { status = "R" }
    /^(index |Binary |--- |GIT binary)/ && path != "" { files[path] = status }
    END { for (p in files) print files[p] "\t" p }
' "$DIFF" | sort -k2 > "$FILES"

section() {
    printf "\n=== %s ===\n" "$1"
}

# hits <path-regex> <line-regex> [<exclude-line-regex>]
# Prints added lines whose path matches <path-regex> and text matches <line-regex>.
# Patterns go through the environment, not `awk -v`, which would strip their backslashes.
hits() {
    PR="$1" LR="$2" XR="${3:-}" awk -F'\t' '
        BEGIN { pr = ENVIRON["PR"]; lr = ENVIRON["LR"]; xr = ENVIRON["XR"] }
        $1 ~ pr && $3 ~ lr && (xr == "" || $3 !~ xr) { print $1 ":" $2 ": " $3 }
    ' "$ADDED"
}

PY='\.py$'
LIB='^vllm_qaic/.*\.py$'

section "Changed files (status, path)"
cat "$FILES"

# --- Routing signals ---

section "Mode branches added in shared code (routing: straddles both modes)"
hits "$LIB" '(is_aot|is_aot_inference|enforce_eager)'

# --- shared: patch / ops ---

section "patch.no_direct_upstream_edit — upstream vllm/ files touched"
awk -F'\t' '$2 ~ /^vllm\// { print $1 "\t" $2 }' "$FILES"

section "patch.last_resort_justified / patch.documented_why — new patch modules, patch/__init__.py edits"
awk -F'\t' '$1 == "A" && $2 ~ /^vllm_qaic\/patch\/patch_.*\.py$/ { print "new: " $2 }' "$FILES"
hits '^vllm_qaic/patch/__init__\.py$' '^[[:space:]]*(from|import)[[:space:]]'

section "patch.replace_not_append — assignments onto upstream modules"
hits "$LIB" '^[[:space:]]*(vllm|torch)\.[A-Za-z0-9_.]+[[:space:]]*=[^=]'

section "patch.mode_gated — patch modules (check for an is_aot guard)"
hits '^vllm_qaic/patch/.*\.py$' '^[[:space:]]*(vllm|torch)\.[A-Za-z0-9_.]+[[:space:]]*=[^=]'

section "ops.forward_oot_only / ops.register_oot"
hits "$LIB" '(def forward_static|def forward_native|def forward_cuda|register_oot)'

# --- shared: config / state ---

section "config.validate_keys — subscripted config dict reads"
hits "$LIB" '(override_qaic_config|additional_config|compile_config|qaic_config|cfg)\[["'"'"'][A-Za-z0-9_]+["'"'"']\]' '\][[:space:]]*=[^=]'

section "config.str_and_bool_forms — truthiness of config/env values"
hits "$LIB" '(if|and|or|not)[[:space:]]+[A-Za-z_.]*(config|cfg|env)[A-Za-z_.]*(\.get\(|\[)'

section "config.localize_to_check_and_update_config — vllm_config field writes"
hits "$LIB" '(vllm_config|parallel_config|model_config|scheduler_config|cache_config|speculative_config|compilation_config)\.[A-Za-z0-9_.]+[[:space:]]*=[^=]'

section "error.no_silent_except / pyt.bare_except"
hits "$PY" '^[[:space:]]*except[[:space:]]*(:|Exception[[:space:]]*:|BaseException[[:space:]]*:|\(Exception)'

section "concurrency.unbounded_queue_get — get() without timeout"
hits "$LIB" '\.get\([[:space:]]*(block[[:space:]]*=[[:space:]]*True[[:space:]]*)?\)' '(dict|config|os\.environ|kwargs|\.get\(["'"'"'])'

section "hygiene.no_nonblocking_cpu_copy"
hits "$LIB" 'non_blocking[[:space:]]*=[[:space:]]*True'

section "hygiene.dead_code — commented-out code, debugger calls"
hits "$PY" '^[[:space:]]*#+[[:space:]]*(import |from [A-Za-z_.]+ import |def |class |return |print\()'
hits "$PY" '(pdb\.set_trace|breakpoint\(\)|ipdb\.set_trace|IPython\.embed)'

section "change.justify_new_knob — new env vars, CLI args, additional_config keys"
hits "$LIB" '(add_argument\(|"VLLM_QAIC_[A-Z0-9_]+"[[:space:]]*:|additional_config\.get\()'

# --- shared: AGENTS.md ground rules ---

section "mode.lazy_package_import — module-scope QEfficient / torch_qaic imports"
hits "$LIB" '^(import|from)[[:space:]]+(QEfficient|torch_qaic)'

section "mode.lazy_package_import — indented imports (confirm inside try/except ImportError or a mode branch)"
hits "$LIB" '^[[:space:]]+(import|from)[[:space:]]+(QEfficient|torch_qaic)'

section "config.env_via_envs_py — env reads outside vllm_qaic/envs.py"
awk -F'\t' '
    $1 ~ /^vllm_qaic\// && $1 != "vllm_qaic/envs.py" && $3 ~ /os\.(getenv\(|environ\.get\(|environ\[)/ {
        print $1 ":" $2 ": " $3
    }
' "$ADDED"

section "hygiene.log_not_print — print() in library code"
hits "$LIB" '^[[:space:]]*print\('

section "hygiene.license_header — new non-Python sources missing the BSD-3-Clause-Clear header"
awk -F'\t' '$1 == "A" { print $2 }' "$FILES" | while read -r f; do
    case "$f" in
        *.sh|*.c|*.cc|*.cpp|*.h|*.hpp|*Dockerfile*) ;;
        *) continue ;;
    esac
    awk -F'\t' -v f="$f" '$1 == f' "$ADDED" | head -12 |
        grep -q 'SPDX-License-Identifier: BSD-3-Clause-Clear' || echo "MISSING: $f"
done

section "hygiene.no_committed_artifacts"
awk -F'\t' '
    $1 != "D" && $2 ~ /(\.onnx|\.qpc|\.safetensors|\.bin|\.pt|\.pth|\.gguf|\.log|\.pyc|\.swp|\.bak|~)$|__pycache__\/|scheduler_output\/|(^|\/)qpcs?\// {
        print $1 "\t" $2
    }
' "$FILES"

# --- shared: leaks (every file, including Markdown and skills_studio/) ---

section "leak.no_machine_specific_path — user homes, workspace trees, network/scratch mounts"
hits '.' '(/home/|/Users/|/usr[0-9]+/|/local/mnt/|/prj/|/net/|/scratch|/afs/|/mnt/[a-z0-9_-]+/[a-z0-9_-]+/)[A-Za-z0-9_.-]+' '(/path/to/|<[a-z_-]+>|\$\{?[A-Z_]+\}?/)'

section "leak.no_internal_reference — emails, intranet hosts, credentials"
hits '.' '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' '(Signed-off-by:|Co-Authored-By:|noreply@|example\.(com|org)|git@github\.com)'
hits '.' 'https?://[A-Za-z0-9.-]+\.(corp|internal|intranet|lan|local)(/|:|[[:space:]]|$)'
hits '.' '(api[_-]?key|secret|token|passwd|password)[[:space:]]*[:=][[:space:]]*["'"'"'][^"'"'"']{8,}'
hits '.' '(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)'

# --- shared: build version pins (reads the repo, not just the diff) ---

section "build.qaic_sdk_version_sync / build.torch_version_sync — values per source"
if ! awk -F'\t' '$2 ~ /^(setup\.py|scripts\/utility\.sh|docker\/Dockerfile\.(aot|pyt)|docs\/installation\.md)$/' "$FILES" | grep -q .; then
    echo "(no version-pin source touched — both rules are na)"
else
    for name in VLLM_QAIC_VERSION TORCH_VERSION_AOT TORCHVISION_VERSION_AOT \
        TORCH_VERSION_PYT TORCHVISION_VERSION_PYT TORCHAUDIO_VERSION_PYT; do
        {
            if [[ "$name" == VLLM_QAIC_VERSION && -f "$REPO_ROOT/setup.py" ]]; then
                awk '/def get_qaic_sdk_version/ { f = 1 } f && /return/ {
                    gsub(/.*return[[:space:]]*["'"'"']|["'"'"'].*/, ""); print "setup.py:get_qaic_sdk_version\t" $0; exit }' \
                    "$REPO_ROOT/setup.py"
            fi
            for src in scripts/utility.sh docker/Dockerfile.aot docker/Dockerfile.pyt; do
                [[ -f "$REPO_ROOT/$src" ]] || continue
                grep -nE "^(ARG[[:space:]]+)?${name}=" "$REPO_ROOT/$src" |
                    sed -E "s/^([0-9]+):.*${name}=\"?(\\\$\{${name}:-)?([^\"}]*).*/${src//\//\\/}:\1\t\3/"
            done
            if [[ -f "$REPO_ROOT/docs/installation.md" ]]; then
                grep -nE "^\| \`${name}\` \|" "$REPO_ROOT/docs/installation.md" |
                    sed -E "s/^([0-9]+):\| \`${name}\` \| \`([^\`]*)\`.*/docs\/installation.md:\1\t\2/"
            fi
        } > "$ADDED.ver"
        [[ -s "$ADDED.ver" ]] || continue
        distinct=$(cut -f2 "$ADDED.ver" | sort -u | wc -l)
        if [[ "$distinct" -gt 1 ]]; then
            echo "$name: MISMATCH ($distinct distinct values)"
        else
            echo "$name: consistent"
        fi
        sed 's/^/    /' "$ADDED.ver"
    done
    rm -f "$ADDED.ver"
fi

# --- aot ---

section "aot.ccl_defaults / aot.mdp_stages_consistency / aot.specialization_covers_shapes — config keys"
hits "$PY" '(comp_ctx_lengths_prefill|comp_ctx_lengths_decode|ccl_enabled|mdp_num_partitions|\[["'"'"']stages["'"'"']\]|prefill_seq_len|vision_size)'

section "aot.session_input_guard — session input feeds"
hits "$LIB" '(session\.(set_buffers|run|input_names)|input_names)'

section "aot.qpc_dict_keys — QPC path extraction"
hits "$LIB" 'qpc[A-Za-z_]*[[:space:]]*=.*\['

section "disagg.device_group_validation / disagg.proc_lifecycle"
hits "$PY" '(device_group|prefill_device|decode_device|encode_device|subprocess\.|multiprocessing\.|Process\(|resource_tracker)'

section "spd.aot_only — speculative decoding paths"
hits "$LIB" '(speculative_config|spec_decode|num_speculative_tokens)'

# --- pyt ---

section "pyt.per_device_seed — global seeding"
hits "$PY" '(torch\.manual_seed|torch\.random\.manual_seed|np\.random\.seed|random\.seed)\('

section "pyt.eager_only_accessor"
hits "$PY" '(get_num_cores|get_num_hvx_threads)\('

section "pyt.register_fake_meta — new qaic:: custom ops"
hits "$PY" '(torch\.library\.custom_op|torch\.library\.define|register_fake)\('

section "pyt.forbidden_import — bare torch_qaic imports"
hits "$PY" '^[[:space:]]*(import[[:space:]]+torch_qaic|from[[:space:]]+torch_qaic[[:space:].])'

section "pyt.torch_qaic_install — hard-coded torch_qaic wheel paths"
hits '\.(py|sh)$|Dockerfile' '/opt/qti-aic/integrations'

section "pyt.test_location — new eager tests / benchmarks outside an eager/ test path"
awk -F'\t' '$1 == "A" && $2 ~ /^tests\// && $2 !~ /\/eager\// && $2 ~ /(eager|custom_op|kernel|bench)/ { print $2 }' "$FILES"

section "pyt.contiguous_operands / pyt.avoid_format_conversion"
hits "$LIB" '(\.contiguous\(|crouton|Crouton|to_flat|from_flat)'

# --- size signals for the main thread's shape verdict ---

section "Added lines per file (top 25)"
cut -f1 "$ADDED" | sort | uniq -c | sort -rn | head -25
