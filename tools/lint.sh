#!/usr/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Run all lint / format checks for vllm-qaic.
# Tool configuration is read from pyproject.toml in the project root.
#
# Usage:
#   ./tools/lint.sh [CI] [PYTHON_VERSION] [--fix]
#
#   CI             0 (default) or 1 — when 1, abort on first failure (set -e)
#   PYTHON_VERSION Python version passed to mypy (default: 3.10)
#   --fix          Auto-fix isort, yapf and ruff violations in place.
#                  mypy is always check-only.
#
# Examples:
#   ./tools/lint.sh              # check mode, non-CI
#   ./tools/lint.sh 1            # check mode, CI (abort on first failure)
#   ./tools/lint.sh 0 3.11 --fix # fix mode, Python 3.11

CI=${1:-0}
PYTHON_VERSION=${2:-3.10}
FIX=0

for arg in "$@"; do
    if [ "$arg" = "--fix" ]; then
        FIX=1
    fi
done

if [ "$CI" -eq 1 ]; then
    set -e
fi

# Resolve the project root (one level above this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PASS=0
FAIL=0

run_check() {
    local name="$1"
    shift
    echo ""
    echo "------------------------------------------------------------"
    echo "  $name"
    echo "  $ $*"
    echo "------------------------------------------------------------"
    if "$@"; then
        echo "  PASSED: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAILED: $name"
        FAIL=$((FAIL + 1))
    fi
}

# -------------------------------------------------------------------------
# 1. isort — import sorting
#    [tool.isort] in pyproject.toml:
#      skip_glob = ["examples/*"]
#      use_parentheses = true
#      skip_gitignore = true
# -------------------------------------------------------------------------
if [ "$FIX" -eq 1 ]; then
    run_check "isort (fix)" \
        isort --settings-path "${ROOT_DIR}/pyproject.toml" .
else
    run_check "isort (check)" \
        isort --settings-path "${ROOT_DIR}/pyproject.toml" --check-only --diff .
fi

# -------------------------------------------------------------------------
# 2. yapf — code formatting
#    [tool.yapfignore] in pyproject.toml:
#      ignore_patterns = ["build/**", "examples/**"]
# -------------------------------------------------------------------------
if [ "$FIX" -eq 1 ]; then
    run_check "yapf (fix)" \
        yapf --in-place --recursive \
             --exclude build \
             --exclude examples \
             .
else
    run_check "yapf (check)" \
        yapf --diff --recursive \
             --exclude build \
             --exclude examples \
             .
fi

# -------------------------------------------------------------------------
# 3. ruff — linting
#    [tool.ruff.lint] in pyproject.toml:
#      select = [E, F, UP, B, SIM, G]
#      ignore = [F405, F403, E731, B007, UP032]
# -------------------------------------------------------------------------
if [ "$FIX" -eq 1 ]; then
    run_check "ruff lint (fix)" \
        ruff check --fix --config "${ROOT_DIR}/pyproject.toml" .
    run_check "ruff format (fix)" \
        ruff format --config "${ROOT_DIR}/pyproject.toml" .
else
    run_check "ruff lint (check)" \
        ruff check --config "${ROOT_DIR}/pyproject.toml" .
    run_check "ruff format (check)" \
        ruff format --check --config "${ROOT_DIR}/pyproject.toml" .
fi

# -------------------------------------------------------------------------
# 4. mypy — static type checking (always check-only)
#    [tool.mypy] in pyproject.toml:
#      ignore_missing_imports = true
#      explicit_package_bases = true
#      check_untyped_defs = true
#      follow_imports = "silent"
# -------------------------------------------------------------------------
run_mypy() {
    echo "Running mypy on $1"
    mypy --config-file "${ROOT_DIR}/pyproject.toml" \
         --python-version "${PYTHON_VERSION}" \
         "$@"
}

run_check "mypy vllm_qaic" run_mypy vllm_qaic
run_check "mypy examples"  run_mypy examples

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Lint summary: ${PASS} passed, ${FAIL} failed"
echo "============================================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
