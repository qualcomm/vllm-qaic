#!/usr/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

CI=${1:-0}
PYTHON_VERSION=${2:-3.10}

if [ "$CI" -eq 1 ]; then
    set -e
fi

run_mypy() {
    echo "Running mypy on $1"
    mypy --ignore-missing-import --follow-imports silent --python-version "${PYTHON_VERSION}" "$@"
}

run_mypy vllm_qaic
run_mypy examples
