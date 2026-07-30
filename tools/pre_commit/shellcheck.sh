#!/bin/bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Adapted from vllm/tools/pre_commit/shellcheck.sh
set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    # automatic local install if linux x86_64
    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    export PATH="$PATH:$(pwd)/shellcheck-${scversion}"
fi

# SC1091: shellcheck cannot follow sourced files (e.g. `source ./utility.sh`)
# at lint time, producing false positives; disable it repo-wide.
find . -path ./.git -prune -o -name "*.sh" -print0 | \
  xargs -0 sh -c "for f in \"\$@\"; do git check-ignore -q \"\$f\" || shellcheck -s bash --exclude=SC1091 \"\$f\"; done" --
