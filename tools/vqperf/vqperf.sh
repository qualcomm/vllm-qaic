#!/usr/bin/env bash
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Run vqperf without installing it.
#
# vqperf drives *other* environments and must never share an interpreter with
# the environments it measures, so it runs from whatever python is on PATH and
# depends on the standard library only.
#
# Usage:
#   ./tools/vqperf/vqperf run --base-sha ... --new-sha ... --configs ...

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" -c 'import sys; sys.path.insert(0, sys.argv.pop(1)); from vqperf.cli import main; sys.exit(main())' "${HERE}" "$@"
