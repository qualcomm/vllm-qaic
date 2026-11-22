# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Allow ``python -m vqperf``."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
