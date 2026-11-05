#!/usr/bin/env python3
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Adapted from vllm/tools/pre_commit/check_forbidden_imports.py
import sys
from dataclasses import dataclass, field

import regex as re


@dataclass
class ForbiddenImport:
    pattern: str
    tip: str
    allowed_pattern: re.Pattern = re.compile(r"^$")  # matches nothing by default
    allowed_files: set[str] = field(default_factory=set)


CHECK_IMPORTS = {
    "pickle/cloudpickle": ForbiddenImport(
        pattern=(
            r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
            r"|from\s+(pickle|cloudpickle)\s+import\b)"
        ),
        tip=(
            "Avoid using pickle or cloudpickle or add this file to "
            "tools/pre_commit/check_forbidden_imports.py."
        ),
    ),
    "base64": ForbiddenImport(
        pattern=r"^\s*(?:import\s+base64(?:$|\s|,)|from\s+base64\s+import)",
        tip=(
            "Replace 'import base64' with 'import pybase64' "
            "or 'import pybase64 as base64'."
        ),
        allowed_pattern=re.compile(r"^\s*import\s+pybase64(\s*|\s+as\s+base64\s*)$"),
    ),
    "re": ForbiddenImport(
        pattern=r"^\s*(?:import\s+re(?:$|\s|,)|from\s+re\s+import)",
        tip="Replace 'import re' with 'import regex as re' or 'import regex'.",
        allowed_pattern=re.compile(r"^\s*import\s+regex(\s*|\s+as\s+re\s*)$"),
        allowed_files={"setup.py"},
    ),
}


def check_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return_code = 0
    # Check all patterns in the whole file
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        # Skip files that are allowed for this import
        if path in forbidden_import.allowed_files:
            continue
        # Search for forbidden imports
        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            # Check if it's allowed
            if forbidden_import.allowed_pattern.match(match.group()):
                continue
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                f"Found forbidden import: {import_name}. {forbidden_import.tip}"
            )
            return_code = 1
    return return_code


def main():
    returncode = 0
    for path in sys.argv[1:]:
        returncode |= check_file(path)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
