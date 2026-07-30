# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
#
# Adapted from vllm/tools/pre_commit/mypy.py
"""
Run mypy on changed files.

This script is designed to be used as a pre-commit hook. It runs mypy
on files that have been changed. It groups files into different mypy calls
based on their directory to avoid import following issues.

Usage:
    python tools/pre_commit/mypy.py <python_version> <changed_files...>

Args:
    python_version: Python version to use (e.g., "3.10") or "local" to use
        the local Python version.
    changed_files: List of changed files to check.
"""

import subprocess
import sys

import regex as re

# Top-level directories whose files are type-checked. Add new source trees
# here as the plugin grows.
INCLUDE_ROOTS = [
    "vllm_qaic",
    "examples",
]

# Directories checked in a separate mypy call with `--follow-imports skip`,
# to avoid mypy chasing imports into trees that are not (yet) fully typed.
# After a directory is made type-clean, move it out of here so it is checked
# with the default (silent) follow-imports behavior.
SEPARATE_GROUPS = [
    # "tests",
]

# Directories skipped entirely (e.g. vendored or known-broken code).
# Extend as needed, mirroring vLLM's EXCLUDE list.
EXCLUDE: list[str] = []


def group_files(changed_files: list[str]) -> dict[str, list[str]]:
    """
    Group changed files into different mypy calls.

    Args:
        changed_files: List of changed files.

    Returns:
        A dictionary mapping file group names to lists of changed files.
    """
    include_pattern = re.compile(f"^({'|'.join(INCLUDE_ROOTS)})/.*")
    exclude_pattern = (
        re.compile(f"^({'|'.join(EXCLUDE)}).*") if EXCLUDE else None
    )
    file_groups: dict[str, list[str]] = {"": []}
    file_groups.update({k: [] for k in SEPARATE_GROUPS})
    for changed_file in changed_files:
        # Skip files which should be ignored completely
        if exclude_pattern is not None and exclude_pattern.match(changed_file):
            continue
        # Group files by mypy call
        for directory in SEPARATE_GROUPS:
            if re.match(f"^{directory}/.*", changed_file):
                file_groups[directory].append(changed_file)
                break
        else:
            if include_pattern.match(changed_file):
                file_groups[""].append(changed_file)
    return file_groups


def mypy(
    targets: list[str],
    python_version: str | None,
    follow_imports: str | None,
    file_group: str,
) -> int:
    """
    Run mypy on the given targets.

    Args:
        targets: List of files or directories to check.
        python_version: Python version to use (e.g., "3.10") or None to use
            the default mypy version.
        follow_imports: Value for the --follow-imports option or None to use
            the default mypy behavior.
        file_group: The file group name for logging purposes.

    Returns:
        The return code from mypy.
    """
    args = ["mypy"]
    if python_version is not None:
        args += ["--python-version", python_version]
    if follow_imports is not None:
        args += ["--follow-imports", follow_imports]
    print(f"$ {' '.join(args)} {file_group}")
    return subprocess.run(args + targets, check=False).returncode


def main():
    python_version = sys.argv[1]
    file_groups = group_files(sys.argv[2:])

    if python_version == "local":
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    returncode = 0
    for file_group, changed_files in file_groups.items():
        follow_imports = None if file_group == "" else "skip"
        if changed_files:
            returncode |= mypy(
                changed_files, python_version, follow_imports, file_group
            )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
