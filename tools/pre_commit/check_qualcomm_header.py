# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import sys
from enum import Enum


class HeaderStatus(Enum):
    """Qualcomm header status enumeration"""

    EMPTY = "empty"  # empty file (e.g. empty __init__.py)
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"  # Only has copyright line
    MISSING_COPYRIGHT = "missing_copyright"  # Only has license line
    MISSING_BOTH = "missing_both"  # Completely missing


RULE_LINE = "# " + "-" * 66
COPYRIGHT_LINE = (
    "# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries."
)
LICENSE_LINE = "# SPDX-License-Identifier: BSD-3-Clause-Clear"

FULL_HEADER = "\n".join([
    RULE_LINE,
    COPYRIGHT_LINE,
    LICENSE_LINE,
    RULE_LINE,
])


def check_header_status(file_path):
    """Check Qualcomm header status of the file"""
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()
        if not lines:
            # Empty file
            return HeaderStatus.EMPTY

        # Skip shebang line
        start_idx = 0
        if lines[0].startswith("#!"):
            start_idx = 1

        has_license = False
        has_copyright = False

        # Check all lines for the header markers (not just the first few)
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line == LICENSE_LINE:
                has_license = True
            elif line == COPYRIGHT_LINE:
                has_copyright = True

        if has_license and has_copyright:
            return HeaderStatus.COMPLETE
        elif has_license and not has_copyright:
            return HeaderStatus.MISSING_COPYRIGHT
        elif not has_license and has_copyright:
            return HeaderStatus.MISSING_LICENSE
        else:
            return HeaderStatus.MISSING_BOTH


def add_header(file_path, status):
    """Add or supplement the Qualcomm header based on status"""
    with open(file_path, "r+", encoding="UTF-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == HeaderStatus.MISSING_BOTH:
            # Completely missing, add the complete header
            if lines and lines[0].startswith("#!"):
                # Preserve shebang line
                file.write(lines[0])
                file.write(FULL_HEADER + "\n")
                file.writelines(lines[1:])
            else:
                file.write(FULL_HEADER + "\n")
                file.writelines(lines)

        elif status == HeaderStatus.MISSING_COPYRIGHT:
            # Only has license line, add copyright line before it
            for i, line in enumerate(lines):
                if line.strip() == LICENSE_LINE:
                    lines.insert(i, f"{COPYRIGHT_LINE}\n")
                    break
            file.writelines(lines)

        elif status == HeaderStatus.MISSING_LICENSE:
            # Only has copyright line, add license line after it
            for i, line in enumerate(lines):
                if line.strip() == COPYRIGHT_LINE:
                    lines.insert(i + 1, f"{LICENSE_LINE}\n")
                    break
            file.writelines(lines)


def main():
    """Main function"""
    files_missing_both = []
    files_missing_copyright = []
    files_missing_license = []

    for file_path in sys.argv[1:]:
        status = check_header_status(file_path)

        if status == HeaderStatus.MISSING_BOTH:
            files_missing_both.append(file_path)
        elif status == HeaderStatus.MISSING_COPYRIGHT:
            files_missing_copyright.append(file_path)
        elif status == HeaderStatus.MISSING_LICENSE:
            files_missing_license.append(file_path)
        else:
            continue

    all_files_to_fix = (
        files_missing_both + files_missing_copyright + files_missing_license
    )
    if all_files_to_fix:
        print("The following files are missing the Qualcomm header:")
        for file_path in files_missing_both:
            print(f"  {file_path}")
            add_header(file_path, HeaderStatus.MISSING_BOTH)
        for file_path in files_missing_copyright:
            print(f"  {file_path}")
            add_header(file_path, HeaderStatus.MISSING_COPYRIGHT)
        for file_path in files_missing_license:
            print(f"  {file_path}")
            add_header(file_path, HeaderStatus.MISSING_LICENSE)

    sys.exit(1 if all_files_to_fix else 0)


if __name__ == "__main__":
    main()
