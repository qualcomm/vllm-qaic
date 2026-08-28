# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import argparse
from collections import defaultdict

import regex as re


def extract_fallback_ops(file_path):
    op_pattern = re.compile(r"CPU fallback op:\s*([^\s,]+),\s*execution started")
    time_pattern = re.compile(
        r"CPU fallback op:\s*([^\s,]+),\s*time elapsed \(us\):\s*(\d+)"
    )
    fallback_ops_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"tc": 0, "tt": 0}
    )

    with open(file_path) as f:
        for line in f:
            ops_identified = op_pattern.search(line)
            if ops_identified:
                op = ops_identified.group(1)
                fallback_ops_stats[op]["tc"] += 1

            sec_identified = time_pattern.search(line)
            if sec_identified:
                op = sec_identified.group(1)
                time = int(sec_identified.group(2))
                fallback_ops_stats[op]["tt"] += time

    print(fallback_ops_stats)
    return fallback_ops_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", type=str, required=True)
    args = parser.parse_args()
    extract_fallback_ops(args.file_name)
