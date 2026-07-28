#!/usr/bin/env python3

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Default test list (optional)
TESTS = []


def discover_all():
    return sorted(
        f
        for f in os.listdir(HERE)
        if f.startswith("test_") and f.endswith(".py")
    )


def run_one(test_file):
    path = os.path.join(HERE, test_file)

    if not os.path.exists(path):
        return "FAIL"

    proc = subprocess.run(
        [sys.executable, path],
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # A test PASSES only when the kernel actually ran AND matched its PyTorch
    # reference. Every test file prints a line starting with "SUCCESS" in that
    # case and a line starting with "FAILURE" otherwise. We rely on that marker
    # rather than the exit code, because many test files call main() without
    # sys.exit(...) and therefore always return 0 even when the reference
    # comparison fails.
    output = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""

    saw_success = False
    saw_failure = False
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("SUCCESS"):
            saw_success = True
        elif stripped.startswith("FAILURE"):
            saw_failure = True

    # A negative return code means the process was killed by a signal
    # (segfault / abort during Triton->QAIC compile or execution) -> FAIL.
    crashed = proc.returncode < 0

    if saw_success and not saw_failure and not crashed:
        return "PASS"
    return "FAIL"


def main(argv):
    if argv == ["--all"]:
        tests = discover_all()
    elif argv:
        tests = argv
    else:
        tests = TESTS

    if not tests:
        print("No tests found.")
        return 1

    total = len(tests)
    name_width = max(len(t) for t in tests)

    passed = []
    failed = []

    print(f"Running {total} kernel test(s)\n")

    for idx, test_file in enumerate(tests, start=1):
        status = run_one(test_file)

        print(
            f"[{idx}/{total}] "
            f"{test_file:<{name_width}} "
            f"{status}"
        )

        if status == "PASS":
            passed.append(test_file)
        else:
            failed.append(test_file)

    print("\n" + "=" * 80)

    print(f"\nPASSED ({len(passed)}):")
    for test in passed:
        print(f"  {test}")

    print(f"\nFAILED ({len(failed)}):")
    for test in failed:
        print(f"  {test}")

    print(f"\n{len(passed)}/{total} kernels passed")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))