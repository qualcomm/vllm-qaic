#!/usr/bin/env python3
# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Collect the facts needed to draft vllm-qaic release notes.

Usage: release_context.py <repo> <prev-ref> <new-ref>

Prints, as markdown, everything the release-notes skill is allowed to use:
  1. the commits in prev..new (PR number, author, handle, subject),
  2. the version pins at both refs and what changed,
  3. the filtered +/- lines of the files that carry those pins.

Read-only: runs `git log` / `git show` / `git diff` only.
"""

import os
import subprocess
import sys

import regex as re

UTILITY = "scripts/utility.sh"
UTILITY_KEYS = [
    "VLLM_VERSION",
    "VLLM_QAIC_VERSION",
    "QEFF_BRANCH",
    "TORCH_VERSION_AOT",
    "TORCHVISION_VERSION_AOT",
    "TORCH_VERSION_PYT",
    "TORCHVISION_VERSION_PYT",
    "TORCHAUDIO_VERSION_PYT",
    "TORCH_QAIC_VERSION",
    "VLLM_QAIC_SDK_PATH",
]
REQ_FILES = [
    "requirements/vllm_dependency_aot.txt",
    "requirements/vllm_dependency_pyt.txt",
]
REQ_PKGS = ["transformers", "mistral_common", "numba", "compressed-tensors"]
PIN_FILES = [UTILITY, "pyproject.toml", "setup.py", *REQ_FILES]


def git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def show(repo, ref, path):
    return git(repo, "show", f"{ref}:{path}", check=False)


def utility_pins(text):
    pins = {}
    for k in UTILITY_KEYS:
        m = re.search(rf'^{k}="\$\{{{k}:-([^}}]*)\}}"', text, re.M)
        pins[k] = m.group(1) if m else None
    return pins


def other_pins(repo, ref):
    pins = {}
    m = re.search(r'return "([\d.]+)"', show(repo, ref, "setup.py"))
    pins["Apps SDK (setup.py get_qaic_sdk_version)"] = m.group(1) if m else None
    m = re.search(
        r'^requires-python\s*=\s*"([^"]*)"', show(repo, ref, "pyproject.toml"), re.M
    )
    pins["requires-python"] = m.group(1) if m else None
    m = re.search(r'"spd"\s*:\s*\[([^\]]*)\]', show(repo, ref, "setup.py"))
    pins["extra: spd"] = m.group(1).strip() if m else None
    for f in REQ_FILES:
        text = show(repo, ref, f)
        for p in REQ_PKGS:
            m = re.search(
                rf"^{re.escape(p)}(?:\[[^\]]*\])?\s*([=<>!~]=?\s*[^\s#]+)", text, re.M
            )
            pins[f"{f.split('/')[-1]}: {p}"] = m.group(1) if m else None
    return pins


def load_map():
    """Read references/handles.md -> {git author: (handle, display name, source)}."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "references", "handles.md"
    )
    people = {}
    if not os.path.exists(path):
        return people
    with open(path) as f:
        for line in f:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 4 and not cells[0].startswith(("Git author", "---")):
                people[cells[0]] = tuple(cells[1:])
    return people


def compare_ref(repo, ref):
    """Strip a leading remote name so the ref resolves on GitHub."""
    head, _, rest = ref.partition("/")
    remotes = git(repo, "remote").split()
    return rest if rest and head in remotes else ref


def handle(email):
    m = re.match(r"\d+\+([^@]+)@users\.noreply\.github\.com", email)
    return m.group(1) if m else None


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    repo, prev, new = sys.argv[1:]

    people = load_map()
    print(f"# Release context: {prev}..{new}\n")
    print(
        "Compare URL refs (remote prefix stripped): "
        f"`{compare_ref(repo, prev)}...{compare_ref(repo, new)}`\n"
    )
    print("## Commits (oldest first)\n")
    print(
        "| PR | Author | GitHub handle (noreply email, else references/handles.md, "
        "else unknown) | Subject |"
    )
    print("|---|---|---|---|")
    log = git(repo, "log", "--reverse", "--format=%an\t%ae\t%s", f"{prev}..{new}")
    for line in log.splitlines():
        name, email, subj = line.split("\t", 2)
        nums = re.findall(r"\(#(\d+)\)", subj)
        pr = nums[-1] if nums else "none"
        title = re.sub(r"(\s*\(#\d+\))+$", "", subj)
        h = handle(email) or (people.get(name) or (None,))[0] or "unknown"
        print(f"| {pr} | {name} | {h} | {title} |")

    print("\n## Contributors (distinct authors, with display names)\n")
    authors = dict.fromkeys(ln.split("\t")[0] for ln in log.splitlines())
    for a in authors:
        disp = (people.get(a) or (None, None))[1]
        print(f"- {disp or a}" + ("" if disp else "  (not in references/handles.md)"))

    print("\n## Version pins\n")
    old = {**utility_pins(show(repo, prev, UTILITY)), **other_pins(repo, prev)}
    cur = {**utility_pins(show(repo, new, UTILITY)), **other_pins(repo, new)}
    print(f"| Pin | {prev} | {new} | Changed |")
    print("|---|---|---|---|")
    for k in cur:
        o, n = old.get(k), cur[k]
        print(f"| {k} | {o} | {n} | {'yes' if o != n else ''} |")

    changed = [k for k in cur if old.get(k) != cur[k]]
    print("\n## Ready-made block (paste verbatim; every changed pin is listed)\n")
    print(f"Changes since {prev}:\n")
    for k in changed:
        print(f"- {k}: {old.get(k)} -> {cur[k]}")

    print("\n## Raw pin-file diff (+/- lines only)\n\n```diff")
    d = git(repo, "diff", f"{prev}..{new}", "--", *PIN_FILES)
    print(
        "\n".join(
            ln
            for ln in d.splitlines()
            if re.match(r"^[+-]", ln) and not ln.startswith(("+++", "---"))
        )
    )
    print("```")


if __name__ == "__main__":
    main()
