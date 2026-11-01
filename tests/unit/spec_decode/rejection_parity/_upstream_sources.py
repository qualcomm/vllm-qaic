# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Normalized source hashes of the upstream rejection-sampler surface (tier C).

Sources are extracted with ``ast`` from the *file* of
``vllm.v1.sample.rejection_sampler`` rather than from live objects, because
``vllm_qaic.patch`` replaces ``RejectionSampler.forward`` and the Numba
backend replaces the kernel globals at runtime.  Each definition segment
spans its first decorator through its last line, so e.g. a change of
``@triton.jit(do_not_specialize=...)`` is detected.  Normalization: strip
trailing whitespace per line, ``textwrap.dedent``, strip leading/trailing
blank lines.  No triton or torch import is needed beyond locating the file.

Run this file directly to refresh the recorded hashes after mandatory Tier A
parity has passed::

    .venv_aot/bin/python tests/unit/spec_decode/rejection_parity/_upstream_sources.py
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path
from typing import Any

UPSTREAM_MODULE = "vllm.v1.sample.rejection_sampler"
HASHES_PATH = Path(__file__).resolve().parent / "fixtures" / "upstream_hashes.json"

KERNELS = (
    "expand_kernel",
    "rejection_greedy_sample_kernel",
    "rejection_random_sample_kernel",
    "sample_recovered_tokens_kernel",
)
CALL_SITES = (
    "rejection_sample",
    "expand_batch_to_tokens",
    "generate_uniform_probs",
    "sample_recovered_tokens",
    "apply_sampling_constraints",
    "RejectionSampler.__init__",
    "RejectionSampler.forward",
)
HASHED_NAMES = KERNELS + CALL_SITES
CONSTANTS = ("MAX_SPEC_LEN", "PLACEHOLDER_TOKEN_ID", "GREEDY_TEMPERATURE")
# Helpers in other modules whose output feeds the kernels: {qualname: module}.
EXTERNAL = {
    "unconditional_to_conditional_rates": "vllm.v1.spec_decode.utils",
}


def normalize_source(src: str) -> str:
    lines = [line.rstrip() for line in src.splitlines()]
    return textwrap.dedent("\n".join(lines)).strip("\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _module_file(module: str) -> Path:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {module}")
    return Path(spec.origin)


def upstream_file() -> Path:
    return _module_file(UPSTREAM_MODULE)


def _defs(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: dict[str, str] = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = normalize_source(_segment(lines, node))
    return out


def external_sources() -> dict[str, str]:
    """Normalized sources of ``EXTERNAL`` helpers, keyed ``module:name``."""
    out: dict[str, str] = {}
    for name, module in EXTERNAL.items():
        defs = _defs(_module_file(module))
        if name not in defs:
            raise RuntimeError(f"{module}: upstream definition not found: {name}")
        out[f"{module}:{name}"] = defs[name]
    return out


def _segment(lines: list[str], node: ast.AST) -> str:
    decorators = getattr(node, "decorator_list", [])
    start = (decorators[0].lineno if decorators else node.lineno) - 1  # type: ignore[attr-defined]
    return "\n".join(lines[start : node.end_lineno])  # type: ignore[attr-defined]


def extract_sources(path: Path | None = None) -> tuple[dict[str, str], dict[str, Any]]:
    """Return ({qualified name: normalized source}, {constant: value})."""
    path = path or upstream_file()
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    sources: dict[str, str] = {}
    consts: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sources[node.name] = normalize_source(_segment(lines, node))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sources[f"{node.name}.{sub.name}"] = normalize_source(
                        _segment(lines, sub)
                    )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id in CONSTANTS:
                    consts[t.id] = ast.literal_eval(node.value)  # type: ignore[arg-type]
    missing = [n for n in HASHED_NAMES if n not in sources]
    missing += [c for c in CONSTANTS if c not in consts]
    if missing:
        raise RuntimeError(f"{path}: upstream definitions not found: {missing}")
    return {n: sources[n] for n in HASHED_NAMES}, consts


def current_record() -> dict[str, Any]:
    from importlib.metadata import version

    sources, consts = extract_sources()
    return {
        "vllm_version": version("vllm"),
        "module": UPSTREAM_MODULE,
        "normalization": "rstrip lines; textwrap.dedent; strip blank edges; "
        "segment = first decorator .. end of def (ast)",
        "hashes": {n: sha256_text(s) for n, s in sources.items()},
        "external_hashes": {n: sha256_text(s) for n, s in external_sources().items()},
        "constants": consts,
    }


def load_recorded() -> dict[str, Any]:
    return json.loads(HASHES_PATH.read_text())


def write_recorded(path: Path = HASHES_PATH) -> dict[str, Any]:
    rec = current_record()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Refresh Tier C upstream hashes")
    parser.add_argument(
        "--output",
        type=Path,
        default=HASHES_PATH,
        help=f"hash file to write (default: {HASHES_PATH})",
    )
    args = parser.parse_args()
    record = write_recorded(args.output)
    print(f"wrote {args.output} for vllm {record['vllm_version']}")


if __name__ == "__main__":
    main()
