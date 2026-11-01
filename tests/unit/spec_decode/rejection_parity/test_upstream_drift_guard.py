# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Tier C: fail when the upstream rejection-sampler kernels/call sites change.

Compares AST-extracted, normalized source hashes (``_upstream_sources``) of the
four Triton kernels, their call sites, ``unconditional_to_conditional_rates``
and the module constants with
``fixtures/upstream_hashes.json``.  No triton needed; the hashes come from the
module *file*, so they are identical with real triton-cpu ``JITFunction``
kernels and under vLLM's ``TritonPlaceholder``.

Refresh (after re-verifying mandatory tier A)::

    .venv_aot/bin/python tests/unit/spec_decode/rejection_parity/_upstream_sources.py
"""

from __future__ import annotations

from importlib.metadata import version

import _upstream_sources as us
import pytest

RECORDED = us.load_recorded()
CURRENT = us.current_record()

REMEDY = (
    "Run mandatory tier A in the dedicated triton-cpu parity environment "
    "(VLLM_QAIC_REQUIRE_TRITON_PARITY=1 pytest -s -q "
    "tests/unit/spec_decode/rejection_parity), then refresh tier C hashes with "
    "python tests/unit/spec_decode/rejection_parity/_upstream_sources.py."
)


def _changed() -> list[str]:
    names = sorted(set(RECORDED["hashes"]) | set(CURRENT["hashes"]))
    changed = [
        n for n in names if RECORDED["hashes"].get(n) != CURRENT["hashes"].get(n)
    ]
    rec_ext, cur_ext = RECORDED["external_hashes"], CURRENT["external_hashes"]
    changed += [
        n
        for n in sorted(set(rec_ext) | set(cur_ext))
        if rec_ext.get(n) != cur_ext.get(n)
    ]
    for c in sorted(set(RECORDED["constants"]) | set(CURRENT["constants"])):
        old, new = RECORDED["constants"].get(c), CURRENT["constants"].get(c)
        if old != new:
            changed.append(f"{c} ({old!r} -> {new!r})")
    return changed


def test_upstream_rejection_sampler_unchanged():
    changed = _changed()
    if changed:
        pytest.fail(
            "upstream vLLM rejection-sampler kernel/call-site changed: "
            f"{', '.join(changed)}. {REMEDY}"
        )


def test_recorded_covers_required_names():
    assert set(RECORDED["hashes"]) == set(us.HASHED_NAMES)
    assert set(RECORDED["constants"]) == set(us.CONSTANTS)
    assert set(RECORDED["external_hashes"]) == {
        f"{m}:{n}" for n, m in us.EXTERNAL.items()
    }


def test_vllm_version_recorded():
    # Informational: a version bump with identical hashes is fine.
    assert RECORDED["vllm_version"]
    if RECORDED["vllm_version"] != version("vllm"):
        print(
            f"NOTE: hashes recorded for vllm {RECORDED['vllm_version']}, "
            f"running {version('vllm')}; sources unchanged={not _changed()}"
        )


def test_live_kernel_source_matches_file():
    """The live objects (JITFunction ``.fn`` or placeholder fn) match the file."""
    import inspect

    from vllm.v1.sample import rejection_sampler as rs

    from _parity_common import ORIGINAL_KERNELS

    sources, _ = us.extract_sources()
    for name in us.KERNELS:
        obj = ORIGINAL_KERNELS[name]
        fn = getattr(obj, "fn", obj)
        live = us.normalize_source(inspect.getsource(fn))
        body = sources[name]
        # placeholder decorators may hide the decorator line; compare from `def`
        assert body[body.index("def ") :] == live[live.index("def ") :], name
    assert CURRENT["constants"]["MAX_SPEC_LEN"] == rs.MAX_SPEC_LEN
