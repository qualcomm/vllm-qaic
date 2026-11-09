# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Per-operator Triton opt-in flags for QAIC CustomOps.

Each QAIC ``forward_oot`` that has a vLLM Triton kernel available checks
``triton_op_enabled("<OP_NAME>")`` first and dispatches to the Triton path only
when the corresponding ``VLLM_QAIC_TRITON_<OP_NAME>`` environment flag is set to
``"1"``. When unset it falls back to the existing QAIC NSP / SDPA path, so every
Triton path is opt-in and can be enabled one operator at a time.

Each per-op flag defaults to ``VLLM_QAIC_TRITON_ALL_OPS_EN`` (see
``vllm_qaic.envs``), so setting that one variable enables every Triton path while
an explicit ``VLLM_QAIC_TRITON_<OP_NAME>=0`` still opts a single op out.

Flags are resolved through ``vllm_qaic.envs``, which keeps
``qaic_environment_variables`` the single place a flag is declared. Each op module
resolves its flag once at import time into a module-level ``triton_en``, so the
environment must be set before ``vllm_qaic`` ops are imported.
"""

import importlib.util

import vllm_qaic.envs as qaic_envs


def triton_op_enabled(op_name: str) -> bool:
    """True if ``VLLM_QAIC_TRITON_<OP_NAME>`` is set to ``"1"``.

    Raises ``AttributeError`` if the flag is not declared in ``vllm_qaic.envs``,
    so a misspelled ``op_name`` surfaces instead of silently reading as off.
    Raises ``RuntimeError`` if the flag is on (explicitly or via
    ``VLLM_QAIC_TRITON_ALL_OPS_EN``) but the ``triton`` package is not installed.
    (Checked with ``find_spec`` rather than ``vllm.triton_utils.HAS_TRITON``,
    which cannot be imported while ``QaicPlatform`` is still loading.)
    """
    enabled = getattr(qaic_envs, f"VLLM_QAIC_TRITON_{op_name}")
    if enabled and importlib.util.find_spec("triton") is None:
        raise RuntimeError(
            f"VLLM_QAIC_TRITON_{op_name} is enabled (directly or through "
            "VLLM_QAIC_TRITON_ALL_OPS_EN) but Triton is not available. "
            "Install the QAIC Triton compiler wheel, or unset the flag."
        )
    return enabled
