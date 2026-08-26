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

Flags are resolved through ``vllm_qaic.envs``, which keeps
``qaic_environment_variables`` the single place a flag is declared. That module's
``__getattr__`` re-evaluates the lambda on every access, so flags are read per
call and can be toggled without re-registering ops.
"""

import vllm_qaic.envs as qaic_envs


def triton_op_enabled(op_name: str) -> bool:
    """True if ``VLLM_QAIC_TRITON_<OP_NAME>`` is set to ``"1"``.

    Raises ``AttributeError`` if the flag is not declared in ``vllm_qaic.envs``,
    so a misspelled ``op_name`` surfaces instead of silently reading as off.
    """
    return getattr(qaic_envs, f"VLLM_QAIC_TRITON_{op_name}")
