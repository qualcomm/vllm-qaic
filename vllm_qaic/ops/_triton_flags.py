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

Flags are read per call (a cheap dict lookup through ``vllm_qaic.envs``) so they
can be toggled without re-registering ops.
"""

import os


def triton_op_enabled(op_name: str) -> bool:
    """True if ``VLLM_QAIC_TRITON_<OP_NAME>`` is set to ``"1"``."""
    return os.environ.get(f"VLLM_QAIC_TRITON_{op_name}", "0") == "1"
