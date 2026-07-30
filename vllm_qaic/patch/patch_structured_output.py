# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/structured_output/utils.py

"""Monkey-patch for vllm.v1.structured_output.utils.apply_grammar_bitmask.

QAIC (PyT mode) is a non-CUDA device, so the upstream implementation fails
two ways when applying the xgrammar bitmask:

  1. It builds the index tensor with ``pin_memory=True``. Pinned host memory
     requires a CUDA allocator, which QAIC does not have, raising
     "Need to provide pin_memory allocator to use pin memory".

  2. It passes that tensor as ``indices`` to
     ``xgr.apply_token_bitmask_inplace``. On a non-CUDA device xgrammar
     dispatches to the CPU kernel (``apply_token_bitmask_inplace_cpu``),
     which only accepts a python ``Sequence[int]``, not a torch.Tensor.

How:
    Replace ``apply_grammar_bitmask`` with a QAIC-safe version that gates the
    pinned-tensor path on ``is_pin_memory_available()`` (False on QAIC) and
    passes the plain python list of indices otherwise. Behaviour is identical
    to upstream on CUDA.
"""

from vllm.v1.structured_output import utils as _so_utils

from vllm_qaic.utils.qaic_utils import qaic_apply_grammar_bitmask

# Patch the definition module.
_so_utils.apply_grammar_bitmask = qaic_apply_grammar_bitmask

# Callers that did `from ...utils import apply_grammar_bitmask` bind the name at
# import time, so also rebind it in any module that already imported it.
try:
    from vllm.v1.worker import gpu_model_runner as _gpu_mr

    if hasattr(_gpu_mr, "apply_grammar_bitmask"):
        _gpu_mr.apply_grammar_bitmask = qaic_apply_grammar_bitmask
except Exception:  # pragma: no cover - defensive; gpu_model_runner may be absent
    pass
