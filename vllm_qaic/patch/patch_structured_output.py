# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm/vllm/v1/structured_output/utils.py

"""Monkey-patch for vllm.v1.structured_output.utils.apply_grammar_bitmask.

On QAIC, the upstream function builds the xgrammar index tensor with
pin_memory=True (no CUDA allocator -> crash) and passes a torch.Tensor to the
CPU xgrammar kernel, which only accepts a python Sequence[int]. This breaks
guided/structured decoding.

Replace apply_grammar_bitmask with a version that gates the pinned tensor path
on is_pin_memory_available() and passes a plain list of indices on QAIC.

The AoT model runner overrides ``sample_tokens`` and calls the QAIC-safe
``qaic_apply_grammar_bitmask`` directly, so it needs no patch. PyT inherits
upstream ``GPUModelRunner.sample_tokens``, whose by-name-imported
``apply_grammar_bitmask`` is only reachable via this monkeypatch, which
rebinds it to the same QAIC-safe function.
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
