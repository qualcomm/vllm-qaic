# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0

"""Use vLLM's PyTorch top-k/top-p implementation for QAIC AOT.

In AOT mode vLLM's logits are host CPU tensors, but the active platform is
QAIC. That combination can select the Triton implementation, whose launch
configuration asks the QAIC platform for a number of compute units that AOT
does not provide. This patch replaces the selector itself at plugin
initialization time, before downstream vLLM modules bind imported aliases.

The patch is deliberately AOT-only. Eager QAIC keeps its existing sampler
behavior, and the old ``vllm_qaic.v1.sample.topk_topp_sampler_shim`` import
path remains available as a compatibility wrapper for local experiments.
"""

from __future__ import annotations

import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_qaic.platform import QaicPlatform

_installed = False


def install() -> None:
    """Replace the AOT top-k/top-p selector with the PyTorch implementation."""
    global _installed
    if _installed or not QaicPlatform.is_aot:
        return

    topk_topp_sampler.apply_top_k_top_p = apply_top_k_top_p_pytorch
    _installed = True


# Register at import time, before vLLM's sampler consumers bind the selector.
install()
