# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Run the fused-MoE GELU-tanh gated activation with upstream's PyTorch math.

``fused_experts_impl`` applies the activation between the two expert GEMMs via
``apply_moe_activation``, which for ``MoEActivation.GELU_TANH`` calls the
compiled op ``torch.ops._C.gelu_tanh_and_mul`` directly -- no CustomOp
dispatch and no native fallback -- and that op is not built for QAIC. This
wraps ``apply_moe_activation`` so GELU_TANH computes the same math as
upstream's ``GeluAndMul(approximate="tanh").forward_native``
(``gelu_tanh(x[..., :d]) * x[..., d:]``); every other activation is passed
through to the original function unchanged.
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

# Saved before rebinding below: the wrapper replaces this module attribute.
_apply_moe_activation = fused_moe.apply_moe_activation


def _qaic_apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """apply_moe_activation with GELU_TANH routed to upstream PyTorch math."""
    if (
        activation == MoEActivation.GELU_TANH
        and kwargs.get("valid_token_counts") is None
    ):
        # Same math as GeluAndMul(approximate="tanh").forward_native, computed
        # inline: a CustomOp cannot be built at forward time (it reads the vLLM
        # config, which is only set while the model is constructed).
        d = input.shape[-1] // 2
        output.copy_(F.gelu(input[..., :d], approximate="tanh") * input[..., d:])
        return output

    return _apply_moe_activation(activation, output, input, **kwargs)


# fused_moe imports apply_moe_activation by name and resolves it as a module
# global at call time, so rebinding it here is enough for fused_experts_impl.
fused_moe.apply_moe_activation = _qaic_apply_moe_activation
