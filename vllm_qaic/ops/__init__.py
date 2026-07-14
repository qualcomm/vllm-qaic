# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

_QAIC_CUSTOMOP_IS_REGISTERED = False


def register_qaic_customop():
    """Register QAic Layers specific CustomOP

    NOTE: if the register branch requires model type, please use
    `vllm.config.get_current_vllm_config`, and ensure this will execute after model
    config is initilazed.
    """
    global _QAIC_CUSTOMOP_IS_REGISTERED
    if _QAIC_CUSTOMOP_IS_REGISTERED:
        return

    from vllm.model_executor.custom_op import CustomOp

    # relative import for cross-compatibility in plugin and fork
    from .activation import QAicSiluAndMul
    from .grouped_topk_router import register_qaic_grouped_topk_router
    from .layernorm import QAicGemmaRMSNorm, QAicRMSNorm, QAicRMSNormGated
    from .mm_encoder_attention import QAicMMEncoderAttention
    from .mrope import QAicMRotaryEmbedding
    from .topk_router import register_qaic_topk_router
    from .unquantized_fused_moe_method import QAicUnquantizedFusedMoEMethod

    register_qaic_topk_router()
    register_qaic_grouped_topk_router()

    CustomOp.register_oot(
        _decorated_op_cls=QAicUnquantizedFusedMoEMethod,
        name="UnquantizedFusedMoEMethod",
    )
    CustomOp.register_oot(_decorated_op_cls=QAicRMSNorm, name="RMSNorm")
    CustomOp.register_oot(_decorated_op_cls=QAicGemmaRMSNorm, name="GemmaRMSNorm")
    CustomOp.register_oot(_decorated_op_cls=QAicRMSNormGated, name="RMSNormGated")
    CustomOp.register_oot(_decorated_op_cls=QAicSiluAndMul, name="SiluAndMul")
    CustomOp.register_oot(
        _decorated_op_cls=QAicMMEncoderAttention, name="MMEncoderAttention"
    )
    CustomOp.register_oot(
        _decorated_op_cls=QAicMRotaryEmbedding, name="MRotaryEmbedding"
    )

    # NOTE: Keep this at last to ensure all custom actions are registered
    _QAIC_CUSTOMOP_IS_REGISTERED = True
