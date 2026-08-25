# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Shared QAIC utilities used across vLLM"""

from typing import TYPE_CHECKING, Any

import regex as re
import torch
from QEfficient import QEFFAutoModelForCausalLM, QEFFAutoModelForImageTextToText
from transformers import AutoConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

from vllm_qaic.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def _get_attention_type(qeff_config: Any, layer_idx: int) -> str | None:
    if hasattr(qeff_config, "text_config"):
        cfg = qeff_config.text_config
    else:
        cfg = qeff_config

    layer_types = getattr(cfg, "layer_types", None)
    if layer_types is None:
        return None

    if layer_idx >= len(layer_types):
        raise ValueError("Invalid layer Index")

    return layer_types[layer_idx]


def _is_swa_layer(qeff_config: Any, layer_idx: int) -> int | None:
    layer_type = _get_attention_type(qeff_config, layer_idx)

    if layer_type == "sliding_attention":
        cfg = (
            qeff_config.text_config
            if hasattr(qeff_config, "text_config")
            else qeff_config
        )
        sliding_window = cfg.sliding_window
        return int(sliding_window)

    return None


def _is_mamba_type(qeff_config: Any, layer_idx: int) -> bool:
    layer_type = _get_attention_type(qeff_config, layer_idx)

    return layer_type == "linear_attention"


def _get_mamba_spec(
    vllm_config: "VllmConfig",
    block_size: int,
    page_size_padded: int | None = None,
) -> MambaSpec:
    from vllm.model_executor.models import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls(
        vllm_config.model_config.architecture,
        model_config=vllm_config.model_config,
    )
    mamba_spec = MambaSpec(
        block_size=vllm_config.cache_config.mamba_block_size or block_size,
        shapes=tuple(model_cls.get_mamba_state_shape_from_config(vllm_config)),
        dtypes=tuple(model_cls.get_mamba_state_dtype_from_config(vllm_config)),
        page_size_padded=page_size_padded,
        mamba_cache_mode=vllm_config.cache_config.mamba_cache_mode,
        num_speculative_blocks=(
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        ),
    )
    return mamba_spec


def _get_kv_cache_spec(
    vllm_config: "VllmConfig", kv_cache_dtype: torch.dtype
) -> dict[str, KVCacheSpec]:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    model_name = model_config.model
    is_multimodal = model_config.is_multimodal_model
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=model_config.trust_remote_code,
        revision=model_config.revision,
    )
    if is_multimodal:
        qeff_config = QEFFAutoModelForImageTextToText.from_pretrained(
            model_name, config=config
        ).model.config
    else:
        qeff_config = QEFFAutoModelForCausalLM.from_pretrained(
            model_name, config=config
        ).model.config

    block_size = cache_config.block_size
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_quant_mode = get_kv_quant_mode(cache_config.cache_dtype)

    start_layer, end_layer = model_config.get_layers_start_end_indices(parallel_config)

    kv_cache_spec: dict[str, KVCacheSpec] = {}

    for local_idx, layer_idx in enumerate(range(start_layer, end_layer)):
        layer_name = f"layer_{local_idx}"
        sliding_window = _is_swa_layer(qeff_config, layer_idx)
        if sliding_window is not None:
            kv_cache_spec[layer_name] = SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=kv_cache_dtype,
                kv_quant_mode=kv_quant_mode,
                sliding_window=sliding_window,
            )
        else:
            kv_cache_spec[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=kv_cache_dtype,
                kv_quant_mode=kv_quant_mode,
            )

    return kv_cache_spec


def _clean_config(
    cfg: dict[str, Any] | None,
    vllm_config: "VllmConfig | None" = None,
) -> dict[str, Any]:
    update_cfg: dict[str, Any] = {}
    if cfg is None:
        return {}
    # compiler args
    if "compiler_args" in cfg:
        if cfg["compiler_args"] is not None:
            vl = re.split(r" |\|", cfg["compiler_args"])
            for v in vl:
                v = v.split("=")
                if len(v) == 1:
                    cfg[v[0]] = True
                else:
                    cfg[v[0]] = v[1]
        del cfg["compiler_args"]
    # Fix key names
    _cfg = {}
    for key in cfg:
        key1 = key.lower().replace("-", "_").strip()
        _cfg[key1] = cfg[key]
    cfg = _cfg
    # Clean override config
    for key in cfg:
        value = cfg[key]
        if value is not None:
            # key = key.lower().replace('-','_')
            if isinstance(value, str | bool):
                if (
                    key != "qpc_path"
                    and key != "mdp_load_partition_config"
                    and key != "aic_pmu_recipe"
                    and key != "mdp_dump_partition_config"
                    and key != "mdp_compiler_dump_path"
                    and key != "node_precision_info"
                ):
                    value = str(value).lower()
                value = str(value).strip()
            # Ignore donot update list
            _ignore_list = [
                "ctx_len",
                "batch_size",
                "full_batch_size",
                "num_speculative_tokens",
            ]
            if key in _ignore_list:
                continue
            # specific filters
            # num_device
            if key in ["device_id", "device_group", "device_ids"]:
                if isinstance(value, str):
                    # value = value.replace('[','').replace(']','')
                    # value = value.split(',')
                    value = re.sub(r"[^0-9]", " ", value).strip()
                    value = re.sub(r" +", ",", value).split(",")
                if isinstance(value, int):
                    value = [value]
                value = [int(v) for v in value]
                update_cfg["device_group"] = value
                update_cfg["num_devices"] = len(value)
            # num_cores
            elif key in ["num_cores", "aic_num_cores"]:
                update_cfg["num_cores"] = int(value)
            # num_devices
            elif key in ["num_devices"]:
                update_cfg["num_devices"] = int(value)
            # mxfp6
            elif key in ["mxfp6", "mxfp6_matmul", "mxfp6_en"] or (value == "mxfp6"):
                update_cfg["mxfp6_matmul"] = value not in ["false", "0"]
            # mxint8
            elif key in ["mxint8", "mxint8_en", "mxint8_kv_cache"] or (
                value == "mxint8"
            ):
                update_cfg["mxint8_kv_cache"] = value not in ["false", "0"]
            # node_precision_info:
            #   encode instance  → value is a file path string, pass through as-is
            #   other instances  → value is True/False, convert to bool for
            #                      on-the-fly NPI YAML generation inside qeff
            elif key in ["node_precision_info"]:
                if value.lower() in ("true", "1"):
                    update_cfg["node_precision_info"] = True
                elif value.lower() in ("false", "0"):
                    update_cfg["node_precision_info"] = False
                else:
                    # Treat as a file-system path; preserve the original string.
                    update_cfg["node_precision_info"] = value
            elif key in ["dfs", "aic_enable_depth_first"]:
                update_cfg["aic_enable_depth_first"] = value not in [
                    "false",
                    "0",
                ]
            # mos
            elif key == "mos":
                update_cfg["mos"] = int(value)
            elif key == "mdts_mos":
                update_cfg[key] = int(value)
            # Anything else will pass as it is
            elif value in ["", "true", "1"]:
                update_cfg[key] = True
            elif value in ["false", "0"]:
                update_cfg[key] = False
            elif key == "embed_seq_len":
                if isinstance(value, str):
                    value = value.strip().split(",")
                    value = list(map(int, value))
                elif isinstance(value, int):
                    assert vllm_config is not None, (
                        "vllm_config required for embed_seq_len"
                    )
                    assert value == vllm_config.model_config.max_model_len, (
                        "sequence length should be the same as max_model_len"
                    )
                assert vllm_config is not None, "vllm_config required for embed_seq_len"
                assert vllm_config.model_config.max_model_len in value, (
                    "max_model_len should be passed in embed_seq_len"
                )
                update_cfg["prefill_seq_len"] = value
            elif key in [
                "comp_ctx_lengths_prefill",
                "comp_ctx_lengths_decode",
            ]:
                try:
                    if isinstance(value, str):
                        value = value.strip().split(",")
                    value = list(map(int, value))
                    assert len(value) > 0, f"{key} should be non-empty"
                    assert vllm_config is not None, (
                        "vllm_config required for comp_ctx_lengths"
                    )
                    assert all(
                        v <= vllm_config.model_config.max_model_len for v in value
                    ), (
                        "All values of comp_ctx_lengths must be integers "
                        "and less than max_model_len"
                    )
                    value.sort()
                    update_cfg[key] = value
                except Exception:
                    logger.warning("Compute Context Lengths not found")
            else:  # For other compiler args
                update_cfg[key] = value
    return update_cfg
