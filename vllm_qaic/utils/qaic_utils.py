# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Shared QAIC utilities used across vLLM"""

from typing import TYPE_CHECKING, Any

import regex as re

from vllm_qaic.logger import init_logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, 
    KVCacheConfig, 
    KVCacheSpec, 
    SlidingWindowSpec,
    get_kv_quant_mode
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

def _is_swa_layer(
    hf_config: Any,
    layer_index: int
) -> bool:
    
    # for the models we support only Gemma4 family and GPT-OSS 
    # has clear per layer attention information inside
    # hf config
    
    if getattr(hf_config, "text_config"):
        layer_types = hf_config["text_config"]["layer_types"]
    else:
        layer_types = hf_config["layer_types"]
    
    if layer_index >= len(layer_types):
        raise ValueError("Layer number out of index")
    
    if layer_types[layer_index] == "sliding_attention":
        sliding_window = getattr(hf_config, "sliding_window", None)
        return int(sliding_window)
    
    return None

def _get_kv_cache_spec(
    vllm_config: "VllmConfig",
    kv_cache_dtype: torch.dtype
) -> dict[str, KVCacheSpec]:
    
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config
    hf_config = model_config.hf_config
    
    # for creatiung KVCacheSpec we need block_size from
    # cache_config, num_kv_heads from model_config
    block_size = cache_config.block_size
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_quant_mode = get_kv_quant_mode(cache_config.cache_dtype)
    
    start_layer, end_layer = model_config.get_layers_start_end_indices(parallel_config)
    kv_cache_spec : dict[str, KVCacheSpec] = {}
    for i in range(start, end+1):
        layer_id = f"layer_{i - start + 1}"
        sliding_attention = _is_swa_layer(hf_config, i)
        if sliding_attention is not None:
            # current attention is a SWA
            kv_cache_spec[layer_id] = SlidingWindowSpec(
                block_size = block_size,
                num_kv_heads = num_kv_heads,
                head_size=head_size,
                dtype=kv_cache_dtype,
                kv_quant_mode=kv_quant_mode,
                sliding_window=sliding_window,
            )
        else
            kv_cache_spec[layer_id] = FullAttentionSpec(
                block_size = block_size,
                num_kv_heads = num_kv_heads,
                head_size = head_size,
                dtype = kv_cache_dtype,
                kv_quant_mode=kv_quant_mode,
            )
    
    return kv_cache_spec                     


def _clean_config(
    cfg: dict[str, Any] | None,
    vllm_config: "VllmConfig | None" = None,
) -> dict[str, Any]:
    update_cfg = {}
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
