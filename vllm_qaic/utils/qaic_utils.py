# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Shared QAIC utilities used across vLLM"""

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from vllm.config.vllm import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (  # type: ignore[reportMissingImports]
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

from vllm_qaic.logger import init_logger
from vllm_qaic.utils import get_qaic_kv_connector_names

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig  # type: ignore[reportMissingImports]

logger = init_logger(__name__)

def _is_linear_attention(layer_type: str | None) -> bool:
    if layer_type is None:
        return False
    value = layer_type.lower()
    return any(
        name in value for name in ("linear", "recurrent", "mamba", "gdn", "delta")
    )


def _get_model_mamba_spec(vllm_config: "VllmConfig") -> MambaSpec:
    from vllm.model_executor.models import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls(
        vllm_config.model_config.architecture,
        model_config=vllm_config.model_config,
    )
    if hasattr(model_cls, "get_mamba_specs_from_config"):
        candidates = model_cls.get_mamba_specs_from_config(vllm_config)
    else:
        candidates = (
            MambaSpec(
                shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
                dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            ),
        )
    if not candidates:
        raise RuntimeError("QAIC could not derive the model's recurrent KV state")
    spec = max(candidates, key=lambda candidate: candidate.page_size_bytes)
    return replace(
        spec,
        block_size=vllm_config.cache_config.block_size,
        page_size_padded=vllm_config.cache_config.mamba_page_size_padded,
    )


def _get_config_kv_cache_spec(
    vllm_config: "VllmConfig", kv_cache_dtype: torch.dtype
) -> dict[str, KVCacheSpec]:
    model_config = vllm_config.model_config
    hf_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    layer_types = getattr(hf_config, "layer_types", None)
    base_block_size = vllm_config.cache_config.block_size
    quant_mode = get_kv_quant_mode(vllm_config.cache_config.cache_dtype)
    default_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
    default_head_size = model_config.get_head_size()
    mamba_spec = None
    if layer_types and any(
        _is_linear_attention(layer_type) for layer_type in layer_types
    ):
        mamba_spec = _get_model_mamba_spec(vllm_config)

    specs: dict[str, KVCacheSpec] = {}
    start_layer, end_layer = model_config.get_layers_start_end_indices(
        vllm_config.parallel_config
    )
    for local_idx, global_idx in enumerate(range(start_layer, end_layer)):
        layer_type = layer_types[global_idx] if layer_types else None
        layer_config = (
            hf_config.per_layer_config[global_idx]
            if hasattr(hf_config, "per_layer_config")
            else hf_config
        )
        num_kv_heads = getattr(layer_config, "num_key_value_heads", default_heads)
        head_size = getattr(layer_config, "head_dim", default_head_size)
        if _is_linear_attention(layer_type):
            assert mamba_spec is not None
            spec = mamba_spec
        elif layer_type == "sliding_attention":
            window = getattr(hf_config, "sliding_window", None)
            if window is None:
                raise ValueError(
                    f"Layer {global_idx} is sliding_attention without a window"
                )
            spec = SlidingWindowSpec(
                block_size=base_block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=kv_cache_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=int(window),
            )
        else:
            spec = FullAttentionSpec(
                block_size=base_block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=kv_cache_dtype,
                kv_quant_mode=quant_mode,
            )
        specs[f"model.layers.{local_idx}"] = replace(
            spec, block_size=base_block_size
        )
    return specs


_PAGED_KV_CONNECTORS = {
    "NixlConnector",
    "QaicNixlPushConnector",
    "MooncakeConnector",
    "MooncakeStoreConnector",
}


def _get_legacy_qaic_kv_cache_spec(
    vllm_config: "VllmConfig", kv_cache_dtype: torch.dtype
) -> dict[str, KVCacheSpec]:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    spec = FullAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=model_config.get_num_kv_heads(parallel_config),
        head_size=model_config.get_head_size(),
        dtype=kv_cache_dtype,
        kv_quant_mode=get_kv_quant_mode(vllm_config.cache_config.cache_dtype),
    )
    return {
        f"model.layers.{layer_idx}": spec
        for layer_idx in range(model_config.get_num_layers(parallel_config))
    }


def _normalize_paged_kv_specs(
    specs: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Keep one 128-token block size while padding physical page payloads."""
    common_page_size = max(spec.page_size_bytes for spec in specs.values())
    return {
        name: replace(spec, page_size_padded=common_page_size)
        if isinstance(spec, (AttentionSpec, MambaSpec))
        else spec
        for name, spec in specs.items()
    }


def get_qaic_kv_cache_spec(
    vllm_config: "VllmConfig", kv_cache_dtype: torch.dtype
) -> dict[str, KVCacheSpec]:
    """Build QAIC cache specs from the model's FA, SWA, and LA layers."""
    connector_names = get_qaic_kv_connector_names(
        vllm_config.kv_transfer_config
    )
    if not connector_names & _PAGED_KV_CONNECTORS:
        return _get_legacy_qaic_kv_cache_spec(vllm_config, kv_cache_dtype)

    layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    if layers:
        specs: dict[str, KVCacheSpec] = {}
        for local_layer_idx, layer in enumerate(layers.values()):
            spec = layer.get_kv_cache_spec(vllm_config)
            if spec is None:
                continue
            if isinstance(spec, AttentionSpec):
                spec = layer.get_attn_backend().customize_spec(spec)
            if not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec, MambaSpec)):
                raise NotImplementedError(
                    f"QAIC hybrid KV does not support {type(spec).__name__}"
                )
            specs[f"model.layers.{local_layer_idx}"] = replace(
                spec, block_size=vllm_config.cache_config.block_size
            )
        if specs:
            return _normalize_paged_kv_specs(specs)
    specs = _get_config_kv_cache_spec(vllm_config, kv_cache_dtype)
    if not specs:
        raise RuntimeError("QAIC model has no transferable KV cache layers")
    return _normalize_paged_kv_specs(specs)


def compute_max_decode_tokens(
    speculative_config: "SpeculativeConfig | None",
) -> int:
    if speculative_config is None:
        return 1
    return 1 + speculative_config.num_speculative_tokens


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
