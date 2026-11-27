# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Shared QAIC utilities used across vLLM"""

from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re

from vllm_qaic.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig

logger = init_logger(__name__)


class KVCacheGranularity(Enum):
    """Whether handoff transfers a context prefix or the complete state."""

    PARTIAL = "PARTIAL"
    FULL = "FULL"


class KVCacheKind(Enum):
    """Simple enum class to store different type of kv$ supported on qaic"""

    FA = "FA"
    LA_CONV = "LA_CONV"
    LA_RECURRENT = "LA_RECURRENT"
    MSA = "MSA"
    INDEX = "INDEX"
    CSA0 = "CSA0"
    CSA1 = "CSA1"
    CSA2 = "CSA2"
    CSA3 = "CSA3"
    CSA4 = "CSA4"
    HCA0 = "HCA0"
    HCA1 = "HCA1"
    HCA2 = "HCA2"
    HCA3 = "HCA3"
    SWA = "SWA"

    @property
    def granularity(self) -> KVCacheGranularity:
        if self in (
            KVCacheKind.FA,
            KVCacheKind.MSA,
            KVCacheKind.INDEX,
            KVCacheKind.CSA1,
            KVCacheKind.CSA2,
        ):
            return KVCacheGranularity.PARTIAL
        return KVCacheGranularity.FULL

    @property
    def compression_ratio(self) -> int:
        return 4 if self in (KVCacheKind.CSA1, KVCacheKind.CSA2) else 1


class KVTransform:
    """TS1 prefill -> TS16 decode layouts for MiniMax M3 and DeepSeek-V4 Flash.

    Shapes are deliberately fixed to the current B1/DP1 handoff ABI. HCA score
    state dimensions, unspecified in the shape table, come from the QPC binding.
    All transforms preserve dtype and packed feature order.
    """

    TS = 16
    PREFILL_SHAPES = {
        KVCacheKind.FA: (1, 4, 65536, 132),
        KVCacheKind.MSA: (1, 4, 65536, 132),
        KVCacheKind.INDEX: (1, 1, 65536, 132),
        KVCacheKind.CSA0: (1, 128, 512),
        KVCacheKind.CSA1: (1, 65536, 512),
        KVCacheKind.CSA2: (1, 65536, 128),
        KVCacheKind.CSA3: (1, 8, 2048),
        KVCacheKind.CSA4: (1, 8, 512),
        KVCacheKind.HCA0: (1, 128, 512),
        KVCacheKind.HCA1: (1, 65536, 512),
        KVCacheKind.HCA2: (1, 65536, 128),
        KVCacheKind.SWA: (1, 128, 512),
    }

    @classmethod
    def native_shape(cls, kind: KVCacheKind, decode_shape: tuple) -> tuple:
        if kind == KVCacheKind.HCA3:
            return (1,) + tuple(decode_shape[2:])
        return cls.PREFILL_SHAPES[kind]

    @staticmethod
    def reshape_transpose(
        cache: np.ndarray, cp: int, block_size: int = 1
    ) -> np.ndarray:
        """T1: cyclic slots, or block-cyclic slots for the MiniMax indexer."""
        if cache.ndim == 3:
            cache = cache[:, None]
        batch, heads, slots, features = cache.shape
        group_size = cp * block_size
        padded_slots = -(-slots // group_size) * group_size
        if padded_slots != slots:
            cache = np.pad(cache, ((0, 0), (0, 0), (0, padded_slots - slots), (0, 0)))
        grouped = cache.reshape(
            batch, heads, padded_slots // group_size, cp, block_size, features
        )
        rows = grouped.transpose(0, 3, 1, 2, 4, 5)
        return np.ascontiguousarray(
            rows.reshape(batch, cp * heads, padded_slots // cp, features)
        )

    @staticmethod
    def partition_local_ring(cache: np.ndarray, cp: int) -> np.ndarray:
        """T2: consecutive local-ring chunks, without interleaving."""
        batch, slots, features = cache.shape
        return np.ascontiguousarray(cache.reshape(batch, cp, slots // cp, features))

    @staticmethod
    def replicate(cache: np.ndarray, cp: int) -> np.ndarray:
        """T3: a complete copy of the packed state on every decode row."""
        return np.repeat(cache[:, None], cp, axis=1)

    @classmethod
    def apply(
        cls, cache: np.ndarray, kind: KVCacheKind, index_block_size: int = 1
    ) -> np.ndarray:
        if cache.size == 0:
            return cache
        if kind in (KVCacheKind.FA, KVCacheKind.MSA, KVCacheKind.INDEX):
            if cache.shape[1] == cls.TS:
                return cache
            cp = cls.TS // cls.PREFILL_SHAPES[kind][1]
            block_size = index_block_size if kind == KVCacheKind.INDEX else 1
            return cls.reshape_transpose(cache, cp, block_size)
        if cache.ndim == 4:
            return cache  # Already in decode layout.
        if kind in (KVCacheKind.CSA0, KVCacheKind.HCA0):
            return cls.partition_local_ring(cache, cls.TS)
        if kind in (KVCacheKind.CSA1, KVCacheKind.CSA2, KVCacheKind.HCA1):
            return cls.reshape_transpose(cache, cls.TS)
        if kind in (
            KVCacheKind.CSA3,
            KVCacheKind.CSA4,
            KVCacheKind.HCA2,
            KVCacheKind.HCA3,
            KVCacheKind.SWA,
        ):
            return cls.replicate(cache, cls.TS)
        raise ValueError(f"No TS16 handoff transform for {kind}")


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
