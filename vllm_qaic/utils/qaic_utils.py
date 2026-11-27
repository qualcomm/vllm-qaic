# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Shared QAIC utilities used across vLLM"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import regex as re

from vllm_qaic.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig, VllmConfig

logger = init_logger(__name__)


class KVGranularity(Enum):
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
    def granularity(self) -> KVGranularity:
        if self in (
            KVCacheKind.FA,
            KVCacheKind.MSA,
            KVCacheKind.INDEX,
            KVCacheKind.CSA1,
            KVCacheKind.CSA2,
        ):
            return KVGranularity.PARTIAL
        return KVGranularity.FULL

    @property
    def compression_ratio(self) -> int:
        return 4 if self in (KVCacheKind.CSA1, KVCacheKind.CSA2) else 1


@dataclass(frozen=True)
class KVCacheInfo:
    """Per-binding KV metadata and its handoff transform policy."""

    TS = 16

    kv_kind: KVCacheKind
    granularity: KVGranularity
    transform_category: str | None

    @staticmethod
    def _t1(cache: np.ndarray, cp: int) -> np.ndarray:
        if cache.ndim == 3:
            cache = cache[:, None]
        batch, heads, slots, features = cache.shape
        padded_slots = -(-slots // cp) * cp
        if padded_slots != slots:
            cache = np.pad(cache, ((0, 0), (0, 0), (0, padded_slots - slots), (0, 0)))
        grouped = cache.reshape(batch, heads, padded_slots // cp, cp, features)
        rows = grouped.transpose(0, 3, 1, 2, 4)
        return np.ascontiguousarray(
            rows.reshape(batch, cp * heads, padded_slots // cp, features)
        )

    def apply(self, kv_shape: tuple, kv_type: Any, kv_size: int) -> tuple:
        """Attach the transform policy to one QPC binding descriptor."""
        return kv_shape, kv_type, kv_size, self

    def native_shape(self, decode_shape: tuple, num_kv_heads: int) -> tuple:
        if self.transform_category is None:
            return tuple(decode_shape)
        batch, _, slots, features = decode_shape
        if self.transform_category == "T1" and self.kv_kind in (
            KVCacheKind.FA,
            KVCacheKind.MSA,
            KVCacheKind.INDEX,
        ):
            heads = 1 if self.kv_kind == KVCacheKind.INDEX else num_kv_heads
            cp = self.TS // heads
            return (batch, heads, slots * cp, features)
        if self.transform_category in ("T1", "T2"):
            return (batch, self.TS * slots, features)
        if self.transform_category == "T3":
            return (batch, slots, features)
        raise ValueError(f"Unsupported KV transform {self.transform_category!r}")

    @staticmethod
    def _t2(cache: np.ndarray, cp: int) -> np.ndarray:
        if cache.ndim == 4 and cache.shape[1] == 1:
            cache = cache[:, 0]
        batch, slots, features = cache.shape
        return np.ascontiguousarray(cache.reshape(batch, cp, slots // cp, features))

    @staticmethod
    def _t3(cache: np.ndarray, cp: int) -> np.ndarray:
        if cache.ndim == 4 and cache.shape[1] == 1:
            cache = cache[:, 0]
        return np.repeat(cache[:, None], cp, axis=1)

    def transform(self, cache: np.ndarray, decode_shape: tuple) -> np.ndarray:
        if self.transform_category is None or cache.size == 0:
            return cache
        if cache.ndim == 4 and cache.shape[1] == decode_shape[1]:
            return cache
        cp = self.TS
        if self.transform_category == "T1":
            if self.kv_kind in (KVCacheKind.FA, KVCacheKind.MSA):
                cp = self.TS // cache.shape[1]
            return self._t1(cache, cp)
        if self.transform_category == "T2":
            return self._t2(cache, cp)
        if self.transform_category == "T3":
            return self._t3(cache, cp)
        raise ValueError(f"Unsupported KV transform {self.transform_category!r}")


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
