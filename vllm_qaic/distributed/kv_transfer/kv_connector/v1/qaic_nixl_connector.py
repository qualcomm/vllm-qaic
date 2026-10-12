# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""NIXL connector for the QAIC decode worker.

The NVIDIA worker exposes the KV cache in the layout required by its attention
backend (FlashAttention/FlashInfer): ``[blocks, 2, block, heads, head_dim]``.
QAIC's QPCs, on the other hand, consume one buffer per K/V tensor in
``[blocks, heads, block, head_dim]`` layout.  NIXL transfers into CPU staging
buffers and this connector performs the layout conversion after a receive has
completed.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm_qaic.logger import init_logger

logger = init_logger(__name__)

_BACKENDS = {"FLASH_ATTN", "FLASHINFER"}
_BACKEND_ENV = "VLLM_ATTENTION_BACKEND"


class _NvidiaAttentionBackend:
    """Minimal backend description needed by NIXL on a QAIC process.

    The actual attention implementation lives on the NVIDIA producer.  Keeping
    this description local avoids importing FlashInfer on a QAIC-only node.
    """

    _name = ""

    @classmethod
    def get_name(cls):
        return cls._name

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return tuple(range(6 if include_num_layers_dimension else 5))


class QaicNixlConnector(NixlConnector):
    """NIXL facade which translates NVIDIA KV pages to QAIC KV pages."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        backend = os.environ.get(_BACKEND_ENV, "").upper()
        if backend not in _BACKENDS:
            raise ValueError(
                f"{_BACKEND_ENV} must be one of {sorted(_BACKENDS)} for "
                f"QaicNixlConnector; got {backend!r}"
            )
        self.attention_backend_name = backend
        super().__init__(vllm_config, role, kv_cache_config)
        if self.connector_worker is not None:
            # The QAIC platform may report its own backend here.  NIXL's
            # topology must nevertheless describe the NVIDIA-side cache.
            backend = type(
                "QaicNixlBackend",
                (_NvidiaAttentionBackend,),
                {"_name": self.attention_backend_name},
            )
            self.connector_worker.attn_backends = [backend]
            self.connector_worker.backend_name = self.attention_backend_name
        self._qaic_kv_caches: dict[str, torch.Tensor] = {}
        self._nvidia_kv_caches: dict[str, torch.Tensor] = {}

    @staticmethod
    def _block_ids(block_ids: Any) -> list[int]:
        """Flatten the block-id representation used by the NIXL metadata."""
        if isinstance(block_ids, int):
            return [block_ids]
        if isinstance(block_ids, torch.Tensor):
            return [int(x) for x in block_ids.reshape(-1).tolist()]
        if isinstance(block_ids, (list, tuple)):
            result: list[int] = []
            for block_id in block_ids:
                result.extend(QaicNixlConnector._block_ids(block_id))
            return result
        return [int(block_ids)]

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register NVIDIA-shaped staging buffers with NIXL.

        ``kv_caches`` is the QAIC paged cache assembled by the model runner.
        The null page is retained in the staging tensor because vLLM's NIXL
        block count includes it; request block ids are physical ids.
        """
        if not kv_caches:
            raise ValueError("QaicNixlConnector received no QAIC KV caches")

        self._qaic_kv_caches = kv_caches
        staging: dict[str, torch.Tensor] = {}
        seen: dict[int, torch.Tensor] = {}
        for layer_name, qaic_cache in kv_caches.items():
            if qaic_cache.ndim != 5 or qaic_cache.shape[0] != 2:
                raise ValueError(
                    "QAIC KV cache must have shape [2, blocks, heads, block, "
                    f"head_dim], got {tuple(qaic_cache.shape)} for {layer_name}"
                )
            # Keep the null block in the NIXL registration as well. vLLM's
            # block ids are physical ids, with block zero reserved and data
            # blocks starting at one. NIXL validates this against
            # KVCacheConfig.num_blocks, which includes the null block.
            num_blocks = int(qaic_cache.shape[1])
            if num_blocks <= 1:
                raise ValueError("QAIC KV cache must contain a data block")
            key = id(qaic_cache)
            if key not in seen:
                shape = (
                    num_blocks,
                    2,
                    int(qaic_cache.shape[3]),
                    int(qaic_cache.shape[2]),
                    int(qaic_cache.shape[4]),
                )
                # CPU DRAM is intentional: it is the hand-off buffer between
                # the NVIDIA producer and the QAIC decode process.
                seen[key] = torch.empty(
                    shape, dtype=qaic_cache.dtype, device="cpu", pin_memory=False
                )
            staging[layer_name] = seen[key]

        self._nvidia_kv_caches = staging
        super().register_kv_caches(staging)
        logger.info(
            "Registered QaicNixlConnector with %d %s staging caches",
            len(staging),
            self.attention_backend_name,
        )

    def _convert_blocks(self, block_ids: Any) -> None:
        blocks = self._block_ids(block_ids)
        for layer_name, staging in self._nvidia_kv_caches.items():
            target = self._qaic_kv_caches[layer_name]
            for source_block in blocks:
                if source_block < 0 or source_block >= staging.shape[0]:
                    raise IndexError(
                        f"NIXL block {source_block} is outside staging cache "
                        f"of size {staging.shape[0]}"
                    )
                # Physical block ids already include QAIC/vLLM's null block.
                # Block zero is never populated by a request.
                target_block = source_block
                if target_block == 0:
                    continue
                target[0, target_block].copy_(
                    staging[source_block, 0].permute(1, 0, 2)
                )
                target[1, target_block].copy_(
                    staging[source_block, 1].permute(1, 0, 2)
                )

    def get_finished(
        self, finished_req_ids: set[str] | None = None
    ) -> tuple[set[str], set[str]]:
        # ``QaicModelRunnerAoT`` passes finished request ids for connector
        # implementations that need them. The upstream NIXL connector does
        # not consume this argument, so it is intentionally ignored here.
        # NixlConnectorWorker removes its receive metadata in get_finished(),
        # therefore capture block ids before delegating to the parent.
        recv_blocks: dict[str, Any] = {}
        metadata = getattr(self, "_connector_metadata", None)
        if metadata is not None:
            for req_id, req_meta in getattr(metadata, "reqs_to_recv", {}).items():
                recv_blocks[req_id] = getattr(
                    req_meta, "local_physical_block_ids", req_meta.local_block_ids
                )

        done_sending, done_recving = super().get_finished(finished_req_ids)
        for req_id in done_recving:
            if req_id in recv_blocks:
                self._convert_blocks(recv_blocks[req_id])
        return done_sending, done_recving
