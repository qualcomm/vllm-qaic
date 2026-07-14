# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""
QaicCache KV Cache Connector for Disaggregated Serving

The QaicConnector can transfer KV caches between prefill vLLM worker
(KV cache producer) and decode vLLM worker (KV cache consumer) using separate process
for book keeping, to support xPyD via shared memory access between processes with
KV caching support.

General Design pointers:
1. All requests will have KV load; Prefill will create empty KV buffer,
   Decode will load populated KV buffers.
2. KV store only for kv_producer.

Shared Memory Naming:
- Format: psm_{uuid16}_{counter}_{pid}
  - uuid16: 16-character hexadecimal UUID4 instance identifier (collision-resistant)
  - counter: Decimal counter (0-999999) for per-instance buffer ordering
  - pid: Hexadecimal process ID (without 0x prefix) for bare metal identification
- Example: psm_a3f9c2b1e4d78956_42_4d2
- UUID ensures uniqueness across containers, pods, and bare metal deployments
- Works correctly with Docker --network=host (no hostname dependency)
"""

import atexit
import gc
import os
import random
import signal
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from queue import Empty, Queue
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import zmq
from qaic_disagg.kv_handoff.protocol import (
    QaicBufferType,
    QaicKvHandOffGetReq,
    QaicKvHandOffGetResp,
    QaicKvHandOffPutReq,
    QaicKvHandOffReqType,
)
from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import is_valid_ipv6_address, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)
KV_LOOKUP_RETRIES = 5
KV_LOOKUP_RETRIES_INTERVAL = 0.05
FORCE_CLEAN_UP_MULTIPLIER = 2
MAX_UID = 1000000
VLLM_QAIC_USE_FULL_KV_TRANSFER_ENV = "VLLM_QAIC_USE_FULL_KV_TRANSFER"


@dataclass
class ReqTrackerObj:
    request: "Request"
    block_id: int | None = None


@dataclass
class ReqMeta:
    # Request tokens
    token_ids: torch.Tensor
    token_hash: int
    # Is store or load
    is_store: bool
    # Is prefill partial
    is_prefill_partial: bool
    # Block Id of the request
    block_id: int | None = None
    # Keeping List[str] for backward compatibility with QAIC handoff server;
    # even though will store only KV shm name.
    kv_handoff_metadata: list[str] | None = None

    @staticmethod
    def make_meta(
        token_ids: list[int], is_store: bool, is_prefill_partial: bool, block_id: int
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        return ReqMeta(
            token_ids=token_ids_tensor,
            token_hash=hash(tuple(token_ids)),
            is_store=is_store,
            is_prefill_partial=is_prefill_partial,
            block_id=block_id,
        )


@dataclass
class QaicConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]

    def __init__(self):
        self.requests = []

    def add_request(
        self,
        token_ids: list[int],
        is_store: bool,
        is_partial_prefill: bool,
        block_id: int,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(token_ids, is_store, is_partial_prefill, block_id)
        )


class ShmBuffer:
    def __init__(
        self,
        kv_cache_info: list[tuple],
        total_bytes_of_buffer: int | None = None,
        num_tokens: int = 1,
        name: str | None = None,
        create: bool = True,
        use_full_kv_transfer: bool = False,
    ):
        """
        A shared memory buffer implementation, wrapper over shared_memory.SharedMemory
        During creation, `name` is None and the buffer is created. We can pass the
        created object to other processes by pickling it. The other processes will
        get the name of the shared memory and open it, so that they can access the
        same shared memory buffer.
        """  # noqa
        self.kv_cache_info = kv_cache_info
        self.name = name
        self.list_of_np_buff = []
        self.cleanup_done = False
        if not isinstance(kv_cache_info, list):
            raise ValueError("shape must be a list")

        self.buff_sizes = []
        for kv_shape, kv_type, _ in kv_cache_info:
            _kv_shape = self._get_kv_shape(use_full_kv_transfer, kv_shape, num_tokens)
            _bytes_of_buffer = 1
            for dim in _kv_shape:
                _bytes_of_buffer *= dim
            if kv_type == np.float16:
                _bytes_of_buffer *= 2
            elif kv_type in [np.float32, np.int32]:
                _bytes_of_buffer *= 4
            elif kv_type in [np.float64, np.int64]:
                _bytes_of_buffer *= 8
            self.buff_sizes.append(_bytes_of_buffer)

        if total_bytes_of_buffer:
            self.total_bytes_of_buffer = total_bytes_of_buffer
        else:
            self.total_bytes_of_buffer = sum(self.buff_sizes)

        self.buff_sizes_cum_sum = np.array(self.buff_sizes).cumsum()
        self.num_buffs = len(kv_cache_info)
        self.is_creator = create
        # print ("Total bytes of buffer: ", self.total_bytes_of_buffer)
        if self.is_creator is True:
            try:
                self.shared_memory = shared_memory.SharedMemory(
                    name=name, create=True, size=self.total_bytes_of_buffer
                )
                self.name = self.shared_memory.name
            except Exception as e:
                raise ValueError(
                    "Exception occurred during creation of shared memory!"
                ) from e
        else:
            # we are opening an existing buffer
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            # with patch("multiprocessing.resource_tracker.register",
            #            lambda *args, **kwargs: None):
            try:
                self.shared_memory = shared_memory.SharedMemory(name=name)
                # See https://docs.python.org/3/library/multiprocessing.shared_memory.html # noqa
                # Some platforms allocate memory based on page size,
                # so the shared memory block size may be larger or equal
                # to the requested size. The size parameter is ignored
                # when attaching to an existing block.
                assert self.shared_memory.size >= self.total_bytes_of_buffer
            except FileNotFoundError:
                raise ValueError("Shared memory buffer not found") from None
            except Exception as e:
                raise ValueError("Shared memory buffer not found") from e

        # create list of numpy arrays
        for i, (kv_shape, kv_type, _) in enumerate(self.kv_cache_info):
            with self.get_data(i) as buff:
                _kv_shape = self._get_kv_shape(
                    use_full_kv_transfer, kv_shape, num_tokens
                )
                self.list_of_np_buff.append(
                    np.ndarray(_kv_shape, dtype=kv_type, buffer=buff)
                )

    def _get_kv_shape(
        self, use_full_kv_transfer: bool, kv_shape: tuple, num_tokens: int
    ) -> tuple:
        if not use_full_kv_transfer and len(kv_shape) > 3:
            seq_len = min(num_tokens, kv_shape[2])
            return (1, kv_shape[1], seq_len) + kv_shape[3:]
        # Use batch size 1 and keep other dims intact
        return (1,) + kv_shape[1:]

    def cleanup(self):
        if not self.cleanup_done:
            self.cleanup_done = True
            self.shared_memory.close()
            if not self.is_creator:
                self.shared_memory.unlink()
            del self.list_of_np_buff
            del self.shared_memory

    def __del__(self):
        self.cleanup()

    @contextmanager
    def get_data(self, current_idx: int = 0):
        start = self.buff_sizes_cum_sum[current_idx - 1] if current_idx > 0 else 0
        end = self.buff_sizes_cum_sum[current_idx]
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf


class QaicKVCacheBank:
    def __init__(self):
        self.MemBank = dict()
        self.uid = 0
        self.pid = hex(os.getpid())[2:]
        # Generate UUID-based instance identifier for collision-resistant naming
        self.instance_id = uuid.uuid4().hex[2:16]
        # Log UUID for startup verification and correlation
        logger.info("[QAIC Connector] instance_id=%s", self.instance_id)
        self.use_full_kv_transfer = (
            os.getenv(VLLM_QAIC_USE_FULL_KV_TRANSFER_ENV, "0") == "1"
        )

    def get_Storage(
        self, kv_cache_info: list, num_tokens: int, name: str | None = None
    ):
        if name is None:
            self.uid = (self.uid + 1) % MAX_UID
            # Format: psm_{uuid16}_{counter}_{pid} (all identifiers for uniqueness)
            name = f"psm_{self.instance_id}_{hex(self.uid)[2:]}_{self.pid}"
            create = True
        else:
            create = False
        shm = ShmBuffer(
            kv_cache_info=kv_cache_info,
            num_tokens=num_tokens,
            name=name,
            create=create,
            use_full_kv_transfer=self.use_full_kv_transfer,
        )
        self.MemBank[shm.name] = shm
        return shm.name, shm.list_of_np_buff

    def release_Storage(self, name: str | None = None):
        """
        Release a shared memory (SHM) segment from the memory bank.

        Behavior:
        - If invoked by the SHM creator (e.g., during prefill), the segment
          is removed from the `resource_tracker`. In this case, no actual
          cleanup occurs because the resource tracker assumes the creator
          will manage its lifecycle.
        - If invoked during decode (or by a non-creator), the SHM is
          explicitly unlinked, ensuring proper cleanup.

        Notes:
        - The `resource_tracker` automatically deletes SHM segments in the
          background when no processes are associated with them.
        - Unlinking removes the SHM name from the namespace, but the memory
          is only freed once all references are closed.

        Args:
            name (Optional[str]): The name of the SHM segment to release.
                If None, defaults to the internally tracked segment name.

        Returns:
            None
        """
        assert name in self.MemBank, f"Storage {name} not found in MemBank"
        shm = self.MemBank[name]
        if shm.is_creator:
            resource_tracker.unregister(shm.shared_memory._name, "shared_memory")
        shm.cleanup()
        del self.MemBank[name]

    def cleanup(self):
        buff_name = list(self.MemBank.keys())
        for _name in buff_name:
            self.release_Storage(_name)


class QaicConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.transfer_config: KVTransferConfig = vllm_config.kv_transfer_config
        self.kv_role: str | None = vllm_config.kv_transfer_config.kv_role
        self.vllm_config: VllmConfig = vllm_config
        self.identity = str(uuid.uuid4()).encode("utf-8")
        self.kv_rank = vllm_config.kv_transfer_config.kv_rank
        self.kv_port = vllm_config.kv_transfer_config.kv_port
        self.is_producer = self.kv_role == "kv_producer"
        self.mem_bank = QaicKVCacheBank()
        self.shm_tracker: Queue = Queue()
        self.force_clean_up_threshold = (
            FORCE_CLEAN_UP_MULTIPLIER * vllm_config.scheduler_config.max_num_seqs
        )
        self.kv_caches: list[list] = []
        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling
        self.is_async_kv_producer: bool = self.use_async_scheduling and self.is_producer

        logger.info(
            "Initializing QaicConnector under kv_transfer_config %s",
            self.transfer_config,
        )

        # KV_both mode is not supported yet
        assert self.kv_role != "kv_both", "KV_BOTH mode is not supported yet"

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        kv_ip = self.transfer_config.kv_ip
        if is_valid_ipv6_address(kv_ip):
            kv_ip = "[" + kv_ip + "]"
            self.ctx.setsockopt(zmq.IPV6, 1)

        ipc_path = f"tcp://{kv_ip}:{self.transfer_config.kv_port}"

        self.socket = make_zmq_socket(
            self.ctx, ipc_path, zmq.constants.DEALER, bind=False, identity=self.identity
        )
        self.decoder = MsgpackDecoder(QaicKvHandOffGetResp)
        self.encoder = MsgpackEncoder()
        self.encoder_send = MsgpackEncoder(QaicKvHandOffPutReq)

        # Request tracker for scheduler for each step
        self._request_tracker: dict[str, ReqTrackerObj] = {}

        # Invoke Threads
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.close)

    # ==============================
    # Helper methods
    # ==============================

    def signal_handler(self, signum, frame):
        logger.info("Received signal %s, exiting...", signum)
        self.close()

    def close(self):
        self.mem_bank.cleanup()
        if hasattr(self, "ctx") and self.ctx:
            self.ctx.destroy(linger=10)

    def send_kv_cache_to_store(self, pkt: QaicKvHandOffPutReq):
        """Send KV cache to external connector"""
        put_req = self.encoder_send.encode(pkt)
        try:
            resp = QaicKvHandOffReqType.RESP_BUFFER_FULL
            retries_cnt = 0
            while resp == QaicKvHandOffReqType.RESP_BUFFER_FULL:
                self.socket.send_multipart(
                    (QaicKvHandOffReqType.PUT.value, put_req[0]), copy=False
                )
                (resp, _) = self.socket.recv_multipart(copy=False)
                resp = QaicKvHandOffReqType(bytes(resp.buffer))
                if resp == QaicKvHandOffReqType.RESP_BUFFER_FULL:
                    time.sleep(random.randint(1, 10) / 1000)
                    retries_cnt += 1
                if retries_cnt > KV_LOOKUP_RETRIES:
                    retries_cnt = 0
                    logger.warning("KV Handoff Storage Full...")
                    # Trigger garbage collection
                    gc.collect()
                    time.sleep(random.randint(20, 100) / 1000)
        except Exception as e:
            raise ValueError(
                f"Unable to access KV store due to an exception: {e}"
            ) from e

    def get_kvcache_from_store(self, prompt_hash) -> QaicKvHandOffGetResp | None:
        """Get kv cache from kv_store."""
        result = None
        # Get kv cache from kv_store
        req_pkt = QaicKvHandOffGetReq(
            buff_type=0,
            timestamp=time.perf_counter(),
            key_hash=prompt_hash,
            rank=self.kv_rank,
        )
        encode_req_pkt = self.encoder.encode(req_pkt)[0]
        max_retries = KV_LOOKUP_RETRIES
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Send request to KV store
                self.socket.send_multipart(
                    (QaicKvHandOffReqType.GET.value, encode_req_pkt)
                )
                (resp, resp_payload) = self.socket.recv_multipart(copy=False)
                resp = QaicKvHandOffReqType(bytes(resp.buffer))
                if resp == QaicKvHandOffReqType.RESP_OK:
                    result = self.decoder.decode(resp_payload)
                    break
                if resp == QaicKvHandOffReqType.RESP_NOT_FOUND:
                    retry_count += 1
                    time.sleep(KV_LOOKUP_RETRIES_INTERVAL)
                    print(f"KV Resp Not Found Received for {req_pkt}")
                    logger.debug("KV Resp Not Found Received for %s", req_pkt)
                    continue
                elif resp == QaicKvHandOffReqType.RESP_ERROR:
                    logger.debug("KV Resp Error Received for %s", req_pkt)
                    retry_count = max_retries
                else:
                    raise ValueError(f"Invalid response from KV store: {resp}")
            except Exception as e:
                raise ValueError(
                    f"Unable to access KV store due to an exception: {e}"
                ) from e
        if self.kv_role == "kv_consumer" and (
            retry_count >= max_retries or result is None
        ):
            raise ValueError(
                f"Unable to find prompt hash {req_pkt.key_hash} in KV store!"
            )
        return result

    def cleanup_callback(self, max_count=1):
        """Cleanup callback for kv_store."""
        for _ in range(max_count):
            try:
                buff = self.shm_tracker.get_nowait()
                self.mem_bank.release_Storage(buff)
            except Empty:
                break

    # ==============================
    # Worker-side methods
    # ==============================

    def register_kv_caches(self, kv_caches: list):
        logger.info("Registering KV caches")
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """

        # Get the metadata
        metadata: KVConnectorMetadata = self._get_connector_metadata()

        if metadata is None:
            logger.warning(
                "In connector.start_load_kv, but the connector metadata is None"
            )
            return
        assert isinstance(metadata, QaicConnectorMetadata)

        if "kv_cache_info" not in kwargs:
            logger.debug("In connector.start_load_kv, but the kv_cache_info is None")
            return
        kv_cache_info = kwargs["kv_cache_info"]

        forward_context.additional_kwargs["cleanup_callback"] = self.cleanup_callback
        # Load the KV for each request
        for request in metadata.requests:
            if request.is_prefill_partial and self.is_producer:
                continue
            elif request.block_id is not None:
                kv_shm_buff_name = None
                # Get kv cache from kv_store
                if not self.is_producer:
                    resp = self.get_kvcache_from_store(request.token_hash)
                    assert resp is not None
                    assert resp.buff_type == 0, (
                        "Raw np.ndarray KV exchange not supported yet"
                    )
                    kv_shm_buff_name = resp.payload[0]
                kv_storage_shm_name, kv_buff = self.mem_bank.get_Storage(
                    kv_cache_info=kv_cache_info,
                    num_tokens=len(request.token_ids),
                    name=kv_shm_buff_name,
                )
                request.kv_handoff_metadata = [kv_storage_shm_name]
                if self.kv_caches[request.block_id - 1]:
                    logger.warning(
                        "Overwriting KV cache for running request at"
                        " block_id=%s; this is unexpected.",
                        request.block_id,
                    )
                self.kv_caches[request.block_id - 1] = kv_buff
            else:
                raise ValueError(f"Block ID not found for request {request.token_hash}")

    def wait_for_layer_load(self, layer_name):
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.
        NOTE: Currently KV cache transfer is not managed layer by layer.

        Args:
            layer_name: the name of that layer
        """
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        """Blocking call to save the KV cache of the layer to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """

        if (
            not self.is_producer
            and self.shm_tracker.qsize() > self.force_clean_up_threshold
        ):
            # Force cleanup if average GL is less than batch size
            logger.info(
                "Forcing clean_up of %s shm buffers,"
                " as average GL is less than batch size",
                self.shm_tracker.qsize(),
            )
            self.cleanup_callback(self.shm_tracker.qsize())

        assert not self.is_async_kv_producer or "connector_metadata" in kwargs, (
            "connector_metadata missing in kwargs for async kv producer"
        )
        connector_metadata = (
            self._get_connector_metadata()
            if "connector_metadata" not in kwargs
            else kwargs["connector_metadata"]
        )

        assert isinstance(connector_metadata, QaicConnectorMetadata)
        for request in connector_metadata.requests:
            # Partial prefill are not expected to have kv_handoff_metadata
            if request.kv_handoff_metadata:
                assert request.block_id is not None
                for buff in request.kv_handoff_metadata:
                    self.shm_tracker.put(buff)
                    # Cleanup registered KV cache block
                    self.kv_caches[request.block_id - 1] = []

            if request.is_store:
                assert request.kv_handoff_metadata is not None, (
                    "KV Handoff metadata is needed to save KV to connector"
                )

                req_pk = QaicKvHandOffPutReq(
                    buff_type=QaicBufferType.SHM,
                    timestamp=time.perf_counter(),
                    key_hash=request.token_hash,
                    rank=self.kv_rank,
                    payload=request.kv_handoff_metadata,
                    num_buff=1,
                )
                self.send_kv_cache_to_store(req_pk)
        return

    def wait_for_save(self, **kwargs):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        NOTE: Currently aysc KV cache transfer is not supported.
        """
        return

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # NOTE: We assume KV store will not have tokens required by producer.
        # This will change is case of KV sharing
        if self.is_producer:
            return 0, False

        # NOTE: We assume KV store will always have tokens ready in case of consumer.
        return (request.num_prompt_tokens - 1) - num_computed_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _request_tracker,
        such that we load the KVs in the next forward pass.
        """
        # Always load KV cache. In case of producer KV cache will be empty buffers
        self._request_tracker[request.request_id] = ReqTrackerObj(request, None)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        """Build the connector metadata for this step.

        This function should NOT modify any fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = QaicConnectorMetadata()

        total_need_load = 0
        for new_req in scheduler_output.scheduled_new_reqs:
            total_prefill_tokens = (
                scheduler_output.num_scheduled_tokens[new_req.req_id]
                + new_req.num_computed_tokens
            )
            if new_req.req_id in self._request_tracker:
                is_partial_prefill = total_prefill_tokens != len(
                    new_req.prompt_token_ids
                )
                block_id = new_req.block_ids[0][0]
                meta.add_request(
                    token_ids=new_req.prompt_token_ids,
                    is_store=self.is_producer and not is_partial_prefill,
                    is_partial_prefill=is_partial_prefill,
                    block_id=block_id,
                )
                self._request_tracker[new_req.req_id].block_id = block_id
                total_need_load += 1

                # If prefill is complete, remove the entry from the dict.
                if not is_partial_prefill:
                    self._request_tracker.pop(new_req.req_id, None)

        # Create metadata for cached requests, including partial prefill requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            total_prefill_tokens = (
                scheduler_output.num_scheduled_tokens[req_id]
                + cached_reqs.num_computed_tokens[i]
            )
            if req_id in self._request_tracker:
                is_partial_prefill = (
                    total_prefill_tokens
                    != self._request_tracker[req_id].request.num_prompt_tokens
                )
                cached_block_id = self._request_tracker[req_id].block_id
                assert cached_block_id is not None
                meta.add_request(
                    token_ids=self._request_tracker[req_id].request.prompt_token_ids,
                    is_store=self.is_producer and not is_partial_prefill,
                    is_partial_prefill=is_partial_prefill,
                    block_id=cached_block_id,
                )  # For QAIC one request is mapped to only one block_id
                total_need_load += 1

                # If prefill is complete, remove the entry from the dict.
                if not is_partial_prefill:
                    self._request_tracker.pop(req_id, None)

        return meta
