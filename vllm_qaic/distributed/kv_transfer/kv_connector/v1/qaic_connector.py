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
import fcntl
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
from vllm_qaic.logger import init_logger
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
DFLASH_STREAM_PREFILL_ENV = "VLLM_QAIC_DFLASH_STREAM_PREFILL"
VLLM_QAIC_USE_FULL_KV_TRANSFER_ENV = "VLLM_QAIC_USE_FULL_KV_TRANSFER"
VLLM_QAIC_DISABLE_HANDOFF_ID_ENV = "VLLM_QAIC_DISABLE_HANDOFF_ID"


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
    req_id: str | None = None

    @staticmethod
    def make_meta(
        token_ids: list[int],
        is_store: bool,
        is_prefill_partial: bool,
        block_id: int,
        handoff_id: str | None = None,
        req_id: str | None = None,
    ) -> "ReqMeta":
        token_ids_tensor = torch.tensor(token_ids)
        return ReqMeta(
            token_ids=token_ids_tensor,
            token_hash=hash((hash(tuple(token_ids)), int(handoff_id, 16)))
            if handoff_id is not None
            else hash(tuple(token_ids)),
            is_store=is_store,
            is_prefill_partial=is_prefill_partial,
            block_id=block_id,
            req_id=req_id,
        )


@dataclass
class QaicConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]
    dflash_loads: list[ReqMeta]

    def __init__(self):
        self.requests = []
        self.dflash_loads = []

    def add_request(
        self,
        token_ids: list[int],
        is_store: bool,
        is_partial_prefill: bool,
        block_id: int,
        handoff_id: str | None = None,
        req_id: str | None = None,
    ) -> None:
        self.requests.append(
            ReqMeta.make_meta(
                token_ids, is_store, is_partial_prefill, block_id, handoff_id, req_id
            )
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

    def cleanup(self, unlink=False):
        if not self.cleanup_done:
            self.cleanup_done = True
            self.shared_memory.close()
            if not self.is_creator or unlink:
                self.shared_memory.unlink()
            del self.list_of_np_buff
            del self.shared_memory

    def __del__(self):
        self.cleanup()

    @contextmanager
    def publication_lock(self):
        # Only protects publication metadata, never device execution.
        fcntl.flock(self.shared_memory._fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self.shared_memory._fd, fcntl.LOCK_UN)

    @contextmanager
    def get_data(self, current_idx: int = 0):
        start = self.buff_sizes_cum_sum[current_idx - 1] if current_idx > 0 else 0
        end = self.buff_sizes_cum_sum[current_idx]
        assert self.shared_memory.buf is not None
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

    def release_Storage(self, name: str | None = None, unlink_creator: bool = False):
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
        if shm.is_creator and not unlink_creator:
            resource_tracker.unregister(shm.shared_memory._name, "shared_memory")
        shm.cleanup(unlink=unlink_creator)
        del self.MemBank[name]

    def cleanup(self):
        buff_name = list(self.MemBank.keys())
        for _name in buff_name:
            self.release_Storage(_name)


@dataclass
class DFlashLoad:
    request: ReqMeta
    position: int = 0
    next_lookup: float = 0
    reported: bool = False


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
        spec = vllm_config.speculative_config
        self.dflash_hybrid = bool(
            spec
            and spec.use_dflash()
            and vllm_config.model_config.hf_config.model_type
            in {"qwen3_5", "qwen3_5_moe"}
        )
        if self.dflash_hybrid:
            self.mem_bank.use_full_kv_transfer = True
        self.dflash_stream_prefill = bool(
            spec
            and spec.use_dflash()
            and os.getenv(DFLASH_STREAM_PREFILL_ENV, "0") == "1"
        )
        self._dflash_load_queue: dict[str, ReqMeta] = {}
        self._dflash_pending: dict[str, DFlashLoad] = {}
        self._dflash_loaded: dict[str, DFlashLoad] = {}
        self._dflash_ready: set[str] = set()
        self._dflash_producers: dict[int, ReqMeta] = {}
        self._dflash_cache_info = None
        self._dflash_consume = None
        self._dflash_cancel = None
        self._dflash_target_seq_len = (
            spec.num_speculative_tokens + 1 if self.dflash_stream_prefill else 0
        )

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
        self.disable_handoff_id = (
            os.getenv(VLLM_QAIC_DISABLE_HANDOFF_ID_ENV, "0") == "1"
        )

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
        if self.dflash_stream_prefill:
            for request in self._dflash_producers.values():
                if request.kv_handoff_metadata:
                    shm = self.mem_bank.MemBank.get(request.kv_handoff_metadata[0])
                    if shm is not None:
                        with shm.publication_lock():
                            shm.list_of_np_buff[-1][0, 1] = -1
            self._dflash_producers.clear()
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

    def get_kvcache_from_store(
        self, prompt_hash, wait=True
    ) -> QaicKvHandOffGetResp | None:
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
        max_retries = KV_LOOKUP_RETRIES if wait else 1
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
                    if not wait:
                        break
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
        if (
            wait
            and self.kv_role == "kv_consumer"
            and (retry_count >= max_retries or result is None)
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

    def register_dflash_stream(
        self, kv_cache_info, consume=None, cancel=None, target_seq_len=None
    ):
        self._dflash_cache_info = kv_cache_info
        self._dflash_consume = consume
        self._dflash_cancel = cancel
        if target_seq_len is not None:
            self._dflash_target_seq_len = target_seq_len

    def publish_dflash_chunk(self, slot, positions, hidden):
        request = self._dflash_producers[slot]
        shm = self.mem_bank.MemBank[request.kv_handoff_metadata[0]]
        valid = positions >= 0
        token_positions = positions[valid]
        start, end = int(token_positions[0]), int(token_positions[-1]) + 1
        if not np.array_equal(token_positions, np.arange(start, end)):
            raise ValueError("DFlash streaming requires contiguous prompt positions.")
        shm.list_of_np_buff[-2][0, 0, start:end] = hidden[0, valid]
        with shm.publication_lock():
            control = shm.list_of_np_buff[-1]
            if control[0, 0] != start:
                raise ValueError("DFlash prompt chunks arrived out of order.")
            control[0, 0] = end

    def _start_dflash_streams(self, metadata):
        for request in metadata.dflash_loads:
            self._dflash_pending.setdefault(request.req_id, DFlashLoad(request))
        for request in metadata.requests:
            slot = request.block_id - 1
            if self.is_producer:
                # A cancelled slot can be reassigned before get_finished runs.
                if (
                    slot in self._dflash_producers
                    and self._dflash_producers[slot].req_id != request.req_id
                ):
                    self.get_finished({self._dflash_producers[slot].req_id})
                if slot not in self._dflash_producers:
                    name, buffers = self.mem_bank.get_Storage(
                        self._dflash_cache_info, len(request.token_ids)
                    )
                    buffers[-1][:] = 0
                    self.kv_caches[slot] = buffers
                    request.kv_handoff_metadata = [name]
                    self._dflash_producers[slot] = request
                    self.send_kv_cache_to_store(
                        QaicKvHandOffPutReq(
                            buff_type=QaicBufferType.SHM,
                            timestamp=time.perf_counter(),
                            key_hash=request.token_hash,
                            rank=self.kv_rank,
                            payload=[name],
                            num_buff=1,
                        )
                    )
                else:
                    request.kv_handoff_metadata = self._dflash_producers[
                        slot
                    ].kv_handoff_metadata
            else:
                # A one-token prompt has no external tokens for an async load.
                if request.req_id not in self._dflash_loaded:
                    if len(request.token_ids) != 1:
                        raise RuntimeError(
                            "DFlash request scheduled before prefill completed."
                        )
                    self._dflash_pending.setdefault(request.req_id, DFlashLoad(request))
                    deadline = time.monotonic() + 120
                    while request.req_id not in self._dflash_loaded:
                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                "DFlash one-token prefill transfer timed out."
                            )
                        self._dflash_ready.update(
                            self._progress_dflash_streams() - {request.req_id}
                        )
                        time.sleep(0.001)
                loaded = self._dflash_loaded.pop(request.req_id)
                request.kv_handoff_metadata = loaded.request.kv_handoff_metadata

    def _progress_dflash_streams(self):
        received = set()
        ran_dlm = False
        for req_id, load in list(self._dflash_pending.items()):
            request = load.request
            if not request.kv_handoff_metadata:
                if time.monotonic() < load.next_lookup:
                    continue
                load.next_lookup = time.monotonic() + 0.005
                response = self.get_kvcache_from_store(request.token_hash, wait=False)
                if response is None:
                    continue
                name, _ = self.mem_bank.get_Storage(
                    self._dflash_cache_info,
                    len(request.token_ids),
                    name=response.payload[0],
                )
                request.kv_handoff_metadata = [name]
            shm = self.mem_bank.MemBank[request.kv_handoff_metadata[0]]
            with shm.publication_lock():
                ready, kv_ready = shm.list_of_np_buff[-1][0]
            if kv_ready < 0:
                # The proxy cancels the corresponding consumer request as well.
                continue
            block = self._dflash_target_seq_len
            final_start = (len(request.token_ids) - 1) // block * block
            if (
                not ran_dlm
                and load.position < final_start
                and ready >= load.position + block
            ):
                if load.position == 0:
                    logger.debug(
                        "DFlash stream %s starts DLM prefill: ready=%d/%d, kv_ready=%s",
                        req_id,
                        ready,
                        len(request.token_ids),
                        bool(kv_ready),
                    )
                features = shm.list_of_np_buff[-2][
                    0, 0, load.position : load.position + block
                ]
                self._dflash_consume(
                    req_id, request.block_id - 1, load.position, features
                )
                load.position += block
                ran_dlm = True
                # Rotate ready requests so long prompts cannot monopolize DLM.
                self._dflash_pending[req_id] = self._dflash_pending.pop(req_id)
            if (
                kv_ready
                and ready == len(request.token_ids) - int(self.dflash_hybrid)
                and load.position == final_start
            ):
                self.kv_caches[request.block_id - 1] = shm.list_of_np_buff
                self._dflash_loaded[req_id] = self._dflash_pending.pop(req_id)
                received.add(req_id)
        return received

    def get_finished(self, finished_req_ids):
        if not self.dflash_stream_prefill:
            return super().get_finished(finished_req_ids)
        if self.is_producer:
            for slot, request in list(self._dflash_producers.items()):
                if request.req_id in finished_req_ids:
                    name = request.kv_handoff_metadata[0]
                    shm = self.mem_bank.MemBank[name]
                    with shm.publication_lock():
                        shm.list_of_np_buff[-1][0, 1] = -1
                    unclaimed = self.get_kvcache_from_store(
                        request.token_hash, wait=False
                    )
                    self.mem_bank.release_Storage(
                        name, unlink_creator=unclaimed is not None
                    )
                    self.kv_caches[slot] = []
                    del self._dflash_producers[slot]
            return None, None
        received = self._dflash_ready
        self._dflash_ready = set()
        for req_id in finished_req_ids:
            load = self._dflash_pending.pop(req_id, None) or self._dflash_loaded.pop(
                req_id, None
            )
            if load is None:
                continue
            request = load.request
            if not request.kv_handoff_metadata:
                response = self.get_kvcache_from_store(request.token_hash, wait=False)
                if response is not None:
                    name, _ = self.mem_bank.get_Storage(
                        self._dflash_cache_info,
                        len(request.token_ids),
                        name=response.payload[0],
                    )
                    request.kv_handoff_metadata = [name]
            if request.kv_handoff_metadata:
                self.mem_bank.release_Storage(request.kv_handoff_metadata[0])
            self.kv_caches[request.block_id - 1] = []
            self._dflash_cancel(req_id)
            if not load.reported:
                received.add(req_id)
        received.update(self._progress_dflash_streams())
        for req_id in received:
            if req_id in self._dflash_loaded:
                self._dflash_loaded[req_id].reported = True
        return None, received or None

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

        if self.dflash_stream_prefill:
            forward_context.additional_kwargs["cleanup_callback"] = (
                self.cleanup_callback
            )
            self._start_dflash_streams(metadata)
            return

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
            if self.dflash_stream_prefill and self.is_producer:
                if request.is_prefill_partial:
                    continue
                shm = self.mem_bank.MemBank[request.kv_handoff_metadata[0]]
                with shm.publication_lock():
                    expected = len(request.token_ids) - int(self.dflash_hybrid)
                    if shm.list_of_np_buff[-1][0, 0] != expected:
                        raise RuntimeError(
                            "DFlash KV completed before prompt features were published."
                        )
                    shm.list_of_np_buff[-1][0, 1] = 1
                del self._dflash_producers[request.block_id - 1]
            # Partial prefill are not expected to have kv_handoff_metadata
            if request.kv_handoff_metadata:
                assert request.block_id is not None
                for buff in request.kv_handoff_metadata:
                    self.shm_tracker.put(buff)
                    # Cleanup registered KV cache block
                    self.kv_caches[request.block_id - 1] = []

            if request.is_store:
                if self.dflash_stream_prefill:
                    continue
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
        count = (request.num_prompt_tokens - 1) - num_computed_tokens
        return count, self.dflash_stream_prefill and count > 0

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        """
        Update KVConnector state after block allocation.

        If blocks were allocated, add to _request_tracker,
        such that we load the KVs in the next forward pass.
        """
        # Always load KV cache. In case of producer KV cache will be empty buffers
        if self.dflash_stream_prefill and not self.is_producer:
            if request.request_id not in self._request_tracker:
                block_id = blocks.get_block_ids()[0][0]
                self._request_tracker[request.request_id] = ReqTrackerObj(
                    request, block_id
                )
                if num_external_tokens:
                    self._dflash_load_queue[request.request_id] = ReqMeta.make_meta(
                        request.prompt_token_ids,
                        False,
                        False,
                        block_id,
                        self._handoff_id(request),
                        request.request_id,
                    )
        else:
            self._request_tracker[request.request_id] = ReqTrackerObj(request, None)

    def _handoff_id(self, request) -> str | None:
        if self.disable_handoff_id:
            return None
        params = request.kv_transfer_params or {}
        handoff_id = params.get("handoff_id")
        if handoff_id is not None:
            assert all(c in "0123456789abcdefABCDEF" for c in handoff_id), (
                f"handoff_id must be hex, got {handoff_id!r}"
            )
        return handoff_id

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
        if self.dflash_stream_prefill:
            for req_id in scheduler_output.finished_req_ids:
                self._request_tracker.pop(req_id, None)
                self._dflash_load_queue.pop(req_id, None)
            meta.dflash_loads = list(self._dflash_load_queue.values())
            self._dflash_load_queue.clear()

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
                    handoff_id=self._handoff_id(
                        self._request_tracker[new_req.req_id].request
                    ),
                    req_id=new_req.req_id,
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
                    handoff_id=self._handoff_id(self._request_tracker[req_id].request),
                    req_id=req_id,
                )  # For QAIC one request is mapped to only one block_id
                total_need_load += 1

                # If prefill is complete, remove the entry from the dict.
                if not is_partial_prefill:
                    self._request_tracker.pop(req_id, None)

        return meta
