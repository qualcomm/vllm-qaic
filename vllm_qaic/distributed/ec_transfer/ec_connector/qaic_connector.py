# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
import atexit
import contextlib
import gc
import random
import signal
import time
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np
import torch
import zmq
from qaic_disagg.ec_handoff.protocol import (
    QaicBufferType,
    QaicECHandOffGetReq,
    QaicECHandOffGetResp,
    QaicECHandOffPutReq,
    QaicECHandOffReqType,
)

from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm_qaic.distributed.kv_transfer.kv_connector.v1.qaic_connector import (
    QaicKVCacheBank as QaicECBank,
)
from vllm_qaic.logger import init_logger
from vllm.utils.network_utils import is_valid_ipv6_address, make_zmq_socket
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)
EC_LOOKUP_RETRIES = 5
EC_LOOKUP_RETRIES_INTERVAL = 0.05


@dataclass
class MMMeta:
    mm_hash: str

    @staticmethod
    def make_meta(mm_hash) -> "MMMeta":
        return MMMeta(mm_hash=mm_hash)


@dataclass
class QaicECConnectorMetadata(ECConnectorMetadata):
    mm_datas: list[MMMeta]

    def __init__(self):
        self.mm_datas = []

    def add_mm_data(self, mm_data: MMMeta):
        self.mm_datas.append(mm_data)


class QaicECConnector(ECConnectorBase):
    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        # req_id -> index
        self._mm_datas_need_loads: list[str] = []
        self.transfer_config = vllm_config.ec_transfer_config
        if self.transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")
        self.identity = str(uuid.uuid4()).encode("utf-8")
        self.ec_rank = self.transfer_config.ec_rank
        self.ec_ip = self.transfer_config.ec_ip
        self.ec_port = self.transfer_config.ec_port
        self.mem_bank = QaicECBank()
        self.mem_bank.use_full_kv_transfer = True
        # Preserve embedding shape verbatim (dim0 is tokens, not batch)
        self.mem_bank.ec_preserve_shape = True
        # Consumer-side map: mm_hash -> shm_name(s), for post-forward release
        self._loaded: dict[str, list[str]] = {}

        logger.info(
            "Initializing QaicECConnector under ec_transfer_config %s",
            self.transfer_config,
        )

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        if is_valid_ipv6_address(self.ec_ip):
            self.ec_ip = "[" + self.ec_ip + "]"
            self.ctx.setsockopt(zmq.IPV6, 1)

        ipc_path = f"tcp://{self.ec_ip}:{self.ec_port}"

        self.socket = make_zmq_socket(
            self.ctx, ipc_path, zmq.constants.DEALER, bind=False, identity=self.identity
        )
        self.decoder = MsgpackDecoder(QaicECHandOffGetResp)
        self.encoder = MsgpackEncoder(QaicECHandOffGetReq)
        self.encoder_send = MsgpackEncoder(QaicECHandOffPutReq)

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

    def _send_recv(self, cmd, body):
        self.socket.send_multipart((cmd, body), copy=False)
        return self.socket.recv_multipart(copy=False)

    # ==============================
    # Scheduler-side methods
    # ==============================

    def has_cache_item(self, identifier: str) -> bool:
        """
        Check if a single encoder cache exists

        Args:
            identifier (str): the identifier of the media.

        Returns:
            A bool where value is True if cache exist for
            the media
        """
        req_pkt = QaicECHandOffGetReq(
            timestamp=time.perf_counter(), mm_hash=identifier, rank=self.ec_rank
        )
        peek_req_pkt = self.encoder.encode(req_pkt)[0]
        try:
            (resp, _) = self._send_recv(
                cmd=QaicECHandOffReqType.PEEK.value, body=peek_req_pkt
            )
            resp = QaicECHandOffReqType(bytes(resp.buffer))
            return resp == QaicECHandOffReqType.RESP_OK
        except Exception as e:
            logger.warning("Unable to find item in ECStore due to an exception: %s", e)
            return False

    def update_state_after_alloc(self, request: "Request", index: int):
        """
        Update ECConnector state to decide allocate cache for requests

        Args:
            request (Request): the request object.
        """
        mm_hash = request.mm_features[index].identifier
        self._mm_datas_need_loads.append(mm_hash)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = QaicECConnectorMetadata()
        for mm_hash in self._mm_datas_need_loads:
            meta.add_mm_data(MMMeta.make_meta(mm_hash))
        self._mm_datas_need_loads.clear()
        return meta

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_caches(self, encoder_cache: dict[str, torch.Tensor], **kwargs) -> None:
        """
        Start loading the cache from the connector into vLLM's encoder cache.

        This method loads the encoder cache based on metadata provided by the scheduler.
        It is called before `_gather_mm_embeddings` for the EC Connector. For EC,
        the `encoder_cache` and `mm_hash` are stored in `kwargs`.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, QaicECConnectorMetadata)
        assert encoder_cache is not None
        if metadata is None:
            logger.warning(
                "In connector.start_load_caches, but the connector metadata is None"
            )
            return
        # Load the EC for each mm data
        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue

            result: QaicECHandOffGetResp = None
            req_pkt = QaicECHandOffGetReq(
                timestamp=time.perf_counter(),
                mm_hash=mm_data.mm_hash,
                rank=self.ec_rank,
            )
            get_req_pkt = self.encoder.encode(req_pkt)[0]
            max_retries = EC_LOOKUP_RETRIES
            retry_count = 0
            while retry_count < max_retries:
                try:
                    (resp, resp_payload) = self._send_recv(
                        QaicECHandOffReqType.GET.value, get_req_pkt
                    )
                    resp = QaicECHandOffReqType(bytes(resp.buffer))
                    if resp == QaicECHandOffReqType.RESP_OK:
                        result = self.decoder.decode(resp_payload)
                        break
                    if resp == QaicECHandOffReqType.RESP_NOT_FOUND:
                        retry_count += 1
                        time.sleep(EC_LOOKUP_RETRIES_INTERVAL)
                        logger.info("ECStore RESP_NOT_FOUND received for %s", req_pkt)
                        continue
                    elif resp == QaicECHandOffReqType.RESP_ERROR:
                        logger.debug("ECStore RESP_ERROR received for %s", req_pkt)
                        retry_count = max_retries
                    else:
                        raise ValueError(f"Invalid response from ECStore: {resp}")
                except Exception as e:
                    raise ValueError(
                        f"Unable to access ECStore due to an exception: {e}"
                    ) from e
            if not self.is_producer and (retry_count >= max_retries or result is None):
                raise ValueError(
                    f"Unable to find prompt hash {req_pkt.mm_hash} in ECStore!"
                )
            try:
                (_, buffs) = self.mem_bank.get_Storage(
                    kv_cache_info=[
                        (tuple(result.shape[0]), np.dtype(result.dtype[0]), None)
                    ],
                    num_tokens=1,
                    name=result.payload[0],
                )
                encoder_cache[mm_data.mm_hash] = torch.from_numpy(buffs[0])
            except Exception as e:
                # Attach/wrap failed: release the segment we opened (non-creator
                # -> close()+unlink()) so it is not leaked, then surface the error.
                with contextlib.suppress(Exception):
                    self.mem_bank.release_Storage(result.payload[0])
                raise ValueError(
                    f"Failed to load EC SHM for {mm_data.mm_hash}: {e}"
                ) from e
            # Track the SHM segment so it can be released after the model
            # is done with this mm_hash (see free_caches, driven by the
            # scheduler's free_encoder_mm_hashes in QaicModelRunner._update_states).
            self._loaded[mm_data.mm_hash] = list(result.payload)

    def free_caches(self, mm_hashes) -> None:
        """
        Release SHM segments for mm_hashes the model is done with (consumer side).

        Called from QaicModelRunner._update_states AFTER the encoder_cache entry
        (a torch view over the SHM buffer) has been popped, so it is safe to unlink.
        """
        if self.is_producer:
            return
        for mm_hash in mm_hashes:
            names = self._loaded.pop(mm_hash, None)
            if not names:
                continue
            for name in names:
                try:
                    self.mem_bank.release_Storage(name, unlink=False)
                except Exception as e:
                    logger.warning(
                        "Failed to release EC SHM %s for mm_hash %s: %s",
                        name,
                        mm_hash,
                        e,
                    )
            req_pkt = QaicECHandOffGetReq(
                timestamp=time.perf_counter(), mm_hash=mm_hash, rank=self.ec_rank
            )
            unpin_req_pkt = self.encoder.encode(req_pkt)[0]
            try:
                (resp, _) = self._send_recv(
                    cmd=QaicECHandOffReqType.UNPIN.value, body=unpin_req_pkt
                )
                _ = QaicECHandOffReqType(bytes(resp.buffer))
            except Exception as e:
                logger.warning(
                    "Unable to unpin item in ECStore due to an exception: %s", e
                )

    def save_caches(
        self, encoder_cache: dict[str, torch.Tensor], mm_hash: str, **kwargs
    ) -> None:
        """
        Save the encoder cache to the connector.

        This method saves the encoder cache from the worker's local storage
        to shared storage or another external connector.

        Args:
            encoder_cache (dict[str, torch.Tensor]): A dictionary mapping multimodal
                data hashes (`mm_hash`) to encoder cache tensors.
            mm_hash (str): The hash of the multimodal data whose cache is being saved.
            kwargs (dict): Additional keyword arguments for the connector.
        """
        # Return if it is PD Instance
        if not self.is_producer:
            return
        emb = encoder_cache[mm_hash].cpu().contiguous().numpy()
        assert isinstance(emb, np.ndarray), (
            "Expected numpy array for multimodal embedding"
        )
        name, buffs = self.mem_bank.get_Storage(
            kv_cache_info=[(emb.shape, emb.dtype, None)], num_tokens=1, name=None
        )
        try:
            np.copyto(buffs[0], emb)
        except Exception as e:
            # Alloc succeeded but fill failed and the segment has NOT been PUT
            # to the store yet, so nothing else references it. Reclaim it now
            # to avoid leaking the /dev/shm segment. The creator path of
            # release_Storage only unregisters+close (no unlink), so unlink
            # explicitly here since we are abandoning this segment entirely.
            try:
                self.mem_bank.release_Storage(name)
                shared_memory.SharedMemory(name=name).unlink()
            except FileNotFoundError:
                pass
            except Exception as rel_e:
                logger.warning(
                    "Failed to reclaim EC SHM %s after copy failure: %s", name, rel_e
                )
            raise ValueError(
                f"Failed to copy embedding into EC SHM for {mm_hash}: {e}"
            ) from e
        req_pkt = QaicECHandOffPutReq(
            buff_type=QaicBufferType.SHM,
            timestamp=time.perf_counter(),
            mm_hash=mm_hash,
            rank=self.ec_rank,
            num_buff=1,
            payload=[name],
            shape=[list(emb.shape)],
            dtype=[str(emb.dtype)],
        )
        put_req_pkt = self.encoder_send.encode(req_pkt)[0]
        try:
            resp = QaicECHandOffReqType.RESP_BUFFER_FULL
            retries_cnt = 0
            while resp == QaicECHandOffReqType.RESP_BUFFER_FULL:
                (resp, _) = self._send_recv(QaicECHandOffReqType.PUT.value, put_req_pkt)
                resp = QaicECHandOffReqType(bytes(resp.buffer))
                if resp == QaicECHandOffReqType.RESP_BUFFER_FULL:
                    time.sleep(random.randint(1, 10) / 1000)
                    retries_cnt += 1
                if retries_cnt > EC_LOOKUP_RETRIES:
                    retries_cnt = 0
                    logger.warning("ECStore is full...")
                    # Trigger garbage collection
                    gc.collect()
                    time.sleep(random.randint(20, 100) / 1000)
        except Exception as e:
            raise ValueError(f"Unable to access ECStore due to an exception: {e}") from e
        else:
            # Detach the producer's own handle now that the name is PUT to the
            # store. Creator path: resource_tracker.unregister + close(), NO
            # unlink -> segment stays alive for the consumer; store/consumer
            # owns the eventual unlink.
            try:
                self.mem_bank.release_Storage(name)
            except Exception as e:
                logger.warning("Failed to detach producer EC SHM %s: %s", name, e)
