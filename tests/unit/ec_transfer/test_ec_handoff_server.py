# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Host-only tests for the qaic_disagg EC hand-off server over ZMQ."""

import contextlib
import socket
import threading
import time
import uuid
from multiprocessing import shared_memory

import pytest
import zmq

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm_qaic.distributed.ec_transfer.ec_connector.protocol import (
    QaicBufferType,
    QaicECHandOffGetReq,
    QaicECHandOffGetResp,
    QaicECHandOffPutReq,
    QaicECHandOffReqType,
)

ec_server = pytest.importorskip("qaic_disagg.ec_handoff.server")

HOST = "127.0.0.1"
ENTRY_SHAPE = [8, 16]
ENTRY_BYTES = 8 * 16 * 2


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


@pytest.fixture
def server():
    port = _free_port()
    instance = ec_server.QaicECCacheServer(HOST, port, ttl=3600.0, max_bytes=1024**3)
    thread = threading.Thread(
        target=instance.process_client_request,
        args=(f"tcp://{HOST}:{port}",),
        daemon=True,
    )
    thread.start()
    time.sleep(0.5)
    yield instance, port


@pytest.fixture
def client(server):
    _, port = server
    ctx = zmq.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.setsockopt(zmq.IDENTITY, uuid.uuid4().bytes)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"tcp://{HOST}:{port}")
    put_encoder = MsgpackEncoder(QaicECHandOffPutReq)
    get_encoder = MsgpackEncoder(QaicECHandOffGetReq)
    resp_decoder = MsgpackDecoder(QaicECHandOffGetResp)

    def send(cmd, body):
        sock.send_multipart((cmd.value, body))
        resp, payload = sock.recv_multipart()
        return QaicECHandOffReqType(resp), payload

    def put(mm_hash, name):
        req = QaicECHandOffPutReq(
            buff_type=QaicBufferType.SHM,
            timestamp=time.perf_counter(),
            mm_hash=mm_hash,
            rank=0,
            num_buff=1,
            payload=[name],
            shape=[ENTRY_SHAPE],
            dtype=["float16"],
        )
        return send(QaicECHandOffReqType.PUT, put_encoder.encode(req)[0])

    def request(cmd, mm_hash):
        req = QaicECHandOffGetReq(
            timestamp=time.perf_counter(), mm_hash=mm_hash, rank=1
        )
        return send(cmd, get_encoder.encode(req)[0])

    api = type("Client", (), {})()
    api.put = put
    api.get = lambda h: request(QaicECHandOffReqType.GET, h)
    api.peek = lambda h: request(QaicECHandOffReqType.PEEK, h)
    api.unpin = lambda h: request(QaicECHandOffReqType.UNPIN, h)
    api.send = send
    api.decode = resp_decoder.decode
    yield api
    sock.close(linger=0)
    ctx.term()


@pytest.fixture
def segment():
    shm = shared_memory.SharedMemory(create=True, size=ENTRY_BYTES)
    yield shm
    shm.close()
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()


def test_put_then_get_returns_location(client, segment):
    resp, _ = client.put("h", segment.name)
    assert resp == QaicECHandOffReqType.RESP_OK

    resp, payload = client.get("h")
    assert resp == QaicECHandOffReqType.RESP_OK
    decoded = client.decode(payload)
    assert decoded.payload == [segment.name]
    assert decoded.shape == [ENTRY_SHAPE]
    assert decoded.dtype == ["float16"]


def test_get_and_peek_missing_hash(client):
    assert client.get("missing")[0] == QaicECHandOffReqType.RESP_NOT_FOUND
    assert client.peek("missing")[0] == QaicECHandOffReqType.RESP_NOT_FOUND


def test_peek_does_not_pin_and_get_pins(server, client, segment):
    instance, _ = server
    client.put("h", segment.name)
    assert client.peek("h")[0] == QaicECHandOffReqType.RESP_OK
    assert instance.data_store.peek("h").ref_count == 0
    client.get("h")
    assert instance.data_store.peek("h").ref_count == 1
    assert client.unpin("h")[0] == QaicECHandOffReqType.RESP_OK
    assert instance.data_store.peek("h").ref_count == 0


def test_entry_survives_get_for_reuse(client, segment):
    client.put("h", segment.name)
    client.get("h")
    client.unpin("h")
    assert client.peek("h")[0] == QaicECHandOffReqType.RESP_OK
    assert client.get("h")[0] == QaicECHandOffReqType.RESP_OK


def test_unpin_missing_hash_is_ok(client):
    assert client.unpin("missing")[0] == QaicECHandOffReqType.RESP_OK


def test_put_returns_buffer_full_when_pinned(server, client, segment):
    instance, _ = server
    instance.data_store.max_bytes = ENTRY_BYTES
    client.put("p", segment.name)
    client.get("p")
    extra = shared_memory.SharedMemory(create=True, size=ENTRY_BYTES)
    try:
        resp, _ = client.put("q", extra.name)
        assert resp == QaicECHandOffReqType.RESP_BUFFER_FULL
    finally:
        extra.close()
        with contextlib.suppress(FileNotFoundError):
            extra.unlink()


def test_malformed_put_returns_error(client):
    resp, _ = client.send(QaicECHandOffReqType.PUT, b"not-msgpack")
    assert resp == QaicECHandOffReqType.RESP_ERROR


def test_unknown_command_returns_invalid(client):
    resp, _ = client.send(QaicECHandOffReqType.RESP_OK, b"")
    assert resp == QaicECHandOffReqType.RESP_INVALID_CMD
