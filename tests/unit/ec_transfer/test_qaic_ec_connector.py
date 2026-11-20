# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Host-only round-trip tests for QaicECConnector against a live EC server."""

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import ECTransferConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm_qaic.distributed.ec_transfer.ec_connector import qaic_connector
from vllm_qaic.distributed.ec_transfer.ec_connector.qaic_connector import (
    MMMeta,
    QaicECConnector,
    QaicECConnectorMetadata,
)

ec_server = pytest.importorskip("qaic_disagg.ec_handoff.server")

HOST = "127.0.0.1"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


@pytest.fixture
def server():
    port = _free_port()
    instance = ec_server.QaicECCacheServer(HOST, port, ttl=3600.0, max_bytes=1024**3)
    threading.Thread(
        target=instance.process_client_request,
        args=(f"tcp://{HOST}:{port}",),
        daemon=True,
    ).start()
    time.sleep(0.5)
    yield instance, port


@pytest.fixture
def connectors(server, monkeypatch):
    monkeypatch.setattr(qaic_connector, "EC_LOOKUP_RETRIES_INTERVAL", 0.001)
    _, port = server
    created = []

    def make(role):
        cfg = ECTransferConfig(
            ec_connector="QaicECConnector",
            ec_role=role,
            ec_ip=HOST,
            ec_port=port,
            ec_rank=0 if role == "ec_producer" else 1,
        )
        conn = QaicECConnector(
            SimpleNamespace(ec_transfer_config=cfg), ECConnectorRole.WORKER
        )
        created.append(conn)
        return conn

    yield make("ec_producer"), make("ec_consumer")
    for conn in created:
        conn.close()


def _load(consumer, mm_hash):
    meta = QaicECConnectorMetadata()
    meta.add_mm_data(MMMeta.make_meta(mm_hash))
    consumer.bind_connector_metadata(meta)
    cache = {}
    try:
        consumer.start_load_caches(cache)
    finally:
        consumer.clear_connector_metadata()
    return cache


@pytest.mark.parametrize(
    "shape",
    [(576, 5120), (187, 8192), (1, 64, 128)],
    ids=["2d-qwen2_5_vl", "2d-qwen3_vl", "3d"],
)
def test_save_then_load_is_byte_exact(connectors, shape):
    producer, consumer = connectors
    mm_hash = f"h-{'x'.join(map(str, shape))}"
    emb = torch.from_numpy(
        (np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) / 7).astype(
            np.float16
        )
    )
    producer.save_caches({mm_hash: emb}, mm_hash)
    assert consumer.has_cache_item(mm_hash)

    loaded = _load(consumer, mm_hash)[mm_hash]
    assert tuple(loaded.shape) == shape
    assert loaded.dtype == emb.dtype
    assert torch.equal(loaded, emb)
    consumer.free_caches([mm_hash])


def test_producer_never_reports_cache_hit(connectors):
    producer, _ = connectors
    emb = torch.ones(4, 8, dtype=torch.float16)
    producer.save_caches({"h": emb}, "h")
    assert not producer.has_cache_item("h")


def test_consumer_save_is_noop(server, connectors):
    instance, _ = server
    _, consumer = connectors
    consumer.save_caches({"h": torch.ones(4, 8, dtype=torch.float16)}, "h")
    assert "h" not in instance.data_store


def test_missing_hash_raises_on_consumer(connectors):
    _, consumer = connectors
    assert not consumer.has_cache_item("missing")
    with pytest.raises(ValueError, match="Unable to find prompt hash"):
        _load(consumer, "missing")


def test_free_caches_unpins_entry(server, connectors):
    instance, _ = server
    producer, consumer = connectors
    producer.save_caches({"h": torch.ones(4, 8, dtype=torch.float16)}, "h")
    _load(consumer, "h")
    assert instance.data_store.peek("h").ref_count == 1
    consumer.free_caches(["h"])
    assert instance.data_store.peek("h").ref_count == 0
    consumer.free_caches(["h"])
    assert instance.data_store.peek("h").ref_count == 0


def test_already_cached_hash_skips_lookup(connectors):
    _, consumer = connectors
    meta = QaicECConnectorMetadata()
    meta.add_mm_data(MMMeta.make_meta("local"))
    consumer.bind_connector_metadata(meta)
    sentinel = torch.zeros(2, 2)
    cache = {"local": sentinel}
    consumer.start_load_caches(cache)
    consumer.clear_connector_metadata()
    assert cache["local"] is sentinel


def test_build_connector_meta_drains_pending_hashes(connectors):
    _, consumer = connectors
    request = SimpleNamespace(
        mm_features=[SimpleNamespace(identifier="a"), SimpleNamespace(identifier="b")]
    )
    consumer.update_state_after_alloc(request, 0)
    consumer.update_state_after_alloc(request, 1)
    meta = consumer.build_connector_meta(SimpleNamespace())
    assert [m.mm_hash for m in meta.mm_datas] == ["a", "b"]
    assert consumer.build_connector_meta(SimpleNamespace()).mm_datas == []
