# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Host-only tests for the qaic_disagg ECStore (LRU, pinning, TTL)."""

import contextlib
import time
from multiprocessing import shared_memory

import pytest

from vllm_qaic.distributed.ec_transfer.ec_connector.protocol import (
    QaicBufferType,
    QaicECHandOffPutReq,
)

ec_store = pytest.importorskip("qaic_disagg.storage.ec_store")
ECStoreEntry = ec_store.ECStoreEntry
QaicECStore = ec_store.QaicECStore

ENTRY_SHAPE = [100, 100]
ENTRY_BYTES = 100 * 100 * 2


def _segment_exists(name):
    try:
        segment = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    segment.close()
    return True


@pytest.fixture
def make_entry():
    segments = []

    def _make(shape=ENTRY_SHAPE, dtype="float16"):
        segment = shared_memory.SharedMemory(create=True, size=ENTRY_BYTES)
        segments.append(segment)
        put = QaicECHandOffPutReq(
            buff_type=QaicBufferType.SHM,
            timestamp=time.perf_counter(),
            mm_hash="unused",
            rank=0,
            num_buff=1,
            payload=[segment.name],
            shape=[shape],
            dtype=[dtype],
        )
        return ECStoreEntry.create_obj(put), segment.name

    yield _make
    for segment in segments:
        segment.close()
        with contextlib.suppress(FileNotFoundError):
            segment.unlink()


def _store(max_bytes):
    store = QaicECStore()
    store.max_bytes = max_bytes
    return store


def test_create_obj_computes_bytes_and_starts_unpinned(make_entry):
    entry, _ = make_entry()
    assert entry.total_bytes == ENTRY_BYTES
    assert entry.ref_count == 0


def test_create_obj_rejects_payload_mismatch():
    put = QaicECHandOffPutReq(
        buff_type=QaicBufferType.SHM,
        timestamp=time.perf_counter(),
        mm_hash="h",
        rank=0,
        num_buff=2,
        payload=["only_one"],
        shape=[ENTRY_SHAPE],
        dtype=["float16"],
    )
    with pytest.raises(ValueError):
        ECStoreEntry.create_obj(put)


def test_add_get_peek_unpin(make_entry):
    store = _store(10 * ENTRY_BYTES)
    entry, _ = make_entry()
    assert store.add("h", entry) == 1
    assert "h" in store and len(store) == 1

    assert store.peek("h") is entry
    assert entry.ref_count == 0

    assert store.get("h") is entry
    assert entry.ref_count == 1
    store.get("h")
    assert entry.ref_count == 2

    store.unpin("h")
    store.unpin("h")
    assert entry.ref_count == 0
    store.unpin("h")
    assert entry.ref_count == 0
    store.unpin("missing")
    assert "h" in store


def test_missing_key_raises(make_entry):
    store = _store(10 * ENTRY_BYTES)
    with pytest.raises(KeyError):
        store.get("missing")
    with pytest.raises(KeyError):
        store.peek("missing")


def test_duplicate_add_keeps_existing_and_releases_new(make_entry):
    store = _store(10 * ENTRY_BYTES)
    first, first_name = make_entry()
    second, second_name = make_entry()
    assert store.add("h", first) == 1
    assert store.add("h", second) == 0
    assert store.peek("h") is first
    assert _segment_exists(first_name)
    assert not _segment_exists(second_name)


@pytest.mark.xfail(
    reason="QaicECStore.add() subtracts the rejected duplicate's bytes from "
    "_total_bytes even though they were never added",
    strict=True,
)
def test_duplicate_add_keeps_byte_accounting(make_entry):
    store = _store(10 * ENTRY_BYTES)
    store.add("h", make_entry()[0])
    store.add("h", make_entry()[0])
    assert store._total_bytes == ENTRY_BYTES


def test_lru_evicts_oldest_unpinned(make_entry):
    store = _store(2 * ENTRY_BYTES + 10)
    entry_a, name_a = make_entry()
    entry_b, _ = make_entry()
    entry_c, _ = make_entry()
    store.add("a", entry_a)
    store.add("b", entry_b)
    assert store.add("c", entry_c) == 1
    assert "a" not in store
    assert "b" in store and "c" in store
    assert not _segment_exists(name_a)


def test_get_refreshes_lru_order(make_entry):
    store = _store(2 * ENTRY_BYTES + 10)
    entry_a, _ = make_entry()
    entry_b, name_b = make_entry()
    entry_c, _ = make_entry()
    store.add("a", entry_a)
    store.add("b", entry_b)
    store.get("a")
    store.unpin("a")
    store.add("c", entry_c)
    assert "a" in store
    assert "b" not in store
    assert not _segment_exists(name_b)


def test_pinned_entry_is_never_evicted(make_entry):
    store = _store(ENTRY_BYTES + 10)
    pinned, pinned_name = make_entry()
    incoming, incoming_name = make_entry()
    store.add("p", pinned)
    store.get("p")
    assert store.add("q", incoming) == -1
    assert "p" in store and _segment_exists(pinned_name)
    assert "q" not in store and not _segment_exists(incoming_name)


def test_entry_larger_than_budget_is_rejected(make_entry):
    store = _store(ENTRY_BYTES - 1)
    entry, name = make_entry()
    assert store.add("big", entry) == -1
    assert "big" not in store
    assert not _segment_exists(name)


def test_evict_expired_reaps_only_unpinned(make_entry):
    store = _store(10 * ENTRY_BYTES)
    idle, idle_name = make_entry()
    busy, busy_name = make_entry()
    store.add("idle", idle)
    store.add("busy", busy)
    store.get("busy")
    time.sleep(0.05)
    assert store.evict_expired(ttl=0.01) == 1
    assert "idle" not in store and not _segment_exists(idle_name)
    assert "busy" in store and _segment_exists(busy_name)
    store.unpin("busy")
    assert store.evict_expired(ttl=0.01) == 1
    assert "busy" not in store and not _segment_exists(busy_name)


def test_evict_expired_keeps_fresh_entries(make_entry):
    store = _store(10 * ENTRY_BYTES)
    entry, name = make_entry()
    store.add("h", entry)
    assert store.evict_expired(ttl=3600) == 0
    assert "h" in store and _segment_exists(name)


def test_byte_accounting(make_entry):
    store = _store(10 * ENTRY_BYTES)
    for key in ("a", "b", "c"):
        store.add(key, make_entry()[0])
    assert store._total_bytes == 3 * ENTRY_BYTES
    time.sleep(0.02)
    store.evict_expired(ttl=0.01)
    assert store._total_bytes == 0
    assert len(store) == 0
