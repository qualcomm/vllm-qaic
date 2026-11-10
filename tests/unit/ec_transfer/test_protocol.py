# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
"""Msgpack round-trip tests for the EC hand-off protocol messages."""

import time

import pytest

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm_qaic.distributed.ec_transfer.ec_connector.protocol import (
    QaicBufferType,
    QaicECHandOffGetReq,
    QaicECHandOffGetResp,
    QaicECHandOffPutReq,
    QaicECHandOffReqType,
)


def _roundtrip(struct_cls, obj):
    return MsgpackDecoder(struct_cls).decode(MsgpackEncoder(struct_cls).encode(obj)[0])


@pytest.mark.parametrize("shape", [[187, 8192], [1, 64, 128], [81, 1, 2040, 4096]])
def test_put_req_roundtrip(shape):
    req = QaicECHandOffPutReq(
        buff_type=QaicBufferType.SHM,
        timestamp=time.perf_counter(),
        mm_hash="abc",
        rank=0,
        num_buff=1,
        payload=["psm_x"],
        shape=[shape],
        dtype=["float16"],
    )
    decoded = _roundtrip(QaicECHandOffPutReq, req)
    assert decoded.buff_type == QaicBufferType.SHM
    assert decoded.mm_hash == "abc"
    assert decoded.rank == 0
    assert decoded.num_buff == 1
    assert decoded.payload == ["psm_x"]
    assert decoded.shape == [shape]
    assert decoded.dtype == ["float16"]


def test_get_req_roundtrip():
    req = QaicECHandOffGetReq(timestamp=time.perf_counter(), mm_hash="abc", rank=1)
    decoded = _roundtrip(QaicECHandOffGetReq, req)
    assert decoded.mm_hash == "abc"
    assert decoded.rank == 1


def test_get_resp_roundtrip():
    resp = QaicECHandOffGetResp(
        buff_type=QaicBufferType.SHM,
        timestamp=time.perf_counter(),
        rank=0,
        num_buff=1,
        payload=["psm_x"],
        shape=[[187, 8192]],
        dtype=["float16"],
    )
    decoded = _roundtrip(QaicECHandOffGetResp, resp)
    assert decoded.buff_type == QaicBufferType.SHM
    assert decoded.payload == ["psm_x"]
    assert decoded.shape == [[187, 8192]]
    assert decoded.dtype == ["float16"]


def test_get_resp_defaults_are_omitted():
    resp = QaicECHandOffGetResp(
        buff_type=QaicBufferType.SHM, timestamp=time.perf_counter(), rank=0
    )
    decoded = _roundtrip(QaicECHandOffGetResp, resp)
    assert decoded.num_buff == 1
    assert decoded.payload is None
    assert decoded.shape is None
    assert decoded.dtype is None


def test_req_type_values_are_unique_single_bytes():
    values = [member.value for member in QaicECHandOffReqType]
    assert len(values) == len(set(values))
    assert all(isinstance(value, bytes) and len(value) == 1 for value in values)
