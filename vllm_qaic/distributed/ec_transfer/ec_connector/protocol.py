# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
from enum import Enum, IntEnum

import msgspec

MMHash = str  # Same as MultiModalHasher.hash_kwargs (vllm/multimodal/hasher.py)


class QaicECHandOffReqType(Enum):
    """The type of ec_handoff message."""

    GET = b"\x00"
    PUT = b"\x01"
    RESP_OK = b"\x02"
    RESP_ERROR = b"\x03"
    RESP_BUFFER_FULL = b"\x04"
    RESP_NOT_FOUND = b"\x05"
    RESP_INVALID_CMD = b"\x06"
    PEEK = b"\x07"
    UNPIN = b"\x08"


class QaicBufferType(IntEnum):
    """The type of ec_handoff buffer."""

    SHM = 0
    NP = 1
    DISK = 2


class QaicECHandOffGetResp(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    Header for ec_handoff GetResp messages.
    """

    buff_type: QaicBufferType
    timestamp: float
    rank: int
    num_buff: int = 1
    payload: list[str] | None = None
    shape: list[list[int]] | None = None
    dtype: list[str] | None = None


class QaicECHandOffPutReq(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    Header for ec_handoff PutReq messages.
    """

    buff_type: QaicBufferType
    timestamp: float
    mm_hash: MMHash
    rank: int
    num_buff: int
    payload: list[str]
    shape: list[list[int]]
    dtype: list[str]


class QaicECHandOffGetReq(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    """
    Header for ec_handoff GetReq messages.
    """

    timestamp: float
    mm_hash: MMHash
    rank: int


ReqType = QaicECHandOffPutReq | QaicECHandOffGetReq
