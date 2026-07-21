# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.envs import (
    environment_variables,
    maybe_convert_bool,
    maybe_convert_int,
)


if TYPE_CHECKING:
    VLLM_QAIC_COMPILER_ARGS: str | None = None
    VLLM_QAIC_DFS_EN: bool = True
    VLLM_QAIC_MAX_CPU_THREADS: int | None = None
    VLLM_QAIC_MOS: int | None = None
    VLLM_QAIC_NUM_CORES: int | None = None
    VLLM_QAIC_QPC_PATH: str | None = None
    VLLM_TORCH_QAIC_PROFILER_DIR: str | None = None

# --8<-- [start:env-vars-definition]
qaic_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_QAIC_COMPILER_ARGS": lambda: os.getenv("VLLM_QAIC_COMPILER_ARGS", None),
    "VLLM_QAIC_DFS_EN": lambda: maybe_convert_bool(os.getenv("VLLM_QAIC_DFS_EN", None)),
    "VLLM_QAIC_MAX_CPU_THREADS": lambda: maybe_convert_int(
        os.getenv("VLLM_QAIC_MAX_CPU_THREADS", None)
    ),
    "VLLM_QAIC_MOS": lambda: maybe_convert_int(os.getenv("VLLM_QAIC_MOS", None)),
    "VLLM_QAIC_NUM_CORES": lambda: maybe_convert_int(
        os.getenv("VLLM_QAIC_NUM_CORES", None)
    ),
    "VLLM_QAIC_QPC_PATH": lambda: os.getenv("VLLM_QAIC_QPC_PATH", None),
    "VLLM_TORCH_QAIC_PROFILER_DIR": lambda: os.getenv(
        "VLLM_TORCH_QAIC_PROFILER_DIR", None
    ),
}
environment_variables.update(qaic_environment_variables)
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    """
    Gets environment variables lazily.
    """
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
