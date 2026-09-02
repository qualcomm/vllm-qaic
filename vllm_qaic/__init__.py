# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

from collections.abc import Callable
from dataclasses import dataclass

try:
    from vllm_qaic._version import __version__
except ImportError:
    __version__ = "unknown"


def _apply_global_platform_patches():
    _patch_graph_pickler_options()
    _patch_torch_fp4_dtype()


# TODO: Remove once torch >= 2.8 is adopted in QAIC envs.
def _patch_graph_pickler_options() -> None:
    import torch.fx._graph_pickler as _gp

    if hasattr(_gp, "Options"):
        return

    @dataclass
    class Options:
        ops_filter: Callable[[str], bool] | None = None

    _gp.Options = Options  # type: ignore[attr-defined]

    _original_dumps = _gp.GraphPickler.dumps.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _dumps_compat(cls, obj: object, options: Options | None = None) -> bytes:
        return _original_dumps(cls, obj)

    _gp.GraphPickler.dumps = _dumps_compat  # type: ignore[method-assign]


# TODO: Remove once QEFF or torch-qaic is moved to Pytorch >= 2.11.x
def _patch_torch_fp4_dtype():
    """Add torch.float4_e2m1fn_x2 sentinel on CPU-only PyTorch builds.

    ``vllm.ir.tolerances`` references ``torch.float4_e2m1fn_x2`` at module level. That dtype
    does not exist in the CPU-only PyTorch build used by QAIC. ``torch.float32`` is used as a
    placeholder because the tolerances dict is only consumed by IR comparison tests, never
    during QAIC inference.
    """
    import torch

    if not hasattr(torch, "float4_e2m1fn_x2"):
        torch.float4_e2m1fn_x2 = torch.float32


_apply_global_platform_patches()


def register():
    _apply_global_platform_patches()
    return "vllm_qaic.platform.QaicPlatform"


def register_connector():
    from vllm_qaic.distributed.kv_transfer.kv_connector import register_connector

    register_connector()
