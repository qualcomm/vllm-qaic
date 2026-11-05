# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------
# Shim for torch < 2.8.0: torch.fx._graph_pickler.Options and the optional
# `options` argument to GraphPickler.dumps were added in torch 2.8.0.
# vllm.compilation.caching imports both at module level, so we inject the
# missing names before that import happens.

# TODO: Remove after qeff upgrade torch to >= 2.8.0
import torch.fx._graph_pickler as _gp

if not hasattr(_gp, "Options"):
    from dataclasses import dataclass
    from collections.abc import Callable

    @dataclass
    class Options:
        ops_filter: Callable[[str], bool] | None = None

    _gp.Options = Options  # type: ignore[attr-defined]

    _original_dumps = _gp.GraphPickler.dumps.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _dumps_compat(cls, obj: object, options: Options | None = None) -> bytes:  # type: ignore[name-defined]
        return _original_dumps(cls, obj)

    _gp.GraphPickler.dumps = _dumps_compat  # type: ignore[method-assign]
