# ------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
# ------------------------------------------------------------------

"""Add functional ``matmul_ogs`` option flags for the QAIC Triton backend.

OpenAI's ``triton_kernels`` currently dispatches option selection only for
CUDA and HIP.  QAIC reaches the final assertion even though the kernel itself
can be compiled by the out-of-tree Qualcomm Triton backend.  This module
intercepts that dispatch and supplies conservative CUDA-like tiling values for
functional bring-up.  They are deliberately not performance tuned.
"""

import triton

from triton_kernels.matmul_ogs_details import opt_flags

QAIC_TRITON_BACKEND_KEY = "qcom_hexagon_backend"

_upstream_make_opt_flags = opt_flags.make_opt_flags


def make_default_opt_flags_qaic(
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    precision_config,
    m,
    n,
    k,
    routing_data,
    can_use_persistent_tma,
    can_use_fused_scatter,
    enforce_bitwise_invariance,
    epilogue_effective_itemsize,
    constraints,
):
    """Return conservative CUDA-like flags without CUDA device queries."""
    del (
        out_dtype,
        lhs_dtype,
        rhs_dtype,
        precision_config,
        k,
        can_use_persistent_tma,
        can_use_fused_scatter,
        enforce_bitwise_invariance,
        epilogue_effective_itemsize,
    )

    supported_constraints = {
        "block_m",
        "block_n",
        "block_k",
        "split_k",
        "is_persistent",
        "fused_scatter",
        "epilogue_subtile",
        "num_stages",
        "idle_sms",
    }
    unsupported = set(constraints) - supported_constraints
    if unsupported:
        raise opt_flags.InapplicableConstraint(
            f"QAIC matmul_ogs does not support constraints: {sorted(unsupported)}"
        )

    if routing_data is None:
        tokens_per_expert = m
    elif routing_data.expected_tokens_per_expt is None:
        tokens_per_expert = max(1, m // routing_data.n_expts_tot)
    else:
        tokens_per_expert = routing_data.expected_tokens_per_expt

    # Follow CUDA's basic M/N tiling shape, but keep the remaining choices
    # intentionally conservative until QAIC-specific tuning is available.
    block_m = constraints.get("block_m") or max(
        16, min(triton.next_power_of_2(tokens_per_expert), 128)
    )
    block_n = constraints.get("block_n") or max(16, min(triton.next_power_of_2(n), 128))
    block_k = constraints.get("block_k") or 32

    requested_persistent = constraints.get("is_persistent", False)
    requested_fused_scatter = constraints.get("fused_scatter", False)
    requested_split_k = constraints.get("split_k", 1)
    if requested_persistent:
        raise opt_flags.InapplicableConstraint(
            "persistent matmul_ogs is not enabled for QAIC bring-up"
        )
    if requested_fused_scatter:
        raise opt_flags.InapplicableConstraint(
            "fused scatter is not enabled for QAIC bring-up"
        )
    if requested_split_k != 1:
        raise opt_flags.InapplicableConstraint(
            "split-K is not enabled for QAIC bring-up"
        )

    result = opt_flags.OptFlags(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=constraints.get("num_stages") or 1,
        group_m=8,
        xcd_swizzle=1,
        w_cache_modifier=None,
        split_k=1,
        is_persistent=False,
        fused_scatter=False,
        idle_sms=constraints.get("idle_sms", 0),
        epilogue_subtile=constraints.get("epilogue_subtile") or 1,
        arch=None,
        target_kernel_kwargs={},
    )

    assert all(
        getattr(result, key) == value
        for key, value in constraints.items()
        if value is not None
    ), f"{result} does not satisfy {constraints}"
    return result


def _qaic_make_opt_flags(
    out_dtype,
    lhs_dtype,
    rhs_dtype,
    precision_config,
    m,
    n,
    k,
    routing_data,
    can_use_persistent_tma,
    can_use_fused_scatter,
    epilogue_effective_itemsize,
):
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend != QAIC_TRITON_BACKEND_KEY:
        return _upstream_make_opt_flags(
            out_dtype,
            lhs_dtype,
            rhs_dtype,
            precision_config,
            m,
            n,
            k,
            routing_data,
            can_use_persistent_tma,
            can_use_fused_scatter,
            epilogue_effective_itemsize,
        )

    if (
        opt_flags._opt_flags_constraints.get("is_persistent", False)
        and not can_use_persistent_tma
    ):
        raise opt_flags.InapplicableConstraint("cannot enforce is_persistent=True")
    if (
        opt_flags._opt_flags_constraints.get("fused_scatter", False)
        and not can_use_fused_scatter
    ):
        raise opt_flags.InapplicableConstraint("cannot enforce fused_scatter=True")

    if opt_flags._opt_flags is not None:
        assert not opt_flags._opt_flags_constraints
        return opt_flags._opt_flags

    return make_default_opt_flags_qaic(
        out_dtype,
        lhs_dtype,
        rhs_dtype,
        precision_config,
        m,
        n,
        k,
        routing_data,
        can_use_persistent_tma,
        can_use_fused_scatter,
        precision_config.enforce_bitwise_invariance,
        epilogue_effective_itemsize,
        opt_flags._opt_flags_constraints,
    )


opt_flags.make_opt_flags = _qaic_make_opt_flags
