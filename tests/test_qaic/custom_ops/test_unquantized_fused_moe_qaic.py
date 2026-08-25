# ---------------------------------------------------------------------------------------
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries. All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc. and/or its subsidiaries.
# ---------------------------------------------------------------------------------------
"""Correctness and latency checks for the QAIC unquantized fused-MoE kernel.

Run as pytest:
    pytest -s tests/test_qaic/custom_ops/test_unquantized_fused_moe_qaic.py

Run as a standalone benchmark:
    python tests/test_qaic/custom_ops/test_unquantized_fused_moe_qaic.py \
        --tokens 16 --experts 8 --hidden-size 128 --intermediate-size 256 --topk 2

The CPU reference in this file mirrors the Triton/source-of-truth semantics:
when apply_router_weight_on_input=True, the router weight scales the input to
W13 before W13 bias is added. The W13 bias is not scaled by the router weight.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

try:
    import vllm_qaic  # noqa: F401
    from vllm_qaic import _custom_ops as _qaic_ops
    from vllm_qaic.ops import register_qaic_customop

    register_qaic_customop()
    _VLLM_QAIC_AVAILABLE = True
except Exception:
    _qaic_ops = None  # type: ignore[assignment]
    _VLLM_QAIC_AVAILABLE = False

def _platform_predicate(name: str) -> bool:
    predicate = getattr(current_platform, name, None)
    return bool(predicate()) if callable(predicate) else False


_REQUIRES_QAIC = pytest.mark.skipif(
    not (
        (_platform_predicate("is_qaic") or _platform_predicate("is_out_of_tree"))
        and _VLLM_QAIC_AVAILABLE
    ),
    reason="Test requires QAIC device and vllm_qaic package.",
)

_ACTIVATIONS = [
    "silu",
    "gelu",
    "gelu_tanh",
    "swigluoai",
    "swiglustep",
    "silu_no_mul",
    "gelu_no_mul",
    "gelu_tanh_no_mul",
    "relu2_no_mul",
]


@dataclass(frozen=True)
class MoeCase:
    tokens: int
    experts: int
    hidden_size: int
    intermediate_size: int
    topk: int
    activation: str
    has_bias: bool
    apply_router_weight_on_input: bool
    seed: int = 0
    ep_size: int = 1
    ep_rank: int = 0


def _sync(device: torch.device) -> None:
    if hasattr(torch, "qaic") and hasattr(torch.qaic, "synchronize"):
        torch.qaic.synchronize(device)


def _make_inputs(case: MoeCase, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case.seed)

    w13_dim = (
        case.intermediate_size
        if case.activation.endswith("_no_mul")
        else 2 * case.intermediate_size
    )
    global_experts = case.experts
    if case.ep_size > 1:
        local_global_experts = list(range(case.ep_rank, global_experts, case.ep_size))
    else:
        local_global_experts = list(range(global_experts))
    local_experts = len(local_global_experts)
    x = torch.randn(
        (case.tokens, case.hidden_size), generator=generator, dtype=torch.float16
    )
    w13_weight = 0.02 * torch.randn(
        (local_experts, w13_dim, case.hidden_size),
        generator=generator,
        dtype=torch.float16,
    )
    w2_weight = 0.02 * torch.randn(
        (local_experts, case.hidden_size, case.intermediate_size),
        generator=generator,
        dtype=torch.float16,
    )
    if case.has_bias:
        w13_bias = 0.01 * torch.randn(
            (local_experts, w13_dim), generator=generator, dtype=torch.float16
        )
        w2_bias = 0.01 * torch.randn(
            (local_experts, case.hidden_size), generator=generator, dtype=torch.float16
        )
    else:
        w13_bias = torch.empty((0,), dtype=torch.float16)
        w2_bias = torch.empty((0,), dtype=torch.float16)

    logits = torch.randn((case.tokens, global_experts), generator=generator)
    topk_weights, topk_ids = torch.topk(torch.softmax(logits, dim=-1), case.topk, dim=-1)
    expert_map = torch.full((global_experts,), -1, dtype=torch.float32)
    for local_idx, global_idx in enumerate(local_global_experts):
        expert_map[global_idx] = float(local_idx)

    tensors = {
        "x": x,
        "topk_weights": topk_weights.to(torch.float32),
        "topk_ids": topk_ids.to(torch.int32),
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "w13_bias": w13_bias,
        "w2_bias": w2_bias,
        "expert_map": expert_map,
    }
    return {name: tensor.to(device=device).contiguous() for name, tensor in tensors.items()}


def _activation(activation: str, gate_up: torch.Tensor) -> torch.Tensor:
    if activation == "silu_no_mul":
        return F.silu(gate_up)
    if activation == "gelu_no_mul":
        return F.gelu(gate_up)
    if activation == "gelu_tanh_no_mul":
        return F.gelu(gate_up, approximate="tanh")
    if activation == "relu2_no_mul":
        return F.relu(gate_up).square()
    if activation == "swigluoai":
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=7.0)
        up = up.clamp(-7.0, 7.0)
        return (up + 1) * (gate * torch.sigmoid(gate * 1.702))

    half = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :half], gate_up[..., half:]
    if activation == "swiglustep":
        return F.silu(gate).clamp(max=7.0) * up.clamp(-7.0, 7.0)
    if activation == "silu":
        return F.silu(gate) * up
    if activation == "gelu":
        return F.gelu(gate) * up
    if activation == "gelu_tanh":
        return F.gelu(gate, approximate="tanh") * up
    raise ValueError(f"Unsupported activation: {activation}")


def torch_triton_semantics_ref(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    activation: str,
    has_bias: bool,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch reference matching Triton MoE semantics, not legacy forward_oot."""
    out = torch.zeros_like(x)
    num_tokens, topk = topk_ids.shape
    for token in range(num_tokens):
        token_out = torch.zeros_like(x[token])
        for route in range(topk):
            expert = int(topk_ids[token, route].item())
            if expert_map is not None:
                expert = int(expert_map[expert].item()) if 0 <= expert < expert_map.numel() else -1
                if expert < 0:
                    continue
            route_weight = topk_weights[token, route].to(dtype=x.dtype)
            w13_input = (
                route_weight * x[token]
                if apply_router_weight_on_input
                else x[token]
            )
            gate_up = w13_input @ w13_weight[expert].t()
            if has_bias:
                gate_up = gate_up + w13_bias[expert]
            hidden = _activation(activation, gate_up)
            expert_out = hidden @ w2_weight[expert].t()
            if has_bias:
                expert_out = expert_out + w2_bias[expert]
            if not apply_router_weight_on_input:
                expert_out = route_weight * expert_out
            token_out = token_out + expert_out
        out[token] = token_out
    return out


def torch_grouped_ref(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_bias: torch.Tensor,
    w2_bias: torch.Tensor,
    activation: str,
    has_bias: bool,
    apply_router_weight_on_input: bool,
    expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Torch implementation shaped like forward_oot: group routes by expert."""
    num_tokens = x.shape[0]
    topk = topk_ids.shape[1]
    token_idx = (
        torch.arange(num_tokens, device=x.device)
        .unsqueeze(1)
        .expand(-1, topk)
        .reshape(-1)
    )
    flat_expert_ids = topk_ids.reshape(-1)
    if expert_map is not None:
        valid_global = (flat_expert_ids >= 0) & (flat_expert_ids < expert_map.numel())
        mapped = torch.full_like(flat_expert_ids, -1)
        mapped[valid_global] = expert_map[flat_expert_ids[valid_global].long()].to(mapped.dtype)
        valid = mapped >= 0
        flat_expert_ids = mapped[valid]
        flat_weights = topk_weights.reshape(-1).to(x.dtype)[valid]
        token_idx = token_idx[valid]
    else:
        flat_weights = topk_weights.reshape(-1).to(x.dtype)

    if flat_expert_ids.numel() == 0:
        return torch.zeros_like(x)

    sorted_idx = torch.argsort(flat_expert_ids, stable=True)
    sorted_expert_ids = flat_expert_ids[sorted_idx]
    sorted_token_idx = token_idx[sorted_idx]
    sorted_weights = flat_weights[sorted_idx]
    sorted_tokens = x[sorted_token_idx]

    out = torch.zeros_like(x)
    changes = torch.cat([
        torch.tensor([True], device=x.device),
        sorted_expert_ids[1:] != sorted_expert_ids[:-1],
        torch.tensor([True], device=x.device),
    ])
    boundary_positions = torch.where(changes)[0]
    unique_experts = sorted_expert_ids[boundary_positions[:-1]].tolist()
    counts = (boundary_positions[1:] - boundary_positions[:-1]).tolist()

    offset = 0
    for expert_id, count in zip(unique_experts, counts):
        expert_id = int(expert_id)
        tokens = sorted_tokens[offset : offset + count]
        weights = sorted_weights[offset : offset + count]
        tgt_idx = sorted_token_idx[offset : offset + count]

        w13_input = (
            weights.unsqueeze(-1) * tokens
            if apply_router_weight_on_input
            else tokens
        )
        gate_up = w13_input @ w13_weight[expert_id].t()
        if has_bias:
            gate_up = gate_up + w13_bias[expert_id]
        hidden = _activation(activation, gate_up)

        expert_out = hidden @ w2_weight[expert_id].t()
        if has_bias:
            expert_out = expert_out + w2_bias[expert_id]
        weighted_out = (
            expert_out
            if apply_router_weight_on_input
            else weights.unsqueeze(-1) * expert_out
        )
        out.index_put_((tgt_idx,), weighted_out, accumulate=True)
        offset += count

    return out



def qaic_hvx_kernel_call(case: MoeCase, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    assert _qaic_ops is not None
    return _qaic_ops.unquantized_fused_moe_hvx(
        tensors["x"],
        tensors["topk_weights"],
        tensors["topk_ids"],
        tensors["w13_weight"],
        tensors["w2_weight"],
        tensors["w13_bias"] if case.has_bias else None,
        tensors["w2_bias"] if case.has_bias else None,
        case.activation,
        case.has_bias,
        case.apply_router_weight_on_input,
        tensors.get("expert_map") if case.ep_size > 1 else None,
    )



def _assert_case(case: MoeCase) -> None:
    device = torch.device("qaic:0")
    qaic_tensors = _make_inputs(case, device)
    cpu_tensors = {name: tensor.cpu() for name, tensor in qaic_tensors.items()}

    expected = torch_triton_semantics_ref(
        cpu_tensors["x"],
        cpu_tensors["topk_weights"],
        cpu_tensors["topk_ids"],
        cpu_tensors["w13_weight"],
        cpu_tensors["w2_weight"],
        cpu_tensors["w13_bias"],
        cpu_tensors["w2_bias"],
        case.activation,
        case.has_bias,
        case.apply_router_weight_on_input,
        cpu_tensors["expert_map"] if case.ep_size > 1 else None,
    )

    _sync(device)
    hvx_actual = qaic_hvx_kernel_call(case, qaic_tensors)
    _sync(device)
    torch.testing.assert_close(
        hvx_actual.cpu(),
        expected,
        atol=4e-2,
        rtol=4e-2,
        msg=f"hvx case={case}",
    )


def _latency_ms(fn, warmup: int, iters: int, device: torch.device | None = None) -> float:
    for _ in range(warmup):
        fn()
    if device is not None:
        _sync(device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if device is not None:
            _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


@_REQUIRES_QAIC
@pytest.mark.parametrize("activation", _ACTIVATIONS)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("apply_router_weight_on_input", [False, True])
def test_unquantized_fused_moe_qaic_correctness(
    activation: str,
    has_bias: bool,
    apply_router_weight_on_input: bool,
) -> None:
    case = MoeCase(
        tokens=4,
        experts=4,
        hidden_size=32,
        intermediate_size=64,
        topk=1 if apply_router_weight_on_input else 2,
        activation=activation,
        has_bias=has_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        seed=17,
    )
    _assert_case(case)


@_REQUIRES_QAIC
def test_unquantized_fused_moe_hvx_expert_map() -> None:
    case = MoeCase(
        tokens=4,
        experts=8,
        hidden_size=32,
        intermediate_size=64,
        topk=2,
        activation="silu",
        has_bias=False,
        apply_router_weight_on_input=False,
        seed=19,
        ep_size=2,
        ep_rank=1,
    )
    _assert_case(case)


@_REQUIRES_QAIC
def test_unquantized_fused_moe_qaic_latency_smoke() -> None:
    case = MoeCase(
        tokens=8,
        experts=8,
        hidden_size=64,
        intermediate_size=128,
        topk=2,
        activation="silu",
        has_bias=False,
        apply_router_weight_on_input=False,
        seed=23,
    )
    device = torch.device("qaic:0")
    qaic_tensors = _make_inputs(case, device)
    cpu_tensors = {name: tensor.cpu() for name, tensor in qaic_tensors.items()}

    kernel_ms = _latency_ms(lambda: qaic_hvx_kernel_call(case, qaic_tensors), 5, 20, device)
    qaic_torch_ms = _latency_ms(
        lambda: torch_grouped_ref(
            qaic_tensors["x"],
            qaic_tensors["topk_weights"],
            qaic_tensors["topk_ids"],
            qaic_tensors["w13_weight"],
            qaic_tensors["w2_weight"],
            qaic_tensors["w13_bias"],
            qaic_tensors["w2_bias"],
            case.activation,
            case.has_bias,
            case.apply_router_weight_on_input,
            qaic_tensors["expert_map"] if case.ep_size > 1 else None,
        ),
        2,
        5,
        device,
    )
    cpu_ref_ms = _latency_ms(
        lambda: torch_triton_semantics_ref(
            cpu_tensors["x"],
            cpu_tensors["topk_weights"],
            cpu_tensors["topk_ids"],
            cpu_tensors["w13_weight"],
            cpu_tensors["w2_weight"],
            cpu_tensors["w13_bias"],
            cpu_tensors["w2_bias"],
            case.activation,
            case.has_bias,
            case.apply_router_weight_on_input,
        ),
        2,
        5,
        None,
    )
    print(
        f"\nQAIC MoE latency smoke: hvx={kernel_ms:.3f} ms, "
        f"qaic_torch_grouped_ref={qaic_torch_ms:.3f} ms, "
        f"cpu_torch_ref={cpu_ref_ms:.3f} ms, case={case}"
    )
    assert kernel_ms > 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--activation", choices=_ACTIVATIONS, default="silu")
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--apply-router-weight-on-input", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip-cpu-ref", action="store_true", help="Skip CPU correctness/reference latency; benchmark HVX only.")
    parser.add_argument("--ep-size", type=int, default=1, help="Simulate expert parallel local rank count.")
    parser.add_argument("--ep-rank", type=int, default=0, help="Simulated expert parallel rank.")
    args = parser.parse_args()

    if not _VLLM_QAIC_AVAILABLE:
        raise RuntimeError("vllm_qaic custom ops are not available in this environment")

    case = MoeCase(
        tokens=args.tokens,
        experts=args.experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        topk=args.topk,
        activation=args.activation,
        has_bias=args.has_bias,
        apply_router_weight_on_input=args.apply_router_weight_on_input,
        seed=123,
        ep_size=args.ep_size,
        ep_rank=args.ep_rank,
    )
    device = torch.device("qaic:0")
    qaic_tensors = _make_inputs(case, device)
    cpu_tensors = {name: tensor.cpu() for name, tensor in qaic_tensors.items()}

    expected = None
    if not args.skip_cpu_ref:
        expected = torch_triton_semantics_ref(
            cpu_tensors["x"],
            cpu_tensors["topk_weights"],
            cpu_tensors["topk_ids"],
            cpu_tensors["w13_weight"],
            cpu_tensors["w2_weight"],
            cpu_tensors["w13_bias"],
            cpu_tensors["w2_bias"],
            case.activation,
            case.has_bias,
            case.apply_router_weight_on_input,
            cpu_tensors["expert_map"] if case.ep_size > 1 else None,
        )

    hvx_actual = qaic_hvx_kernel_call(case, qaic_tensors)
    _sync(device)
    if expected is not None:
        torch.testing.assert_close(hvx_actual.cpu(), expected, atol=4e-2, rtol=4e-2)

    hvx_ms = _latency_ms(
        lambda: qaic_hvx_kernel_call(case, qaic_tensors),
        args.warmup,
        args.iters,
        device,
    )
    qaic_torch_ms = _latency_ms(
        lambda: torch_grouped_ref(
            qaic_tensors["x"],
            qaic_tensors["topk_weights"],
            qaic_tensors["topk_ids"],
            qaic_tensors["w13_weight"],
            qaic_tensors["w2_weight"],
            qaic_tensors["w13_bias"],
            qaic_tensors["w2_bias"],
            case.activation,
            case.has_bias,
            case.apply_router_weight_on_input,
            qaic_tensors["expert_map"] if case.ep_size > 1 else None,
        ),
        max(1, args.warmup // 2),
        max(1, args.iters // 4),
        device,
    )
    cpu_ref_ms = None
    if not args.skip_cpu_ref:
        cpu_ref_ms = _latency_ms(
            lambda: torch_triton_semantics_ref(
                cpu_tensors["x"],
                cpu_tensors["topk_weights"],
                cpu_tensors["topk_ids"],
                cpu_tensors["w13_weight"],
                cpu_tensors["w2_weight"],
                cpu_tensors["w13_bias"],
                cpu_tensors["w2_bias"],
                case.activation,
                case.has_bias,
                case.apply_router_weight_on_input,
                cpu_tensors["expert_map"] if case.ep_size > 1 else None,
            ),
            max(1, args.warmup // 2),
            max(1, args.iters // 4),
            None,
        )
    print(f"correctness: PASS {case}")
    print(f"latency_ms.qaic_hvx_kernel: {hvx_ms:.3f}")
    print(f"latency_ms.qaic_torch_grouped_ref: {qaic_torch_ms:.3f}")
    if cpu_ref_ms is not None:
        print(f"latency_ms.cpu_torch_ref: {cpu_ref_ms:.3f}")
    else:
        print("latency_ms.cpu_torch_ref: SKIPPED")


if __name__ == "__main__":
    main()
