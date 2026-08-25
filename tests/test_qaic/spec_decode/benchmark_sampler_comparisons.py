"""Measure QAIC eager sampler comparison fallbacks.

Run this in a fresh process so ``QAIC_VISIBLE_DEVICES`` is set before PyTorch
imports the QAIC backend, for example:

    QAIC_VISIBLE_DEVICES=0 .venv_eager/bin/python \
      tests/test_qaic/spec_decode/benchmark_sampler_comparisons.py

The benchmark mirrors the explicit CPU predicates in
``rejection_sampler_shim.py``. It reports stage-level timings with QAIC
synchronization on both sides of every sample; tensor construction and warm-up
are deliberately excluded from timed regions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
import torch_qaic  # noqa: F401


_SAMPLING_EPS = 1e-5


def _latency_us(samples: list[int]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "median_us": statistics.median(samples) / 1_000,
        "p95_us": ordered[p95_index] / 1_000,
    }


def _measure(iterations: int, operation: Callable[[], object]) -> dict[str, float]:
    samples: list[int] = []
    for _ in range(iterations):
        torch.qaic.synchronize()
        started = time.perf_counter_ns()
        operation()
        torch.qaic.synchronize()
        samples.append(time.perf_counter_ns() - started)
    return _latency_us(samples)


def _warm_up(iterations: int, operation: Callable[[], object]) -> None:
    for _ in range(iterations):
        operation()
    torch.qaic.synchronize()


def _temperature_case(batch_size: int, warmup: int, iterations: int) -> dict[str, object]:
    temp = torch.linspace(0.0, 1.0, batch_size, dtype=torch.float32, device="qaic")

    def q2h() -> torch.Tensor:
        return temp.cpu()

    host_temp = temp.cpu()

    def predicate() -> torch.Tensor:
        return torch.where(
            host_temp < _SAMPLING_EPS,
            torch.ones_like(host_temp),
            host_temp,
        )

    sanitized = predicate()

    def h2q() -> torch.Tensor:
        return sanitized.to(device=temp.device)

    def complete() -> torch.Tensor:
        host = temp.cpu()
        host = torch.where(
            host < _SAMPLING_EPS,
            torch.ones_like(host),
            host,
        )
        return host.to(device=temp.device)

    for operation in (q2h, predicate, h2q, complete):
        _warm_up(warmup, operation)

    byte_count = batch_size * temp.element_size()
    return {
        "kind": "temperature",
        "batch_size": batch_size,
        "dtype": str(temp.dtype),
        "q2h_bytes": byte_count,
        "h2q_bytes": byte_count,
        "q2h": _measure(iterations, q2h),
        "cpu_predicate": _measure(iterations, predicate),
        "h2q": _measure(iterations, h2q),
        "complete_fallback": _measure(iterations, complete),
    }


def _mask_case(
    kind: str,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    cpu_logits = torch.linspace(
        -8.0,
        8.0,
        batch_size * vocab_size,
        dtype=dtype,
    ).reshape(batch_size, vocab_size)
    logits = cpu_logits.to(device="qaic")
    row_threshold = logits[:, vocab_size // 2 : vocab_size // 2 + 1].clone()
    top_p = torch.full((batch_size,), 0.9, dtype=dtype, device="qaic")

    if kind == "top_k":
        def q2h_operands() -> tuple[torch.Tensor, torch.Tensor]:
            return logits.cpu(), row_threshold.cpu()

        host_logits, host_threshold = q2h_operands()

        def predicate() -> torch.Tensor:
            return torch.lt(host_logits, host_threshold)

        def complete() -> torch.Tensor:
            mask = torch.lt(logits.cpu(), row_threshold.cpu()).to(device="qaic")
            values = logits.clone()
            values.masked_fill_(mask, -float("inf"))
            return values

        threshold_bytes = row_threshold.numel() * row_threshold.element_size()
    elif kind == "top_p":
        probs_sum = logits.softmax(dim=-1).cumsum(dim=-1)

        def q2h_operands() -> tuple[torch.Tensor, torch.Tensor]:
            return probs_sum.cpu(), (1 - top_p.unsqueeze(dim=1)).cpu()

        host_logits, host_threshold = q2h_operands()

        def predicate() -> torch.Tensor:
            return torch.le(host_logits, host_threshold)

        def complete() -> torch.Tensor:
            mask = torch.le(
                probs_sum.cpu(), (1 - top_p.unsqueeze(dim=1)).cpu()
            ).to(device="qaic")
            mask[:, -1] = False
            values = logits.clone()
            values.masked_fill_(mask, -float("inf"))
            return values

        threshold_bytes = top_p.numel() * top_p.element_size()
    elif kind == "combined":
        def q2h_operands() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return logits.cpu(), row_threshold.cpu(), top_p.cpu()

        host_logits, host_threshold, host_top_p = q2h_operands()

        def predicate() -> tuple[torch.Tensor, torch.Tensor]:
            top_k_mask = torch.lt(host_logits, host_threshold)
            filtered_logits = host_logits.masked_fill(top_k_mask, -float("inf"))
            probs_sum = filtered_logits.softmax(dim=-1).cumsum(dim=-1)
            top_p_mask = torch.le(probs_sum, 1 - host_top_p.unsqueeze(dim=1))
            top_p_mask[:, -1] = False
            return top_k_mask, top_p_mask

        def complete() -> torch.Tensor:
            values = logits.clone()
            top_k_mask = torch.lt(
                values.cpu(), row_threshold.cpu()
            ).to(device="qaic")
            values.masked_fill_(top_k_mask, -float("inf"))
            probs_sum = values.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sum, dim=-1, out=probs_sum)
            top_p_mask = torch.le(
                probs_sum.cpu(), (1 - top_p.unsqueeze(dim=1)).cpu()
            ).to(device="qaic")
            top_p_mask[:, -1] = False
            values.masked_fill_(top_p_mask, -float("inf"))
            return values

        threshold_bytes = (
            row_threshold.numel() * row_threshold.element_size()
            + top_p.numel() * top_p.element_size()
        )
    else:
        raise ValueError(f"Unsupported mask kind: {kind}")

    mask = predicate()

    def q2h() -> tuple[torch.Tensor, torch.Tensor]:
        return q2h_operands()

    def h2q() -> torch.Tensor | tuple[torch.Tensor, ...]:
        if isinstance(mask, tuple):
            return tuple(value.to(device="qaic") for value in mask)
        return mask.to(device="qaic")

    for operation in (q2h, predicate, h2q, complete):
        _warm_up(warmup, operation)

    operand_bytes = logits.numel() * logits.element_size() + threshold_bytes
    return {
        "kind": kind,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "q2h_bytes": operand_bytes,
        "h2q_bytes": logits.numel() * (2 if kind == "combined" else 1),
        "q2h": _measure(iterations, q2h),
        "cpu_predicate": _measure(iterations, predicate),
        "h2q": _measure(iterations, h2q),
        "complete_mask_and_fill": _measure(iterations, complete),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--include-float32", action="store_true")
    args = parser.parse_args()

    if torch.qaic.device_count() != 1:
        raise RuntimeError(
            "Run with exactly one visible QAIC device, for example "
            "QAIC_VISIBLE_DEVICES=0."
        )

    dtypes = [torch.float16]
    if args.include_float32:
        dtypes.append(torch.float32)

    results: list[dict[str, object]] = []
    for batch_size in (1, 8, 32, 128):
        results.append(_temperature_case(batch_size, args.warmup, args.iterations))
    for batch_size in (1, 8, 32):
        for vocab_size in (32_768, 128_256):
            for dtype in dtypes:
                for kind in ("top_k", "top_p", "combined"):
                    results.append(
                        _mask_case(
                            kind,
                            batch_size,
                            vocab_size,
                            dtype,
                            args.warmup,
                            args.iterations,
                        )
                    )

    print(json.dumps({"results": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
