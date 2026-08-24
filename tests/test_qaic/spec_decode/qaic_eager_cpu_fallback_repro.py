"""Capture eager QAIC CPU-fallback evidence for sampler comparisons.

Run one case in a fresh process after setting ``QAIC_VISIBLE_DEVICES`` before
Python imports Torch, for example:

    QAIC_VISIBLE_DEVICES=0 QAIC_DEBUG=1 TORCH_SHOW_CPP_STACKTRACES=1 \
      .venv_eager/bin/python \
      tests/test_qaic/spec_decode/qaic_eager_cpu_fallback_repro.py \
      --case temperature

The command emits a JSON summary after all backend diagnostics. Redirect both
stdout and stderr to preserve the complete runtime record.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import torch
import torch_qaic


SAMPLING_EPS = 1e-5


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "is_contiguous": tensor.is_contiguous(),
    }


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_qaic": getattr(torch_qaic, "__version__", "unavailable"),
        "qaic_visible_devices": os.environ.get("QAIC_VISIBLE_DEVICES"),
        "qaic_debug": os.environ.get("QAIC_DEBUG"),
        "torch_show_cpp_stacktraces": os.environ.get(
            "TORCH_SHOW_CPP_STACKTRACES"
        ),
        "qaic_cpu_fallback_ops": os.environ.get("QAIC_CPU_FALLBACK_OPS"),
        "qaic_num_cores": os.environ.get("QAIC_NUM_CORES"),
        "qaic_num_threads": os.environ.get("QAIC_NUM_THREADS"),
    }


def _dispatcher_table(schema: str) -> str:
    return torch._C._dispatch_dump_table(schema)


def _to_qaic(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device="qaic")


def _cpu_values(tensor: torch.Tensor) -> list[Any]:
    return tensor.detach().cpu().tolist()


def _equal_with_nan(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    return bool(
        torch.allclose(
            actual.detach().cpu(),
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    )


def _mismatch_count(actual: torch.Tensor, expected: torch.Tensor) -> int:
    actual_cpu = actual.detach().cpu()
    matching_nan = torch.isnan(actual_cpu) & torch.isnan(expected)
    return int(((actual_cpu != expected) & ~matching_nan).sum().item())


def _record_operation(
    name: str,
    schema: str,
    inputs: list[torch.Tensor],
    cpu_reference: torch.Tensor,
    operation: Callable[[], torch.Tensor],
    consume: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "schema": schema,
        "dispatcher_table": _dispatcher_table(schema),
        "inputs": [_tensor_metadata(tensor) for tensor in inputs],
        "cpu_reference": _cpu_values(cpu_reference),
    }
    try:
        result = operation()
        record["result"] = _tensor_metadata(result)
        record["result_values"] = _cpu_values(result)
        record["cpu_reference_parity"] = _equal_with_nan(result, cpu_reference)
        try:
            consumed = consume(result)
            record["downstream"] = {
                "succeeded": True,
                "result": _tensor_metadata(consumed),
                "values": _cpu_values(consumed),
            }
        except Exception as error:
            record["downstream"] = {
                "succeeded": False,
                "exception": repr(error),
                "traceback": traceback.format_exc(),
            }
    except Exception as error:
        record["exception"] = repr(error)
        record["traceback"] = traceback.format_exc()
    return record


def _layouts(tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    if tensor.ndim != 2:
        return {"contiguous": tensor}
    padded = torch.cat((tensor, tensor + 0.25), dim=1)
    return {
        "contiguous": tensor,
        "noncontiguous": padded[:, ::2],
    }


def _temperature_case() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for size in (1, 8, 32):
        cpu_temp = torch.tensor(
            [0.0, SAMPLING_EPS / 2, SAMPLING_EPS, 0.5, 1.0, 2.0, 4.0, 8.0]
            [:size],
            dtype=torch.float32,
        )
        if size > len(cpu_temp):
            cpu_temp = torch.linspace(0.0, 2.0, size, dtype=torch.float32)
            cpu_temp[1] = SAMPLING_EPS / 2
            cpu_temp[2] = SAMPLING_EPS
        qaic_temp = _to_qaic(cpu_temp)
        cpu_reference = torch.ops.aten.lt.Scalar(cpu_temp, SAMPLING_EPS)
        records.append(
            _record_operation(
                name=f"temperature_size_{size}",
                schema="aten::lt.Scalar",
                inputs=[qaic_temp],
                cpu_reference=cpu_reference,
                operation=lambda: torch.ops.aten.lt.Scalar(
                    qaic_temp, SAMPLING_EPS
                ),
                consume=lambda mask: torch.ops.aten.where.self(
                    mask,
                    torch.ones_like(qaic_temp),
                    qaic_temp,
                ),
            )
        )
    return records


def _comparison_case(
    name: str,
    schema: str,
    dtype: torch.dtype,
    comparison: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    threshold_dtype: torch.dtype | None = None,
) -> list[dict[str, Any]]:
    cpu_source = torch.tensor(
        [[-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0],
         [-2.0, -1.0, 0.0, 0.125, 0.25, 0.5, 1.0, 1.5]],
        dtype=dtype,
    )
    threshold_dtype = threshold_dtype or dtype
    cpu_threshold = torch.tensor([[0.5], [0.25]], dtype=threshold_dtype)
    records: list[dict[str, Any]] = []
    for layout, source in _layouts(cpu_source).items():
        threshold = cpu_threshold.clone()
        qaic_source = _to_qaic(source)
        qaic_threshold = _to_qaic(threshold)
        cpu_reference = comparison(source, threshold)
        records.append(
            _record_operation(
                name=f"{name}_{dtype}_{threshold_dtype}_{layout}",
                schema=schema,
                inputs=[qaic_source, qaic_threshold],
                cpu_reference=cpu_reference,
                operation=lambda: comparison(qaic_source, qaic_threshold),
                consume=lambda mask: qaic_source.clone().masked_fill_(mask, -9.0),
            )
        )
    return records


def _mixed_case() -> list[dict[str, Any]]:
    cpu_temp = torch.tensor(
        [0.0, SAMPLING_EPS / 2, SAMPLING_EPS, 1.0], dtype=torch.float32
    )
    cpu_greedy = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    cpu_random = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
    qaic_temp = _to_qaic(cpu_temp)
    qaic_greedy = _to_qaic(cpu_greedy)
    qaic_random = _to_qaic(cpu_random)
    cpu_mask = torch.ops.aten.lt.Scalar(cpu_temp, SAMPLING_EPS)
    cpu_reference = torch.ops.aten.where.self(cpu_mask, cpu_greedy, cpu_random)
    where_record = _record_operation(
        name="mixed_where_independent",
        schema="aten::where.self",
        inputs=[_to_qaic(cpu_mask), qaic_greedy, qaic_random],
        cpu_reference=cpu_reference,
        operation=lambda: torch.ops.aten.where.self(
            _to_qaic(cpu_mask), qaic_greedy, qaic_random
        ),
        consume=lambda result: result + 1,
    )
    combined_record = _record_operation(
        name="mixed_greedy_random",
        schema="aten::lt.Scalar",
        inputs=[qaic_temp, qaic_greedy, qaic_random],
        cpu_reference=cpu_reference,
        operation=lambda: torch.ops.aten.where.self(
            torch.ops.aten.lt.Scalar(qaic_temp, SAMPLING_EPS),
            qaic_greedy,
            qaic_random,
        ),
        consume=lambda result: result + 1,
    )
    combined_record["downstream_schema"] = "aten::where.self"
    combined_record["downstream_dispatcher_table"] = _dispatcher_table(
        "aten::where.self"
    )
    return [where_record, combined_record]


def _latency_us(samples: list[int]) -> dict[str, float]:
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "median_us": statistics.median(samples) / 1_000,
        "p95_us": ordered[p95_index] / 1_000,
    }


def _measure(
    iterations: int, operation: Callable[[], torch.Tensor]
) -> dict[str, float]:
    samples: list[int] = []
    for _ in range(iterations):
        torch.qaic.synchronize()
        started = time.perf_counter_ns()
        operation()
        torch.qaic.synchronize()
        samples.append(time.perf_counter_ns() - started)
    return _latency_us(samples)


def _warm_up(iterations: int, operation: Callable[[], torch.Tensor]) -> None:
    for _ in range(iterations):
        operation()
    torch.qaic.synchronize()


def _sampler_top_p_case(
    warmup: int,
    iterations: int,
    batch_size: int,
    vocab_size: int,
) -> list[dict[str, Any]]:
    import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_module

    upstream_top_p = topk_topp_module.apply_top_k_top_p_pytorch
    records: list[dict[str, Any]] = []

    for dtype in (torch.float16, torch.float32):
        cpu_logits = torch.linspace(
            -8.0, 8.0, batch_size * vocab_size, dtype=dtype
        ).reshape(batch_size, vocab_size)
        cpu_top_p = torch.linspace(0.5, 0.9, batch_size, dtype=torch.float32)
        cpu_reference = upstream_top_p(
            cpu_logits.clone(), None, cpu_top_p.clone()
        )

        def direct(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
            logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = torch.le(probs_sum, 1 - top_p.unsqueeze(dim=1))
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))
            return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

        def explicit_cpu_predicate(
            logits: torch.Tensor, top_p: torch.Tensor
        ) -> torch.Tensor:
            logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
            probs_sort = logits_sort.softmax(dim=-1)
            probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
            top_p_mask = torch.le(
                probs_sum.cpu(),
                (1 - top_p.unsqueeze(dim=1)).cpu(),
            ).to(device=logits.device)
            top_p_mask[:, -1] = False
            logits_sort.masked_fill_(top_p_mask, -float("inf"))
            return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

        qaic_logits = _to_qaic(cpu_logits)
        qaic_top_p = _to_qaic(cpu_top_p)

        def run_direct() -> torch.Tensor:
            return direct(qaic_logits.clone(), qaic_top_p)

        def run_explicit_transfer() -> torch.Tensor:
            return explicit_cpu_predicate(qaic_logits.clone(), qaic_top_p)

        direct_result = run_direct()
        explicit_result = run_explicit_transfer()
        _warm_up(warmup, run_direct)
        _warm_up(warmup, run_explicit_transfer)

        threshold_bytes = cpu_top_p.numel() * cpu_top_p.element_size()
        probs_bytes = cpu_logits.numel() * cpu_logits.element_size()
        records.append(
            {
                "name": f"sampler_top_p_{dtype}",
                "schema": "aten::le.Tensor",
                "batch_size": batch_size,
                "vocab_size": vocab_size,
                "implementation": {
                    "direct": "upstream apply_top_k_top_p_pytorch",
                    "explicit_reference": "local CPU top-p predicate",
                },
                "cpu_reference": _cpu_values(cpu_reference),
                "direct": {
                    "result": _tensor_metadata(direct_result),
                    "cpu_reference_parity": _equal_with_nan(
                        direct_result, cpu_reference
                    ),
                    "cpu_reference_mismatches": _mismatch_count(
                        direct_result, cpu_reference
                    ),
                    "timing": _measure(iterations, run_direct),
                    "transfer_bytes": 0,
                },
                "explicit_transfer": {
                    "result": _tensor_metadata(explicit_result),
                    "cpu_reference_parity": _equal_with_nan(
                        explicit_result, cpu_reference
                    ),
                    "cpu_reference_mismatches": _mismatch_count(
                        explicit_result, cpu_reference
                    ),
                    "matches_direct": _equal_with_nan(
                        explicit_result, direct_result.cpu()
                    ),
                    "direct_mismatches": _mismatch_count(
                        explicit_result, direct_result.cpu()
                    ),
                    "timing": _measure(iterations, run_explicit_transfer),
                    "transfer_bytes": (
                        probs_bytes + threshold_bytes + cpu_logits.numel()
                    ),
                },
            }
        )
    return records


def _run_case(
    case: str,
    warmup: int,
    iterations: int,
    batch_size: int,
    vocab_size: int,
) -> list[dict[str, Any]]:
    if case == "temperature":
        return _temperature_case()
    if case == "topk":
        return [
            *_comparison_case(
                "topk", "aten::lt.Tensor", torch.float16, torch.ops.aten.lt.Tensor
            ),
            *_comparison_case(
                "topk", "aten::lt.Tensor", torch.float32, torch.ops.aten.lt.Tensor
            ),
        ]
    if case == "topp":
        return [
            *_comparison_case(
                "topp", "aten::le.Tensor", torch.float16, torch.ops.aten.le.Tensor
            ),
            *_comparison_case(
                "topp", "aten::le.Tensor", torch.float32, torch.ops.aten.le.Tensor
            ),
            *_comparison_case(
                "topp",
                "aten::le.Tensor",
                torch.float16,
                torch.ops.aten.le.Tensor,
                threshold_dtype=torch.float32,
            ),
        ]
    if case == "mixed":
        return _mixed_case()
    if case == "sampler-topp":
        return _sampler_top_p_case(warmup, iterations, batch_size, vocab_size)
    raise ValueError(f"Unknown case: {case}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("temperature", "topk", "topp", "mixed", "sampler-topp"),
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=32)
    arguments = parser.parse_args()

    if not os.environ.get("QAIC_VISIBLE_DEVICES"):
        raise RuntimeError("Set QAIC_VISIBLE_DEVICES before starting this process")

    summary = {
        "runtime": _runtime_metadata(),
        "case": arguments.case,
        "records": _run_case(
            arguments.case,
            arguments.warmup,
            arguments.iterations,
            arguments.batch_size,
            arguments.vocab_size,
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
