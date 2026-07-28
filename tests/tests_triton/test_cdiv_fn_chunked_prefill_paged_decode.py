"""
Standalone QAIC validation for `cdiv_fn` (device helper).

Source under test:
vllm/v1/attention/ops/chunked_prefill_paged_decode.py
  - cdiv_fn  (ceiling division helper)

`cdiv_fn(x, y)` computes ceiling division `(x + y - 1) // y`. It is a
`@triton.jit` DEVICE HELPER used inside the chunked-prefill/paged-decode
attention kernel to size tile loops, so we wrap it in a tiny standalone
`@triton.jit` kernel that evaluates it elementwise over a sweep of (a, b).

Reference: pure PyTorch ceiling division. Integer exact comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.chunked_prefill_paged_decode import cdiv_fn

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_cdiv_fn_chunked_prefill_paged_decode.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/chunked_prefill_paged_decode.py"

DEVICE = "qaic"
torch.manual_seed(42)

# Sweep of numerators and matching divisors (all positive, exact-int math).
A = torch.tensor([1, 7, 8, 9, 15, 16, 31, 64, 100], dtype=torch.int32, device=DEVICE)
B = torch.tensor([4, 4, 8, 8, 16, 16, 16, 32, 7], dtype=torch.int32, device=DEVICE)
N = A.numel()


@triton.jit
def _cdiv_wrapper(a_ptr, b_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, cdiv_fn(a, b), mask=mask)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(a, b):
    """Pure PyTorch ceiling division: (a + b - 1) // b."""
    a = a.cpu().to(torch.int64)
    b = b.cpu().to(torch.int64)
    return torch.div(a + b - 1, b, rounding_mode="floor")


def kernel_impl(a, b):
    out = torch.empty_like(a)
    BLOCK_SIZE = triton.next_power_of_2(a.numel())
    _cdiv_wrapper[(1,)](a, b, out, a.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return out


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns dict of latency stats (ms).

    Uses time.perf_counter with torch.qaic.synchronize() because
    torch.Event-based timing is broken on the QAIC backend in this env.
    """
    import time

    import numpy as np

    def _sync():
        try:
            torch.qaic.synchronize()
        except Exception:
            pass

    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        _sync()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(A, B)
        kernel_out = kernel_impl(A, B)

        kernel_cpu = kernel_out.cpu().to(torch.int64)

        assert torch.equal(kernel_cpu, ref_out), "cdiv_fn mismatch"

        stats = {
            "input_shape": (tuple(A.shape), tuple(B.shape)),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(A.dtype),
            "device": str(A.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
        }
        pt_stats = _bench(lambda: pytorch_ref(A, B))
        kern_stats = _bench(lambda: kernel_impl(A, B))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: cdiv_fn (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"Input shapes (a,b): {stats['input_shape']}\n")
            lines.append(f"Output shape: {stats['output_shape']}\n")
            lines.append(f"Dtype: {stats['dtype']}\n")
            lines.append(f"Device: {stats['device']}\n")
            lines.append(f"Max abs diff: {stats['max_abs_diff']}\n")
            lines.append(f"Mean abs diff: {stats['mean_abs_diff']}\n")
            lines.append("Rel error: exact-integer compare\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
