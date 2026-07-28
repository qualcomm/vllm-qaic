"""
Standalone QAIC validation for `matmul_kernel_persistent`.

Source under test:
vllm/model_executor/layers/batch_invariant.py
  - matmul_kernel_persistent  (persistent-CTA GEMM, deterministic mm/addmm)

Computes C = A @ B [+ bias] using a persistent-CTA GEMM with a fixed
sequential K-reduction order, the deterministic (batch-invariant) replacement
for torch.mm / torch.addmm. Launched through the file's `matmul_persistent`
wrapper.

Reference: pure PyTorch A @ B (+ bias). Float32, rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.batch_invariant import matmul_persistent

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_matmul_kernel_persistent.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/batch_invariant.py"

DEVICE = "qaic"
torch.manual_seed(42)

M, K, N = 64, 64, 64
A = torch.randn(M, K, dtype=torch.float32, device=DEVICE)
B = torch.randn(K, N, dtype=torch.float32, device=DEVICE)
BIAS = torch.randn(N, dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(a, b, bias):
    """Pure PyTorch GEMM with optional 1D bias."""
    out = a @ b
    if bias is not None:
        out = out + bias
    return out


def kernel_impl(a, b, bias):
    return matmul_persistent(a, b, bias=bias)


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
        ref_out = pytorch_ref(A, B, BIAS)
        kernel_out = kernel_impl(A, B, BIAS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-12)
        stats = {
            "input_shape": (tuple(A.shape), tuple(B.shape), tuple(BIAS.shape)),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(A.dtype),
            "device": str(A.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff / denom).max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(A, B, BIAS))
        kern_stats = _bench(lambda: kernel_impl(A, B, BIAS))
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
            "Kernel: matmul_kernel_persistent\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"Input shapes (A,B,bias): {stats['input_shape']}\n")
            lines.append(f"Output shape: {stats['output_shape']}\n")
            lines.append(f"Dtype: {stats['dtype']}\n")
            lines.append(f"Device: {stats['device']}\n")
            lines.append(f"Max abs diff: {stats['max_abs_diff']}\n")
            lines.append(f"Mean abs diff: {stats['mean_abs_diff']}\n")
            lines.append(f"Rel error: {stats['rel_error']}\n")
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
