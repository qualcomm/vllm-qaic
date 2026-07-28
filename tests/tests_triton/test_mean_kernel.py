"""
Standalone QAIC validation for `mean_kernel`.

Source under test:
vllm/model_executor/layers/batch_invariant.py
  - mean_kernel  (deterministic single-dim mean reduction)

Reduces the input, viewed as (M, N, K), along the N dimension by summing in a
fixed order and dividing by N. This is the deterministic (batch-invariant)
replacement for aten::mean.dim. Launched via the file's `mean_dim` wrapper.

Reference: pure PyTorch x.mean(dim=...). Float32, rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.batch_invariant import mean_dim

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_mean_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/batch_invariant.py"

DEVICE = "qaic"
torch.manual_seed(42)

REDUCE_DIM = 1
X = torch.randn(8, 16, 4, dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(x, dim):
    """Pure PyTorch mean over a single dimension."""
    return x.mean(dim=dim)


def kernel_impl(x, dim):
    return mean_dim(x, dim=dim)


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
        ref_out = pytorch_ref(X, REDUCE_DIM)
        kernel_out = kernel_impl(X, REDUCE_DIM)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-12)
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(X.dtype),
            "device": str(X.device),
            "reduce_dim": REDUCE_DIM,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff / denom).max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, REDUCE_DIM))
        kern_stats = _bench(lambda: kernel_impl(X, REDUCE_DIM))
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
            "Kernel: mean_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"Input shape: {stats['input_shape']}\n")
            lines.append(f"Reduce dim: {stats['reduce_dim']}\n")
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
