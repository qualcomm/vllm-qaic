"""
Standalone QAIC validation for `tanh`.

Source under test:
vllm/v1/attention/ops/triton_decode_attention.py
  - tanh (device helper: tanh via scaled sigmoid, `2*sigmoid(2x) - 1`), used
    for attention logit capping.

`tanh` is a @triton.jit device helper. We wrap it in a tiny standalone
@triton.jit kernel that loads a block of x, calls tanh, and stores the result.
Reference: torch.tanh. Float compare.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import tanh

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_tanh_triton_decode_attention.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_decode_attention.py"
KERNEL_NAME = "tanh"

DEVICE = "qaic"
N = 1024
BLOCK = 1024

torch.manual_seed(42)
X = (torch.randn(N, dtype=torch.float32) * 4.0).to(DEVICE)


@triton.jit
def _tanh_wrapper(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tanh(x)
    tl.store(out_ptr + offs, out, mask=mask)


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
        f"Kernel: {KERNEL_NAME}\n",
        f"Kernel file: {KERNEL_FILE_PATH}\n",
        f"Device target: QAIC (device='{DEVICE}')\n",
        f"Status: {status}\n\n",
    ]
    if status == "SUCCESS":
        for k, v in stats.items():
            lines.append(f"- {k}: {v}\n")
        if "pytorch_latency_ms" in stats:
            lines.append("Timing:\n")
            lines.append(
                f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")

    else:
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


def pytorch_ref(x):
    return torch.tanh(x.cpu().to(torch.float32))


def kernel_impl(x):
    out = torch.empty_like(x)
    _tanh_wrapper[(1,)](x, out, N, BLOCK=BLOCK)
    return out


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns latency stats (ms)."""
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
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(X)
        out = kernel_impl(X)
        out_cpu = out.cpu().to(torch.float32)
        torch.testing.assert_close(out_cpu, ref, rtol=1e-3, atol=1e-3)
        diff = (out_cpu - ref).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(X.dtype),
            "output_dtype": str(out.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
