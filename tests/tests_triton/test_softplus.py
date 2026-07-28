"""
Standalone QAIC validation for the `softplus` Triton device helper.

Source under test:
vllm/model_executor/layers/mamba/ops/mamba_ssm.py
  - softplus  (device helper)

`softplus` computes a numerically-stable softplus of the SSM time-step dt. The
source branch-selects on the Triton version:
    TRITON3:  tl.where(dt <= 20.0, log(exp(dt) + 1), dt)
    else:     tl.where(dt <= 20.0, log1p(exp(dt)),   dt)
Both are mathematically softplus with a large-x linear cutoff at dt > 20 where
softplus(dt) ~= dt. It is a `@triton.jit` device helper, so we wrap it in a
tiny standalone kernel that loads dt, applies softplus, and stores the result.

Reference: torch.nn.functional.softplus (with the same large-x linear branch,
which F.softplus itself implements via its `threshold` argument).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_softplus.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/mamba_ssm.py"

DEVICE = "qaic"
N = 256
BLOCK = 256

torch.manual_seed(42)
# Include values on both sides of the dt <= 20 branch cutoff.
DT = torch.linspace(-15.0, 30.0, N, dtype=torch.float32, device=DEVICE)


@triton.jit
def _softplus_wrap_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    o = softplus(x)
    tl.store(o_ptr + offs, o, mask=mask)


def pytorch_ref(dt):
    """Pure PyTorch reference: numerically-stable softplus with a linear
    branch for dt > 20 (matching the kernel's tl.where cutoff)."""
    # F.softplus(x) = log(1 + exp(x)), and for x*beta > threshold returns x.
    return torch.nn.functional.softplus(dt, beta=1.0, threshold=20.0)


def kernel_impl(dt):
    out = torch.empty_like(dt)
    grid = (triton.cdiv(N, BLOCK),)
    _softplus_wrap_kernel[grid](dt, out, N, BLOCK=BLOCK)
    return out


def _log(text):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


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


def _timing_lines(stats):
    if "pytorch_latency_ms" not in stats:
        return []
    return [
        "Timing:\n",
        f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
        f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
        f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
        f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
        f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
        f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
        f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
        f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
        f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
    ]


_TIMING_KEYS = (
    "pytorch_latency_ms",
    "kernel_latency_ms",
    "speedup_kernel_over_pytorch",
)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(DT)
        kernel_out = kernel_impl(DT)
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        rel = diff / (ref_cpu.abs() + 1e-8)
        stats = {
            "input_shape": tuple(DT.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(DT.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(DT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "max_rel_err": rel.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(DT))
        kern_stats = _bench(lambda: kernel_impl(DT))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: softplus (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                if k in _TIMING_KEYS:
                    continue
                lines.append(f"- {k}: {v}\n")
            lines += _timing_lines(stats)
        else:
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
