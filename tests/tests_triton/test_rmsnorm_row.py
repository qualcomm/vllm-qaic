"""
Standalone QAIC validation for the `_rmsnorm_row` @triton.jit device helper.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py
  - _rmsnorm_row(x, w_ptr, out_row_ptr, block, mask, eps, HIDDEN)

`_rmsnorm_row` is a device-side helper (the shared RMSNorm body used by both
_fused_mtp_input_rmsnorm_kernel and _mtp_shared_head_rmsnorm_kernel). It takes
an already-loaded fp32 register row `x`, computes:
    variance = sum(x*x) / HIDDEN
    rrms     = rsqrt(variance + eps)
    w        = load(w_ptr + block)
    y        = x * rrms * w
and stores y at out_row_ptr[block]. Since it has no standalone launcher, we
wrap it in a minimal @triton.jit kernel (`_rmsnorm_row_launcher`) that loads one
row, calls the helper, and stores the result (mirrors test_apply_softcap.py).

Config tested: single row, HIDDEN=192 (BLOCK_SIZE=256, mask exercised), fp32.
Reference: pure PyTorch RMSNorm  x * rsqrt(mean(x^2)+eps) * w.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_rmsnorm_row.txt")
KERNEL_FILE_PATH = (
    "vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (  # noqa: E402
    _rmsnorm_row,
)

torch.manual_seed(42)

HIDDEN = 192
BLOCK_SIZE = 256  # next_power_of_2(HIDDEN); tail is masked out
EPS = 1e-6

# ---- Global shared inputs (used by BOTH implementations) ----
X = torch.randn(HIDDEN, dtype=torch.float32, device=DEVICE)
WEIGHT = torch.randn(HIDDEN, dtype=torch.float32, device=DEVICE)


@triton.jit
def _rmsnorm_row_launcher(
    x_ptr,
    w_ptr,
    out_ptr,
    eps,
    HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < HIDDEN
    x = tl.load(x_ptr + block, mask=mask, other=0.0)
    _rmsnorm_row(x, w_ptr, out_ptr, block, mask, eps, HIDDEN)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


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


def pytorch_ref(x, weight, eps):
    """Pure PyTorch RMSNorm of a single row."""
    xf = x.cpu().float()
    var = (xf * xf).sum() / HIDDEN
    rrms = torch.rsqrt(var + eps)
    return xf * rrms * weight.cpu().float()


def kernel_impl(x, weight, eps):
    """Kernel launch only (wraps the _rmsnorm_row device helper)."""
    out = torch.empty(HIDDEN, dtype=torch.float32, device=x.device)
    _rmsnorm_row_launcher[(1,)](
        x, weight, out, eps, HIDDEN=HIDDEN, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X, WEIGHT, EPS)
        kernel_out = kernel_impl(X, WEIGHT, EPS)

        ref_cpu = ref_out.cpu().float()
        kernel_cpu = kernel_out.cpu().float()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(X, WEIGHT, EPS))
        kern_stats = _bench(lambda: kernel_impl(X, WEIGHT, EPS))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _rmsnorm_row (device helper, wrapped)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
