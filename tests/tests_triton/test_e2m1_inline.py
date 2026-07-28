"""
Standalone QAIC validation for `_e2m1_inline`.

Source under test:
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
  - _e2m1_inline  (@triton.jit DEVICE HELPER)

`_e2m1_inline(magnitude)` decodes a 3-bit E2M1 magnitude code (0..7) to its
float value using a binary tree of comparisons (bit decomposition), giving the
grid {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}. Because it is a device helper
(not launchable on its own), we wrap it in a tiny standalone @triton.jit kernel
that loads the magnitude codes, calls the helper, and stores the float results.

Reference: pure-PyTorch E2M1 magnitude lookup over the same grid.
FLOAT/exact compare (grid values are exactly representable) at rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _e2m1_inline,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_e2m1_inline.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
)

DEVICE = "qaic"
torch.manual_seed(42)

# E2M1 magnitude grid (code 0..7 -> magnitude).
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

N = 8  # all 8 codes
CODES = torch.arange(N, dtype=torch.int32, device=DEVICE)


@triton.jit
def _e2m1_inline_wrapper(code_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    mag = tl.load(code_ptr + offs)
    val = _e2m1_inline(mag)
    tl.store(out_ptr + offs, val)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(codes):
    codes = codes.cpu().to(torch.int64)
    grid = torch.tensor(_E2M1_MAG, dtype=torch.float32)
    return grid[codes]


def kernel_impl(codes):
    out = torch.empty(N, dtype=torch.float32, device=codes.device)
    _e2m1_inline_wrapper[(1,)](codes, out, N)
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
        ref_out = pytorch_ref(CODES)
        kernel_out = kernel_impl(CODES)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(CODES.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(CODES.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": DEVICE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(CODES))
        kern_stats = _bench(lambda: kernel_impl(CODES))
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
            "Kernel: _e2m1_inline (device helper, wrapped)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- codes shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (E2M1 grid decode, exact/float rtol/atol=1e-3):\n",
                f"- out shape: {stats['output_shape']} dtype {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n"
                )
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
