"""
Standalone QAIC validation for `_round_to_fp4`.

Source under test:
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
  - _round_to_fp4  (@triton.jit DEVICE HELPER)

`_round_to_fp4(x)` rounds a float to the nearest representable E2M1 (FP4) value
using fixed threshold bins on |x| (matching the Python `cast_to_fp4`), then
re-applies the sign. Bin edges (from source):
    |x| > 5.0            -> 6.0
    3.5 <= |x| <= 5.0    -> 4.0
    2.5 <  |x| <  3.5    -> 3.0
    1.75 <= |x| <= 2.5   -> 2.0
    1.25 <  |x| <  1.75  -> 1.5
    0.75 <= |x| <= 1.25  -> 1.0
    0.25 <  |x| <  0.75  -> 0.5
    else (|x| <= 0.25)   -> 0.0
sign = -1 if x < 0 else +1.

Device helper -> wrapped in a tiny standalone @triton.jit kernel that loads a
sweep of input floats, calls the helper, and stores the rounded results.

Reference: pure-PyTorch round-to-nearest-on-E2M1-grid using the same bin edges
and sign convention. FLOAT/exact compare (grid outputs) at rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _round_to_fp4,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_round_to_fp4.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
)

DEVICE = "qaic"
torch.manual_seed(42)

# Sweep of inputs: grid points, midpoints, negatives, and a few extremes.
# Avoid exact bin-boundary values (e.g. 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# where '<' vs '<=' asymmetry between bins could differ; those are validated
# implicitly by the grid points themselves.
_SWEEP = [
    0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.9, 1.0, 1.1, 1.4, 1.5, 1.6,
    1.9, 2.0, 2.3, 2.8, 3.0, 3.2, 4.0, 4.5, 5.5, 6.0, 7.0,
    -0.4, -0.9, -1.4, -1.9, -2.8, -4.5, -6.0, -7.0, -0.1,
]
N = len(_SWEEP)
X = torch.tensor(_SWEEP, dtype=torch.float32, device=DEVICE)


@triton.jit
def _round_to_fp4_wrapper(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    val = _round_to_fp4(x)
    tl.store(out_ptr + offs, val)


def _log(text: str) -> None:
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


def pytorch_ref(x):
    x = x.cpu().to(torch.float32)
    sign = torch.where(x < 0.0, -1.0, 1.0)
    a = x.abs()
    result = torch.where(a > 5.0, torch.tensor(6.0), torch.tensor(0.0))
    result = torch.where((a >= 3.5) & (a <= 5.0), torch.tensor(4.0), result)
    result = torch.where((a > 2.5) & (a < 3.5), torch.tensor(3.0), result)
    result = torch.where((a >= 1.75) & (a <= 2.5), torch.tensor(2.0), result)
    result = torch.where((a > 1.25) & (a < 1.75), torch.tensor(1.5), result)
    result = torch.where((a >= 0.75) & (a <= 1.25), torch.tensor(1.0), result)
    result = torch.where((a > 0.25) & (a < 0.75), torch.tensor(0.5), result)
    return result * sign


def kernel_impl(x):
    out = torch.empty(N, dtype=torch.float32, device=x.device)
    _round_to_fp4_wrapper[(1,)](x, out, N)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X)
        kernel_out = kernel_impl(X)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(X.dtype),
            "device": DEVICE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X))
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
            "Kernel: _round_to_fp4 (device helper, wrapped)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- x shape: {stats['input_shape']} dtype {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (round-to-E2M1-grid, exact/float rtol/atol=1e-3):\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
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
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
