"""
Standalone QAIC validation for the `round_up_128` @triton.jit device helper.

Source under test:
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
  - round_up_128(x)

`round_up_128` is a device-side helper used by `_fwd_kernel_ep_scatter_1` to
align per-expert token counts up to the nearest multiple of 128 (the DeepGEMM
contiguous-layout block size).

Exact source logic:
    y = 128
    return ((x + y - 1) // y) * y

The helper operates element-wise on a Triton vector, so we launch a minimal
`@triton.jit` wrapper (`_round_up_128_launcher`) that loads a tile of ints,
applies the helper, and stores the aligned result.

Reference: pure PyTorch  ((x + 127) // 128) * 128.  Integer output ->
exact-match comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import round_up_128
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_round_up_128.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py"

DEVICE = "qaic"
N = 8  # power of 2 for tl.arange

torch.manual_seed(42)
# Mix of exact multiples of 128, values just below/above, and 0.
X = torch.tensor(
    [0, 1, 127, 128, 129, 200, 256, 300], dtype=torch.int32, device=DEVICE
)


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
    """Pure PyTorch round-up-to-128:  ((x + 127) // 128) * 128."""
    x = x.cpu().to(torch.int64)
    return (((x + 127) // 128) * 128).to(torch.int32)


@triton.jit
def _round_up_128_launcher(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    res = round_up_128(x)
    tl.store(out_ptr + offs, res)


def kernel_impl(x):
    out = torch.empty_like(x)
    _round_up_128_launcher[(1,)](x, out, N=N)
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

        exact = bool(torch.equal(kernel_cpu, ref_cpu))
        assert exact, "round-up-128 integer mismatch"
        diff = (kernel_cpu.to(torch.int64) - ref_cpu.to(torch.int64)).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.to(torch.float32).mean().item()),
            "exact_match": exact,
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
            "Kernel: round_up_128 (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- x: {X.cpu().tolist()}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- exact_match: {stats['exact_match']}\n",
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
