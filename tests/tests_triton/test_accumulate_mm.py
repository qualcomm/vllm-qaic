"""
Standalone QAIC validation for `_accumulate_mm`.

Source under test:
vllm/lora/ops/triton_ops/fp8_kernel_utils.py
  - _accumulate_mm  (FP8 matmul-accumulate of one K-tile with dequant scales)

`_accumulate_mm` is a @triton.jit device helper used by `fp8_mm_k`. It takes
already-loaded A/B tiles and folds `tl.dot(a, b)` into the running fp32
accumulator. In block-wise mode it multiplies by per-block a/b scales inline;
in the *tensor-wise* / per-channel mode (group_k==0, group_n==0) it simply
does `accumulator = tl.dot(a, b, acc=accumulator)` and the tensor-wise dequant
scales are applied later by the CALLER (do_shrink/do_expand), not here.

We use the simplest path (tensor-wise, use_fp8_w8a8=True, group_k=group_n=0)
and wrap the helper in `_acc_wrapper` for a single K-tile.

FP8 REPRESENTATION: A and B are stored as torch.float8_e4m3fn; we compare
against a pure-PyTorch fp32 matmul of the dequantized (i.e. .float()) tile
values. Tensor-wise scales are intentionally excluded from the reference
because the helper itself does not apply them in this path.

Reference: a_tile.float() @ b_tile.float(). FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fp8_kernel_utils import _accumulate_mm
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_accumulate_mm.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fp8_kernel_utils.py"
DEVICE = "qaic"
FP8_DTYPE = torch.float8_e4m3fn

torch.manual_seed(42)

# Global shared inputs (single K-tile GEMM)
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16

# Small magnitudes so the fp8 cast is well-behaved.
A_TILE = (torch.randn(BLOCK_M, BLOCK_K, device=DEVICE) * 0.25).to(FP8_DTYPE)
B_TILE = (torch.randn(BLOCK_K, BLOCK_N, device=DEVICE) * 0.25).to(FP8_DTYPE)


@triton.jit
def _acc_wrapper(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    tiled_a = tl.load(a_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :])
    tiled_b = tl.load(b_ptr + offs_k[:, None] * BLOCK_N + offs_n[None, :])
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accumulator = _accumulate_mm(
        tiled_a,
        tiled_b,
        accumulator,
        scale_ptr,          # a_scale_ptr (unused in tensor-wise path)
        scale_ptr,          # b_scale_ptr (unused in tensor-wise path)
        0,                  # a_scale_k_stride
        0,                  # b_scale_k_stride
        0,                  # iter_k
        0,                  # group_k -> tensor-wise
        0,                  # group_n -> tensor-wise
        True,               # use_fp8_w8a8
    )
    c_ptrs = c_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    tl.store(c_ptrs, accumulator)


def pytorch_ref(a_tile, b_tile):
    return a_tile.float() @ b_tile.float()


def kernel_impl(a_tile, b_tile):
    out = torch.zeros(BLOCK_M, BLOCK_N, dtype=torch.float32, device=a_tile.device)
    scale = torch.ones(1, dtype=torch.float32, device=a_tile.device)
    _acc_wrapper[(1,)](a_tile, b_tile, out, scale, BLOCK_M, BLOCK_N, BLOCK_K)
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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(A_TILE, B_TILE)
        kernel_out = kernel_impl(A_TILE, B_TILE)
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(A_TILE.shape), tuple(B_TILE.shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(A_TILE.dtype),
            "device": str(A_TILE.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(A_TILE, B_TILE))
        kern_stats = _bench(lambda: kernel_impl(A_TILE, B_TILE))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n", error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _accumulate_mm (wrapped, tensor-wise fp8 path)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
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
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
