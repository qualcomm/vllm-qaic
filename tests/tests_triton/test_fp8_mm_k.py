"""
Standalone QAIC validation for `fp8_mm_k`.

Source under test:
vllm/lora/ops/triton_ops/fp8_kernel_utils.py
  - fp8_mm_k  (tiled K-looped FP8 matmul device helper w/ dequant scales)

`fp8_mm_k` is a @triton.jit device helper (used by do_shrink_kernel_fp8 /
do_expand_kernel_fp8). It walks the K dimension, loading A/B tiles (with
masking on the tail) and calling `_accumulate_mm`. In the *tensor-wise* path
(group_k==0, group_n==0) it accumulates dot products in fp32 and the caller
applies the a/b tensor-wise scales afterwards. We reproduce that: the wrapper
applies `a_scale * b_scale` after the accumulator returns, matching
do_shrink/do_expand behaviour.

FP8 REPRESENTATION: A/B stored as torch.float8_e4m3fn. Reference computes the
dequantized matmul  (a.float() * a_scale) @ (b.float() * b_scale).

Reference: dequant(A) @ dequant(B). FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fp8_kernel_utils import fp8_mm_k
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fp8_mm_k.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fp8_kernel_utils.py"
DEVICE = "qaic"
FP8_DTYPE = torch.float8_e4m3fn

torch.manual_seed(42)

# Global shared inputs
M = 16
N = 16
K = 16
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
EVEN_K = K % BLOCK_K == 0
A_SCALE = 0.75
B_SCALE = 1.25

A = (torch.randn(M, K, device=DEVICE) * 0.25).to(FP8_DTYPE)
B = (torch.randn(K, N, device=DEVICE) * 0.25).to(FP8_DTYPE)


@triton.jit
def _fp8_mm_k_wrapper(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = fp8_mm_k(
        a_ptrs,
        b_ptrs,
        a_scale_ptr,
        b_scale_ptr,
        stride_ak,
        stride_bk,
        0,          # a_scale_k_stride
        0,          # b_scale_k_stride
        offs_k,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        1,          # SPLIT_K
        0,          # group_k -> tensor-wise
        0,          # group_n -> tensor-wise
        True,       # use_fp8_w8a8
        False,      # per_channel_quant
        False,      # CAST_TYPE
        tl.float8e4nv,
        False,      # USE_GDC
        0,          # base_k
    )
    # tensor-wise dequant applied by caller (see do_shrink_kernel_fp8)
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr)
    acc = acc * a_scale * b_scale
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def pytorch_ref(a, b):
    return (a.float() * A_SCALE) @ (b.float() * B_SCALE)


def kernel_impl(a, b):
    out = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    a_scale = torch.tensor([A_SCALE], dtype=torch.float32, device=a.device)
    b_scale = torch.tensor([B_SCALE], dtype=torch.float32, device=a.device)
    _fp8_mm_k_wrapper[(1,)](
        a,
        b,
        out,
        a_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
    )
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
        ref_out = pytorch_ref(A, B)
        kernel_out = kernel_impl(A, B)
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(A.shape), tuple(B.shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(A.dtype),
            "device": str(A.device),
            "a_scale": A_SCALE,
            "b_scale": B_SCALE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(A, B))
        kern_stats = _bench(lambda: kernel_impl(A, B))
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
            "Kernel: fp8_mm_k (wrapped, tensor-wise)\n",
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
