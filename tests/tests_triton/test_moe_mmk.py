"""
Standalone QAIC validation for the `moe_mmk` @triton.jit device helper.

Source under test:
vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py
  - moe_mmk(...)

`moe_mmk` is the inner block-tile matmul-accumulate primitive shared by the
batched MoE GEMM kernels. Given pre-computed A/B pointer grids it iterates over
the K dimension accumulating `A @ B` into an fp32 accumulator, with optional
fp8 (w8a8) / int8 (w8a16) block / per-token / tensor scaling paths.

Quantization path chosen: UNQUANTIZED (use_w8a8=False, use_w8a16=False,
per_act_token_quant=False, group_n=0, group_k=0). On this path every scale
pointer/branch is skipped and the helper computes a plain `accumulator +=
tl.dot(a, b)` reduction, converting to compute_type (fp32) at the end. This is
the simplest faithful exercise of the primitive.

Because `moe_mmk` consumes pointer grids (not tensors) we author a minimal
`@triton.jit` wrapper (`_moe_mmk_launcher`) that builds `a_ptrs`/`b_ptrs`
exactly like `expert_triton_kernel` does, calls `moe_mmk`, and stores the
resulting tile.

Reference: pure PyTorch  A @ B  (fp32).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import moe_mmk
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_moe_mmk.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py"

DEVICE = "qaic"
M = 8
N = 16
K = 16
BLOCK_M = 8
BLOCK_N = 16
BLOCK_K = 16

torch.manual_seed(42)
A = torch.randn(M, K, dtype=torch.float32, device=DEVICE)  # [M, K]
B = torch.randn(K, N, dtype=torch.float32, device=DEVICE)  # [K, N]


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


def pytorch_ref(a, b):
    """Pure PyTorch unquantized matmul:  A @ B."""
    return a.cpu().to(torch.float32) @ b.cpu().to(torch.float32)


@triton.jit
def _moe_mmk_launcher(
    a_ptr,
    b_ptr,
    c_ptr,
    K,
    stride_am: tl.int64,
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_bn: tl.int64,
    stride_cm: tl.int64,
    stride_cn: tl.int64,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = moe_mmk(
        a_ptrs,
        b_ptrs,
        K,
        0,  # expert_id (unused on unquantized path)
        a_ptr,  # a_scale_ptr placeholder (never dereferenced)
        b_ptr,  # b_scale_ptr placeholder (never dereferenced)
        stride_ak,
        stride_bk,
        0,  # stride_ase
        0,  # stride_asm
        0,  # stride_ask
        0,  # stride_bse
        0,  # stride_bsk
        0,  # stride_bsn
        offs_m,
        offs_n,
        offs_n,  # offs_bn
        mask_m,
        0,  # group_n
        0,  # group_k
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        compute_type,
        False,  # use_w8a8
        False,  # use_w8a16
        False,  # per_act_token_quant
    )

    offs_cn = tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def kernel_impl(a, b):
    c = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    _moe_mmk_launcher[(1,)](
        a,
        b,
        c,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        M,
        N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        compute_type=tl.float32,
    )
    return c


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(A, B)
        kernel_out = kernel_impl(A, B)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().max().clamp_min(1e-12)
        stats = {
            "a_shape": tuple(A.shape),
            "b_shape": tuple(B.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(A.dtype),
            "device": str(A.device),
            "quant_path": "unquantized",
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff.max() / denom).item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(A, B))
        kern_stats = _bench(lambda: kernel_impl(A, B))
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
            "Kernel: moe_mmk (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- A shape: {stats['a_shape']}\n",
                f"- B shape: {stats['b_shape']}\n",
                f"- quant path: {stats['quant_path']}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- rel_error: {stats['rel_error']}\n",
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
