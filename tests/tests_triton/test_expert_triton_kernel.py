"""
Standalone QAIC validation for `expert_triton_kernel`.

Source under test:
vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py
  - expert_triton_kernel  (@triton.jit)

`expert_triton_kernel` computes a single expert's GEMM `C = A @ B` for a tile of
up to BLOCK_M tokens. It builds the A/B pointer grids, delegates the K-loop to
the `moe_mmk` primitive, and stores the resulting [M, N] tile into the output
cache. In production it is invoked once per (expert, tile) by
`batched_triton_kernel`; here we launch it directly with grid = (1,) for a
single small expert.

Quantization path: UNQUANTIZED (use_fp8_w8a8=False, use_int8_w8a16=False,
per_act_token_quant=False, group_n=group_k=0). Scale pointers are passed as
placeholders and never dereferenced on this path.

A: [M, K], B: [K, N] (contiguous), C: [M, N].

Reference: pure PyTorch  A @ B  (fp32).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    expert_triton_kernel,
)
from vllm.triton_utils import tl

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_expert_triton_kernel.txt")
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
    """Pure PyTorch single-expert GEMM:  A @ B."""
    return a.cpu().to(torch.float32) @ b.cpu().to(torch.float32)


def kernel_impl(a, b):
    c = torch.zeros(M, N, dtype=torch.float32, device=a.device)
    # offs_bn as used by the source when launched from batched kernel.
    import torch as _t

    offs_bn = _t.arange(0, BLOCK_N, device=a.device, dtype=_t.int64) % N
    expert_triton_kernel[(1,)](
        a,
        b,
        c,
        0,  # expert_id
        tl.float32,  # compute_type
        M,
        N,
        K,
        a,  # a_scale_ptr placeholder
        b,  # b_scale_ptr placeholder
        None,  # b_zp_ptr
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        0,  # stride_ase
        0,  # stride_asm
        0,  # stride_ask
        0,  # stride_bse
        0,  # stride_bsk
        0,  # stride_bsn
        offs_bn,
        0,  # group_n
        0,  # group_k
        False,  # use_fp8_w8a8
        False,  # use_int8_w8a16
        False,  # per_act_token_quant
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
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
            "Kernel: expert_triton_kernel\n",
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
