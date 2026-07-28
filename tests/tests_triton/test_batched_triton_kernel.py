"""
Standalone QAIC validation for `batched_triton_kernel`.

Source under test:
vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py
  - batched_triton_kernel  (@triton.jit)

`batched_triton_kernel` is the grid-dispatch driver for the batched-experts MoE
GEMM. The activation/weight/output tensors are laid out per-expert:
    A: [E, max_num_tokens, K]   B: [E, N, K]   C: [E, max_num_tokens, N]
Grid axis 0 is the expert id; axis 1 enumerates (M_block, N_block) tiles.
Each program loads that expert's token count from `expert_num_tokens[e]`,
early-exits if the expert is empty or the tile is out of range, then calls
`expert_triton_kernel` to compute `C[e] = A[e] @ B[e].T` for the tile.

Launcher: the repo's own `invoke_moe_batched_triton_kernel`.
Quantization path: UNQUANTIZED (use_fp8_w8a8 / use_int8_w8a16 / use_int4_w4a16
all False, per_act_token_quant=False, block_shape=None).

Reference: pure PyTorch per-expert batched matmul honoring expert_num_tokens:
    for each expert e, rows [0, expert_num_tokens[e]) of C[e] = A[e,:n] @ B[e].T.
Empty experts leave C[e] as its initialized zeros (the kernel skips them).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.experts.fused_batched_moe import (
    invoke_moe_batched_triton_kernel,
)
from vllm.triton_utils import tl

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_batched_triton_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py"

DEVICE = "qaic"
E = 4
MAX_TOKENS = 8
K = 16
N = 16
BLOCK_M = 8
BLOCK_N = 16
BLOCK_K = 16

torch.manual_seed(42)
A = torch.randn(E, MAX_TOKENS, K, dtype=torch.float32, device=DEVICE)
B = torch.randn(E, N, K, dtype=torch.float32, device=DEVICE)  # [E, N, K]
# Per-expert real token counts (includes an empty expert e=2).
EXPERT_NUM_TOKENS = torch.tensor([5, 8, 0, 3], dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(a, b, expert_num_tokens):
    """Pure PyTorch per-expert batched matmul: C[e,:n] = A[e,:n] @ B[e].T."""
    a = a.cpu().to(torch.float32)
    b = b.cpu().to(torch.float32)
    ent = expert_num_tokens.cpu()
    c = torch.zeros(E, MAX_TOKENS, N, dtype=torch.float32)
    for e in range(E):
        n = int(ent[e].item())
        if n > 0:
            c[e, :n, :] = a[e, :n, :] @ b[e].t()
    return c


def kernel_impl(a, b, expert_num_tokens):
    c = torch.zeros(E, MAX_TOKENS, N, dtype=torch.float32, device=a.device)
    config = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
    }
    invoke_moe_batched_triton_kernel(
        A=a,
        B=b,
        C=c,
        expert_num_tokens=expert_num_tokens,
        compute_type=tl.float32,
        A_scale=None,
        B_scale=None,
        B_zp=None,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        config=config,
        per_act_token_quant=False,
        block_shape=None,
    )
    return c


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(A, B, EXPERT_NUM_TOKENS)
        kernel_out = kernel_impl(A, B, EXPERT_NUM_TOKENS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().max().clamp_min(1e-12)
        stats = {
            "a_shape": tuple(A.shape),
            "b_shape": tuple(B.shape),
            "output_shape": tuple(kernel_out.shape),
            "expert_num_tokens": EXPERT_NUM_TOKENS.cpu().tolist(),
            "dtype": str(A.dtype),
            "device": str(A.device),
            "quant_path": "unquantized",
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff.max() / denom).item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(A, B, EXPERT_NUM_TOKENS))
        kern_stats = _bench(lambda: kernel_impl(A, B, EXPERT_NUM_TOKENS))
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
            "Kernel: batched_triton_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- A shape: {stats['a_shape']}\n",
                f"- B shape: {stats['b_shape']}\n",
                f"- expert_num_tokens: {stats['expert_num_tokens']}\n",
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
