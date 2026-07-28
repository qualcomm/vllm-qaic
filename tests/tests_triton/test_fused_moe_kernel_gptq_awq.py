"""
Standalone QAIC validation for `fused_moe_kernel_gptq_awq`.

Source under test:
vllm/model_executor/layers/fused_moe/fused_moe.py
  - fused_moe_kernel_gptq_awq  (@triton.jit)

This is the fused MoE GEMM that dequantizes GPTQ/AWQ grouped low-bit expert
weights on the fly, then multiplies each token by its assigned expert's
(dequantized) weight matrix.

Simplifications used here (smallest faithful example, documented layout):
  * int4 weights (use_int4_w4a16=True), NO zero point (has_zp=False).
    With has_zp False and int4, the kernel subtracts a fixed zero of 8:
        w = (nibble - 8) * scale
  * a single quant group (group_size == K), so B_scale has one column.

Weight packing (matching the kernel's addressing
`b_ptr + off_e*stride_be + (offs_k//2)*stride_bk + offs_bn*stride_bn`
with `b_shifter = (offs_k % 2) * 4`):
  * B is uint8 of shape (E, N, K//2). Byte at (e, n, j) holds two nibbles
    along the K axis: low nibble -> k=2j (even), high nibble -> k=2j+1 (odd).
  * B_scale is fp32 of shape (E, N, K//group_size) == (E, N, 1).

Launcher: the repo's own `invoke_fused_moe_wna16_triton_kernel`.
Metadata: the repo's own `moe_align_block_size`.

Reference: dequantize (unpack nibbles, subtract 8, multiply by group scale)
then out[m, slot, :] = A[m] @ W_dequant[e].T with e = topk_ids[m, slot].
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_moe_kernel_gptq_awq.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/fused_moe.py"

# Global inputs (tiny)
NUM_EXPERTS = 4
TOP_K = 2
NUM_TOKENS = 8
K_DIM = 16  # hidden (multiple of 2 for int4 packing)
N_DIM = 16  # intermediate
GROUP_SIZE = 16  # == K_DIM -> single quant group
BLOCK_M = 16
DEVICE = "qaic"

_IS_CHILD = os.environ.get("FMKGA_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        invoke_fused_moe_wna16_triton_kernel,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.triton_utils import tl

    torch.manual_seed(42)
    A = torch.randn(NUM_TOKENS, K_DIM, dtype=torch.float32, device=DEVICE)
    # Random nibbles in [0, 15], shape (E, N, K)
    NIBBLES = torch.randint(
        0, 16, (NUM_EXPERTS, N_DIM, K_DIM), dtype=torch.int64, device=DEVICE
    )
    # Pack along K: byte j holds nibble(2j) low, nibble(2j+1) high.
    low = NIBBLES[:, :, 0::2]
    high = NIBBLES[:, :, 1::2]
    B_PACKED = (low | (high << 4)).to(torch.uint8).contiguous()  # (E, N, K//2)
    NUM_GROUPS = K_DIM // GROUP_SIZE
    B_SCALE = (
        torch.rand(NUM_EXPERTS, N_DIM, NUM_GROUPS, dtype=torch.float32, device=DEVICE)
        * 0.1
        + 0.05
    )
    TOPK_IDS = torch.stack(
        [
            torch.randperm(NUM_EXPERTS, device=DEVICE)[:TOP_K]
            for _ in range(NUM_TOKENS)
        ]
    ).to(torch.int32)


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


def pytorch_ref(a, nibbles, b_scale, topk_ids):
    """Pure PyTorch: dequantize int4 (nibble-8)*scale, then MoE GEMM."""
    a = a.cpu()
    nibbles = nibbles.cpu().to(torch.float32)
    b_scale = b_scale.cpu()
    topk_ids = topk_ids.cpu()

    # group index along K for each k: k // GROUP_SIZE
    group_idx = (torch.arange(K_DIM) // GROUP_SIZE).long()  # (K,)
    # W_dequant[e, n, k] = (nibble - 8) * scale[e, n, group_idx[k]]
    scale_per_k = b_scale[:, :, group_idx]  # (E, N, K)
    w_dequant = (nibbles - 8.0) * scale_per_k  # (E, N, K)

    out = torch.zeros(NUM_TOKENS, TOP_K, N_DIM, dtype=torch.float32)
    for m in range(NUM_TOKENS):
        for k in range(TOP_K):
            e = int(topk_ids[m, k].item())
            out[m, k, :] = a[m] @ w_dequant[e].t()  # (K,) @ (K,N) -> (N,)
    return out


def kernel_impl(a, b_packed, b_scale, topk_ids):
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, BLOCK_M, NUM_EXPERTS
    )
    c = torch.zeros(NUM_TOKENS, TOP_K, N_DIM, dtype=torch.float32, device=a.device)
    config = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": N_DIM,
        "BLOCK_SIZE_K": K_DIM,
        "GROUP_SIZE_M": 1,
    }
    invoke_fused_moe_wna16_triton_kernel(
        a,
        b_packed,
        c,
        b_scale,
        None,  # B_zp (no zero point)
        None,  # topk_weights
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        False,  # mul_routed_weight
        TOP_K,
        config,
        tl.float32,  # compute_type
        False,  # use_int8_w8a16
        True,  # use_int4_w4a16
        [0, GROUP_SIZE],  # block_shape: [0, group_size]
    )
    return c


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(A, NIBBLES, B_SCALE, TOPK_IDS)
        kernel_out = kernel_impl(A, B_PACKED, B_SCALE, TOPK_IDS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(A.shape),
            "packed_weight_shape": tuple(B_PACKED.shape),
            "scale_shape": tuple(B_SCALE.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(A.dtype),
            "device": str(A.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff.max() / (ref_cpu.abs().max() + 1e-12)).item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(A, NIBBLES, B_SCALE, TOPK_IDS))
        kern_stats = _bench(lambda: kernel_impl(A, B_PACKED, B_SCALE, TOPK_IDS))
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
            "Kernel: fused_moe_kernel_gptq_awq\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- A (tokens) shape: {stats['input_shape']}\n")
            lines.append(f"- B packed (int4) shape: {stats['packed_weight_shape']}\n")
            lines.append(f"- B_scale shape: {stats['scale_shape']}\n")
            lines.append(
                f"- num_experts={NUM_EXPERTS}, top_k={TOP_K}, group_size={GROUP_SIZE}\n"
            )
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_error: {stats['rel_error']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("FMKGA_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, FMKGA_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: fused_moe_kernel_gptq_awq\n"
            f"Kernel file: {KERNEL_FILE_PATH}\n"
            f"Device target: QAIC (device='{DEVICE}')\n"
            "Status: FAILURE\n\n"
            "Error:\n"
            f"Child killed by signal (exit {proc.returncode}) during "
            "Triton->Hexagon compile/execution.\n"
            "\n------------------------------------\n\n"
        )
    sys.exit(proc.returncode if proc.returncode >= 0 else 1)


if __name__ == "__main__":
    _run_with_crash_guard()
