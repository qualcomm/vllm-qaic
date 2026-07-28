"""
Standalone QAIC validation for `pack_bitmatrix`.

Source under test:
vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
  - pack_bitmatrix  (@triton.jit)

`pack_bitmatrix` packs a per-token top-k expert-id table into a bitmatrix used
by the OAI triton_kernels sparse-MoE routing path. For each token row t and each
of its top-k expert ids e, it sets bit (e % 32) of int32 word (e // 32) in
`bitmatrix[t]`. Invalid ids (-1) set no bit (guarded by a `valid` mask so a
negative id cannot flip bit 31). `bitmatrix` has `bm_cols = cdiv(n_experts, 32)`
uint32 words per row.

Launcher: replicated from the repo's `make_routing_data`
(BLOCK_SIZE_M=512, BLOCK_SIZE_K=32, grid = cdiv(n_rows, BLOCK_SIZE_M),
topk_ids cast to int16). We call `pack_bitmatrix` directly rather than going
through `make_routing_data`, which additionally requires the external
`triton_kernels` package.

Reference: pure PyTorch OR of (1 << (e % 32)) into word (e // 32) for each
token's valid top-k ids. Packed uint32 words are compared exactly.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.experts.gpt_oss_triton_kernels_moe import (
    pack_bitmatrix,
)
from vllm.triton_utils import triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_pack_bitmatrix.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
)

DEVICE = "qaic"
NUM_EXPERTS = 4  # <= 32 -> single int32 bitpack column
NUM_TOKENS = 8
TOP_K = 2
BLOCK_SIZE_M = 512
BLOCK_SIZE_K = 32
BM_COLS = triton.cdiv(NUM_EXPERTS, BLOCK_SIZE_K)  # = 1

torch.manual_seed(42)
# Each token picks TOP_K distinct valid experts.
TOPK_IDS = torch.stack(
    [torch.randperm(NUM_EXPERTS, device=DEVICE)[:TOP_K] for _ in range(NUM_TOKENS)]
).to(torch.int16)


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


def pytorch_ref(topk_ids):
    """Pure PyTorch bit-packing: OR (1 << e%32) into word e//32 per token."""
    topk_ids = topk_ids.cpu().to(torch.int64)
    bm = torch.zeros(NUM_TOKENS, BM_COLS, dtype=torch.int64)  # int64 avoids overflow
    for t in range(NUM_TOKENS):
        for k in range(TOP_K):
            e = int(topk_ids[t, k].item())
            if e >= 0:
                word = e // 32
                bit = e % 32
                bm[t, word] |= 1 << bit
    # match kernel's uint32 storage
    return (bm & 0xFFFFFFFF).to(torch.int64)


def kernel_impl(topk_ids):
    bitmatrix = torch.zeros(
        NUM_TOKENS, BM_COLS, dtype=torch.uint32, device=topk_ids.device
    )
    grid = (triton.cdiv(NUM_TOKENS, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        NUM_TOKENS,
        BM_COLS,
        TOP_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return bitmatrix


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(TOPK_IDS)
        kernel_out = kernel_impl(TOPK_IDS)

        # Compare packed words exactly (bring both to int64 for safe equality).
        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu().to(torch.int64)

        exact = bool(torch.equal(kernel_cpu, ref_cpu))
        assert exact, "packed bitmatrix mismatch"
        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(TOPK_IDS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(TOPK_IDS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(TOPK_IDS.device),
            "bm_cols": BM_COLS,
            "exact_match": exact,
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.to(torch.float32).mean().item()),
        }

        pt_stats = _bench(lambda: pytorch_ref(TOPK_IDS))
        kern_stats = _bench(lambda: kernel_impl(TOPK_IDS))
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
            "Kernel: pack_bitmatrix\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- topk_ids shape: {stats['input_shape']}\n",
                f"- num_tokens={NUM_TOKENS}, num_experts={NUM_EXPERTS}, "
                f"top_k={TOP_K}, bm_cols={stats['bm_cols']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- bitmatrix shape: {stats['output_shape']}\n",
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
