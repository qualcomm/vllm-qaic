"""
Standalone QAIC validation for `_fwd_kernel_ep_scatter_1`.

Source under test:
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
  - _fwd_kernel_ep_scatter_1  (@triton.jit)

This is stage-1 of the DeepGEMM expert-parallel scatter. Given the number of
received tokens per (local) expert, it:
  1. Rounds each expert's token count up to the nearest multiple of 128
     (the DeepGEMM contiguous-layout block size) via the `round_up_128` helper.
  2. Computes the exclusive cumulative sum of those aligned counts to get each
     expert's *start offset* into the packed buffer, written to
     `expert_start_loc[e]`.
  3. Fills the DeepGEMM `m_indices` array: for each expert e, the rows
     [start .. start + real_token_count) are set to `e`; all other (padding)
     rows are left as their initial -1 (invalid, skipped by DeepGEMM).

Launcher: replicated from the repo's `ep_scatter` (grid = num_experts,
BLOCK_E = 128, BLOCK_EXPERT_NUM = next_pow2(num_experts)). `m_indices` is
pre-filled with -1 exactly as `deepgemm_moe_permute` does.

Reference: pure PyTorch aligned exclusive-cumsum + m_indices fill. Integer ->
exact-match comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    _fwd_kernel_ep_scatter_1,
)
from vllm.triton_utils import triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_deep_gemm_fwd_kernel_ep_scatter_1.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py"

DEVICE = "qaic"
BLOCK_E = 128
NUM_EXPERTS = 4

torch.manual_seed(42)
# Received token counts per local expert (includes an empty expert).
NUM_RECV = torch.tensor([3, 5, 0, 2], dtype=torch.int32, device=DEVICE)


def _aligned_counts(num_recv):
    return ((num_recv.to(torch.int64) + (BLOCK_E - 1)) // BLOCK_E) * BLOCK_E


def _m_sum(num_recv):
    return int(_aligned_counts(num_recv).sum().item())


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


def pytorch_ref(num_recv):
    """Pure PyTorch: aligned exclusive-cumsum offsets + m_indices fill."""
    num_recv = num_recv.cpu().to(torch.int64)
    aligned = _aligned_counts(num_recv)
    # exclusive cumsum
    starts = torch.cumsum(aligned, 0) - aligned
    m_sum = int(aligned.sum().item())
    m_indices = torch.full((m_sum,), -1, dtype=torch.int32)
    for e in range(num_recv.numel()):
        start = int(starts[e].item())
        cnt = int(num_recv[e].item())
        if cnt > 0:
            m_indices[start : start + cnt] = e
    return starts.to(torch.int32), m_indices


def kernel_impl(num_recv):
    m_sum = _m_sum(num_recv)
    expert_start_loc = torch.empty(NUM_EXPERTS, dtype=torch.int32, device=num_recv.device)
    # DeepGEMM sentinel: padding rows must stay -1 (skipped).
    m_indices = torch.full((m_sum,), -1, dtype=torch.int32, device=num_recv.device)
    _fwd_kernel_ep_scatter_1[(NUM_EXPERTS,)](
        num_recv,
        expert_start_loc,
        m_indices,
        num_experts=NUM_EXPERTS,
        num_warps=8,
        BLOCK_E=BLOCK_E,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(NUM_EXPERTS),
    )
    return expert_start_loc, m_indices


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_start, ref_m = pytorch_ref(NUM_RECV)
        k_start, k_m = kernel_impl(NUM_RECV)

        ref_start_cpu, ref_m_cpu = ref_start.cpu(), ref_m.cpu()
        k_start_cpu, k_m_cpu = k_start.cpu(), k_m.cpu()

        start_ok = bool(torch.equal(k_start_cpu, ref_start_cpu))
        m_ok = bool(torch.equal(k_m_cpu, ref_m_cpu))
        assert start_ok, "expert_start_loc mismatch"
        assert m_ok, "m_indices mismatch"

        stats = {
            "num_recv": NUM_RECV.cpu().tolist(),
            "start_shape": tuple(k_start.shape),
            "m_indices_shape": tuple(k_m.shape),
            "dtype": str(k_start.dtype),
            "device": str(k_start.device),
            "start_exact_match": start_ok,
            "m_indices_exact_match": m_ok,
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
        }

        pt_stats = _bench(lambda: pytorch_ref(NUM_RECV))
        kern_stats = _bench(lambda: kernel_impl(NUM_RECV))
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
            "Kernel: _fwd_kernel_ep_scatter_1\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- num_recv_tokens_per_expert: {stats['num_recv']}\n",
                f"- num_experts: {NUM_EXPERTS}, BLOCK_E: {BLOCK_E}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- expert_start_loc shape: {stats['start_shape']}\n",
                f"- m_indices shape: {stats['m_indices_shape']}\n",
                f"- start_exact_match: {stats['start_exact_match']}\n",
                f"- m_indices_exact_match: {stats['m_indices_exact_match']}\n",
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
