"""
Standalone QAIC validation for the `resolve_seq_and_query_len` @triton.jit
device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - resolve_seq_and_query_len(query_start_len_ptr, seq_lens_ptr,
        q_block_global_idx, num_seqs, BLOCK_Q)

Given a flattened q-block program id, it binary-searches the cumulative
query-length prefix (`find_seq_idx` in q-block mode) to recover the owning
sequence, then loads per-sequence lengths. It returns the 5-tuple:
    (seq_idx, q_block_local_idx, cur_batch_in_all_start_index,
     cur_batch_query_len, seq_len)
with the exact source arithmetic:
    seq_idx           = find_seq_idx(cu, gidx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = cu[seq_idx] // BLOCK_Q + seq_idx
    q_block_local_idx = gidx - q_block_start_idx
    cur_start         = cu[seq_idx]
    cur_batch_query_len = cu[seq_idx + 1] - cu[seq_idx]
    seq_len           = seq_lens[seq_idx]

This helper is device-side; we wrap it in a launcher kernel that stores the
five resolved integers into an int32 buffer. Comparison is EXACT integer
equality (no floating tolerance).

Reference: pure PyTorch replication of the index arithmetic + binary search.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import resolve_seq_and_query_len

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_resolve_seq_and_query_len.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_Q = 4
NUM_SEQS = 3
# cumulative query lengths (cu_seqlens): query lens [5, 3, 8]
CU = torch.tensor([0, 5, 8, 16], dtype=torch.int32, device=DEVICE)
SEQ_LENS = torch.tensor([10, 6, 20], dtype=torch.int32, device=DEVICE)
Q_BLOCK_GLOBAL_IDX = 3

torch.manual_seed(42)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns latency stats (ms)."""
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
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _find_seq_idx_ref(cu, target, num_seqs, block_q):
    """Pure-python replica of find_seq_idx in use_q_block_mode=True."""
    left, right = 0, num_seqs
    while left < right:
        mid = (left + right) // 2
        val = int(cu[mid].item())
        mid_val = val // block_q + mid
        if mid_val <= target:
            left = mid + 1
        else:
            right = mid
    return left - 1


def pytorch_ref(cu, seq_lens, gidx, num_seqs, block_q):
    """Pure PyTorch replication of resolve_seq_and_query_len."""
    cu = cu.cpu()
    seq_lens = seq_lens.cpu()
    seq_idx = _find_seq_idx_ref(cu, gidx, num_seqs, block_q)
    cur_start = int(cu[seq_idx].item())
    q_block_start_idx = cur_start // block_q + seq_idx
    q_block_local_idx = gidx - q_block_start_idx
    cur_batch_query_len = int(cu[seq_idx + 1].item()) - cur_start
    seq_len = int(seq_lens[seq_idx].item())
    return torch.tensor(
        [seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len],
        dtype=torch.int32,
    )


@triton.jit
def _resolve_launcher(
    cu_ptr, seq_lens_ptr, out_ptr, gidx, num_seqs, BLOCK_Q: tl.constexpr
):
    seq_idx, q_local, cur_start, q_len, seq_len = resolve_seq_and_query_len(
        cu_ptr, seq_lens_ptr, gidx, num_seqs, BLOCK_Q
    )
    tl.store(out_ptr + 0, seq_idx)
    tl.store(out_ptr + 1, q_local)
    tl.store(out_ptr + 2, cur_start)
    tl.store(out_ptr + 3, q_len)
    tl.store(out_ptr + 4, seq_len)


def kernel_impl(cu, seq_lens, gidx, num_seqs, block_q):
    out = torch.empty(5, dtype=torch.int32, device=cu.device)
    _resolve_launcher[(1,)](cu, seq_lens, out, gidx, num_seqs, BLOCK_Q=block_q)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(CU, SEQ_LENS, Q_BLOCK_GLOBAL_IDX, NUM_SEQS, BLOCK_Q)
        kernel_out = kernel_impl(CU, SEQ_LENS, Q_BLOCK_GLOBAL_IDX, NUM_SEQS, BLOCK_Q)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        # Integer index outputs -> exact match required.
        exact = bool(torch.equal(ref_cpu, kernel_cpu))
        assert exact, f"mismatch ref={ref_cpu.tolist()} kernel={kernel_cpu.tolist()}"

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(CU.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(CU.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(CU.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.float().mean().item()),
            "exact_match": exact,
            "resolved": kernel_cpu.tolist(),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(CU, SEQ_LENS, Q_BLOCK_GLOBAL_IDX, NUM_SEQS, BLOCK_Q))
        kern_stats = _bench(
            lambda: kernel_impl(CU, SEQ_LENS, Q_BLOCK_GLOBAL_IDX, NUM_SEQS, BLOCK_Q))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: resolve_seq_and_query_len (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- cu (cu_seqlens): {CU.cpu().tolist()}\n",
                f"- seq_lens: {SEQ_LENS.cpu().tolist()}\n",
                f"- q_block_global_idx: {Q_BLOCK_GLOBAL_IDX}\n",
                f"- BLOCK_Q: {BLOCK_Q}, num_seqs: {NUM_SEQS}\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Output (seq_idx, q_local, cur_start, q_len, seq_len):\n",
                f"- resolved: {stats['resolved']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- exact_match: {stats['exact_match']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact-match comparison)\n",
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
