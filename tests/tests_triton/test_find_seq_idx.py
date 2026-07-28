"""
Standalone QAIC validation for the `find_seq_idx` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q,
        use_q_block_mode)

Binary search over the cumulative query-length prefix mapping a flattened
query-block/token index to its owning sequence index. Exact source loop:

    left = 0; right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = cu[mid]
        mid_val = val // BLOCK_Q + mid   if use_q_block_mode else val
        if mid_val <= target_idx: left = mid + 1
        else: right = mid
    return left - 1

This is a rightmost-predecessor search: returns the largest seq index whose
(transformed) prefix value is <= target_idx. Equivalent to
`searchsorted(transformed_prefix, target, side='right') - 1`.

We validate BOTH modes (use_q_block_mode True and False) at several targets.
This helper is device-side; a launcher kernel stores each result into a
buffer. Comparison is EXACT integer equality.

Reference: pure PyTorch/python replication of the binary search.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import find_seq_idx

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_find_seq_idx.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_Q = 4
NUM_SEQS = 3
# cu_seqlens style prefix; query lens [5, 3, 8]
CU = torch.tensor([0, 5, 8, 16], dtype=torch.int32, device=DEVICE)
# Targets to probe (in q-block units when use_q_block_mode=True).
TARGETS = [0, 1, 2, 3, 4, 5]

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


def _find_seq_idx_ref(cu, target, num_seqs, block_q, use_q_block_mode):
    left, right = 0, num_seqs
    while left < right:
        mid = (left + right) // 2
        val = int(cu[mid].item())
        mid_val = (val // block_q + mid) if use_q_block_mode else val
        if mid_val <= target:
            left = mid + 1
        else:
            right = mid
    return left - 1


def pytorch_ref(cu, targets, num_seqs, block_q, use_q_block_mode):
    cu = cu.cpu()
    res = [
        _find_seq_idx_ref(cu, t, num_seqs, block_q, use_q_block_mode)
        for t in targets
    ]
    return torch.tensor(res, dtype=torch.int32)


@triton.jit
def _find_launcher(
    cu_ptr,
    targets_ptr,
    out_ptr,
    num_seqs,
    n_targets,
    BLOCK_Q: tl.constexpr,
    USE_Q_BLOCK_MODE: tl.constexpr,
):
    # One program per target; grid = (n_targets,).
    pid = tl.program_id(0)
    if pid < n_targets:
        target = tl.load(targets_ptr + pid)
        idx = find_seq_idx(cu_ptr, target, num_seqs, BLOCK_Q, USE_Q_BLOCK_MODE)
        tl.store(out_ptr + pid, idx)


def kernel_impl(cu, targets, num_seqs, block_q, use_q_block_mode):
    targets_t = torch.tensor(targets, dtype=torch.int32, device=cu.device)
    out = torch.empty(len(targets), dtype=torch.int32, device=cu.device)
    _find_launcher[(len(targets),)](
        cu,
        targets_t,
        out,
        num_seqs,
        len(targets),
        BLOCK_Q=block_q,
        USE_Q_BLOCK_MODE=use_q_block_mode,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        results = {}
        max_diff = 0
        for mode in (True, False):
            ref_out = pytorch_ref(CU, TARGETS, NUM_SEQS, BLOCK_Q, mode)
            kernel_out = kernel_impl(CU, TARGETS, NUM_SEQS, BLOCK_Q, mode)
            ref_cpu = ref_out.cpu()
            kernel_cpu = kernel_out.cpu()
            assert torch.equal(ref_cpu, kernel_cpu), (
                f"mode={mode} ref={ref_cpu.tolist()} k={kernel_cpu.tolist()}"
            )
            results[mode] = kernel_cpu.tolist()
            max_diff = max(max_diff, int((kernel_cpu - ref_cpu).abs().max().item()))

        stats = {
            "input_shape": tuple(CU.shape),
            "output_shape": (len(TARGETS),),
            "in_dtype": str(CU.dtype),
            "out_dtype": "torch.int32",
            "device": str(CU.device),
            "max_abs_diff": max_diff,
            "mean_abs_diff": 0.0,
            "results": results,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(CU, TARGETS, NUM_SEQS, BLOCK_Q, True))
        kern_stats = _bench(
            lambda: kernel_impl(CU, TARGETS, NUM_SEQS, BLOCK_Q, True))
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
            "Kernel: find_seq_idx (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- cu (prefix): {CU.cpu().tolist()}\n",
                f"- targets: {TARGETS}\n",
                f"- BLOCK_Q: {BLOCK_Q}, num_seqs: {NUM_SEQS}\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Output (seq idx per target):\n",
                f"- use_q_block_mode=True:  {stats['results'][True]}\n",
                f"- use_q_block_mode=False: {stats['results'][False]}\n",
                f"- out dtype: {stats['out_dtype']}\n",
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
