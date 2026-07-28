"""
Standalone QAIC validation for `fetch_id_to_ragged_kernel`.

Source under test:
vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
  - fetch_id_to_ragged_kernel
  - fetch_id_to_ragged_triton  (launcher)

Scatters per-sequence fixed-width top-k id rows [num_seq, TOPK] into a ragged
(variable-length, concatenated) layout addressed by cumulative seq offsets. For
each seq (grid = (num_seq, blocks)):
  token_start = cumsum[seq] ; token_end = cumsum[seq+1]
  for j in [0, token_end - token_start):   # capped at TOPK
    out[token_start + j] = in_tensor[seq, j]

Integer arithmetic; validated with EXACT equality over the written ragged
region.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    fetch_id_to_ragged_triton,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fetch_id_to_ragged_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
KERNEL_NAME = "fetch_id_to_ragged_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
TOPK = 8
RAGGED_LENS = [5, 3, 6]  # ragged length per seq (each <= TOPK)
NUM_SEQ = len(RAGGED_LENS)

# Fixed-width per-seq id rows; distinct values so scatter is unambiguous.
IN_TENSOR = torch.arange(
    NUM_SEQ * TOPK, dtype=torch.int32, device=DEVICE
).reshape(NUM_SEQ, TOPK) + 100
CUMSUM = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(RAGGED_LENS), 0).tolist()),
    dtype=torch.int32,
    device=DEVICE,
)
TOTAL = int(CUMSUM[-1].item())


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


def pytorch_ref(in_tensor, cumsum):
    in_tensor = in_tensor.cpu()
    cumsum = cumsum.cpu()
    out = torch.zeros(TOTAL, dtype=torch.int32)
    for seq in range(NUM_SEQ):
        start = int(cumsum[seq].item())
        end = int(cumsum[seq + 1].item())
        token_num = end - start
        for j in range(token_num):
            if j < TOPK:
                out[start + j] = in_tensor[seq, j]
    return out


def kernel_impl(in_tensor, cumsum):
    out = torch.zeros(TOTAL, dtype=torch.int32, device=DEVICE)
    fetch_id_to_ragged_triton(in_tensor, cumsum, out, TOPK)
    return out


def _exact(ref, ker):
    ref = ref.cpu()
    ker = ker.cpu()
    mism = int((ref != ker).sum().item())
    maxdiff = (
        int((ref.to(torch.int64) - ker.to(torch.int64)).abs().max().item())
        if ref.numel() > 0
        else 0
    )
    return mism, maxdiff


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(IN_TENSOR, CUMSUM)
        ker = kernel_impl(IN_TENSOR, CUMSUM)
        mism, maxdiff = _exact(ref, ker)
        assert mism == 0, f"{mism} mismatched elements"

        stats = {
            "num_seq": NUM_SEQ,
            "topk": TOPK,
            "ragged_lens": RAGGED_LENS,
            "total_out": TOTAL,
            "in_shape": tuple(IN_TENSOR.shape),
            "dtype": str(IN_TENSOR.dtype),
            "device": str(IN_TENSOR.device),
            "mismatch_count": mism,
            "max_abs_int_diff": maxdiff,
            "grid": f"({NUM_SEQ}, cdiv({TOPK},64))",
        }
        pt_stats = _bench(lambda: pytorch_ref(IN_TENSOR, CUMSUM))
        kern_stats = _bench(lambda: kernel_impl(IN_TENSOR, CUMSUM))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            f"Kernel: {KERNEL_NAME}\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
