"""
Standalone QAIC validation for `_update_min_larger_stats`.

Source under test:
vllm/v1/sample/ops/topk_topp_triton.py
  - _update_min_larger_stats (device helper: merge per-tile (min-above-pivot,
    count) stats across vocab tiles for the top-k / top-p ternary pivot search)

The helper, per tile, computes:
  tile_min = min(data where data > pivot else sentinel)
  tile_cnt = count(data > pivot AND |data - tile_min| < 1e-9)
and merges into the running (min_larger, num_min_larger):
  is_new  = tile_min < min_larger        -> replace count
  is_same = |tile_min - min_larger| < 1e-9 -> accumulate count
  min_larger = min(min_larger, tile_min)

We wrap it in a tiny @triton.jit kernel that scans the data in BLOCK_TRUNC
tiles (single program) with `above_mask = data > pivot`, and stores the final
(min_larger, num_min_larger). The result is the smallest value strictly above
the pivot and how many times it occurs. Float compare on min_larger + exact
integer compare on the count.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import _update_min_larger_stats

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_update_min_larger_stats.txt")
KERNEL_FILE_PATH = "vllm/v1/sample/ops/topk_topp_triton.py"
KERNEL_NAME = "_update_min_larger_stats"

DEVICE = "qaic"
N = 512
BLOCK_TRUNC = 128
PIVOT = 0.0
SENTINEL = float("inf")

torch.manual_seed(42)
# Include repeated values so the count accumulation path is exercised.
_base = torch.randn(N, dtype=torch.float32)
_base[10] = 0.5
_base[20] = 0.5
_base[30] = 0.5  # a repeated "min above pivot" candidate
DATA = _base.to(DEVICE)


@triton.jit
def _stats_wrapper(
    data_ptr,
    out_min_ptr,
    out_cnt_ptr,
    n,
    pivot,
    sentinel,
    BLOCK_TRUNC: tl.constexpr,
):
    min_larger = float("inf")
    num_min_larger = tl.zeros((), dtype=tl.uint32)
    num_tiles = (n + BLOCK_TRUNC - 1) // BLOCK_TRUNC
    for i in range(0, num_tiles):
        offs = i * BLOCK_TRUNC + tl.arange(0, BLOCK_TRUNC)
        mask = offs < n
        data = tl.load(data_ptr + offs, mask=mask, other=sentinel)
        above_mask = (data > pivot) & mask
        min_larger, num_min_larger = _update_min_larger_stats(
            data, above_mask, min_larger, num_min_larger, sentinel
        )
    tl.store(out_min_ptr, min_larger)
    tl.store(out_cnt_ptr, num_min_larger.to(tl.int32))


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
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
            lines.append(
                f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")

    else:
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


def pytorch_ref(data, pivot):
    x = data.cpu().to(torch.float32)
    above = x[x > pivot]
    if above.numel() == 0:
        return float("inf"), 0
    min_larger = float(above.min().item())
    cnt = int((torch.abs(above - min_larger) < 1e-9).sum().item())
    return min_larger, cnt


def kernel_impl(data, pivot):
    out_min = torch.empty((), dtype=torch.float32, device=DEVICE)
    out_cnt = torch.empty((), dtype=torch.int32, device=DEVICE)
    _stats_wrapper[(1,)](
        data,
        out_min,
        out_cnt,
        N,
        pivot,
        SENTINEL,
        BLOCK_TRUNC=BLOCK_TRUNC,
    )
    return out_min, out_cnt


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_min, ref_cnt = pytorch_ref(DATA, PIVOT)
        out_min, out_cnt = kernel_impl(DATA, PIVOT)
        min_val = float(out_min.cpu().item())
        cnt_val = int(out_cnt.cpu().item())

        torch.testing.assert_close(
            torch.tensor(min_val), torch.tensor(ref_min), rtol=1e-3, atol=1e-3
        )
        assert cnt_val == ref_cnt, f"count mismatch: {cnt_val} vs {ref_cnt}"

        stats = {
            "input_shape": tuple(DATA.shape),
            "input_dtype": str(DATA.dtype),
            "device": str(DATA.device),
            "pivot": PIVOT,
            "min_larger_kernel": min_val,
            "min_larger_ref": ref_min,
            "num_min_larger_kernel": cnt_val,
            "num_min_larger_ref": ref_cnt,
            "min_abs_diff": abs(min_val - ref_min),
            "count_exact_match": cnt_val == ref_cnt,
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(DATA, PIVOT))
        kern_stats = _bench(lambda: kernel_impl(DATA, PIVOT))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
