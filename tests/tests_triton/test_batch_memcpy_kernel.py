"""
Standalone QAIC validation for `batch_memcpy_kernel`.

Source under test:
vllm/v1/worker/mamba_utils.py
  - batch_memcpy_kernel  (launched via `batch_memcpy`)

Generic batched byte-wise memory copy. Grid = (batch,). For item `pid`:
  src_ptr = src_ptrs[pid]; dst_ptr = dst_ptrs[pid]; size = sizes[pid] (bytes)
It copies `size` bytes from src_ptr to dst_ptr as uint8, in BLOCK_SIZE chunks.
Used to bulk-copy mamba state blocks between physical KV-cache locations.

POINTER ARRAYS: this kernel dereferences arrays of raw addresses (int64). We
build the src/dst pointer arrays ourselves from the data_ptr() of a set of
distinct source/destination row tensors (kept alive at module scope), and pass
per-item byte sizes -- exactly the contract of the repo's `batch_memcpy`
launcher, which we invoke directly. pytorch_ref copies src[i][:n_i] ->
dst[i][:n_i] for each item.

Integer-exact validation of the destination buffers after copy.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_batch_memcpy_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/mamba_utils.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.mamba_utils import batch_memcpy

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
BATCH = 4
ROW_LEN = 16  # int32 elements per row.
ELEM_SIZE = 4  # int32 = 4 bytes.
# Number of ELEMENTS to copy per item (<= ROW_LEN); byte size = n * ELEM_SIZE.
COPY_ELEMS = [16, 8, 4, 12]

# Distinct source rows (unique values) and zeroed destination rows.
# Keep alive at module scope so data_ptr() stays valid.
SRC_ROWS = [
    torch.arange(i * 1000, i * 1000 + ROW_LEN, dtype=torch.int32, device=DEVICE)
    for i in range(BATCH)
]
DST_ROWS = [
    torch.zeros(ROW_LEN, dtype=torch.int32, device=DEVICE) for _ in range(BATCH)
]

SRC_PTRS = torch.tensor([t.data_ptr() for t in SRC_ROWS], dtype=torch.int64, device=DEVICE)
DST_PTRS = torch.tensor([t.data_ptr() for t in DST_ROWS], dtype=torch.int64, device=DEVICE)
SIZES = torch.tensor([n * ELEM_SIZE for n in COPY_ELEMS], dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(src_rows, copy_elems, row_len):
    # Destination starts zeroed; copy the first copy_elems[i] elements of src.
    out = []
    for i, src in enumerate(src_rows):
        dst = torch.zeros(row_len, dtype=torch.int32)
        n = copy_elems[i]
        dst[:n] = src.cpu()[:n]
        out.append(dst)
    return out


def kernel_impl(src_ptrs, dst_ptrs, sizes, dst_rows):
    # dst_rows are mutated in place by the kernel.
    batch_memcpy(src_ptrs, dst_ptrs, sizes)
    return [d.cpu() for d in dst_rows]


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_rows = pytorch_ref(SRC_ROWS, COPY_ELEMS, ROW_LEN)
        ker_rows = kernel_impl(SRC_PTRS, DST_PTRS, SIZES, DST_ROWS)
        total_mism = 0
        for r, k in zip(ref_rows, ker_rows):
            total_mism += int((k != r).sum().item())
        assert total_mism == 0, f"dst mismatch count={total_mism}"
        stats = {
            "batch": BATCH,
            "row_len": ROW_LEN,
            "copy_elems": COPY_ELEMS,
            "dtype": "torch.int32",
            "device": DEVICE,
            "mismatch": total_mism,
            "max_abs_diff": 0,
            "grid": f"({BATCH},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(SRC_ROWS, COPY_ELEMS, ROW_LEN))
        kern_stats = _bench(lambda: kernel_impl(SRC_PTRS, DST_PTRS, SIZES, DST_ROWS))
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
            "Kernel: batch_memcpy_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"batch: {stats['batch']}  row_len: {stats['row_len']}\n")
            lines.append(f"copy_elems: {stats['copy_elems']}\n")
            lines.append(f"dtype: {stats['dtype']}  device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch: {stats['mismatch']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(
                "Note: src/dst pointer arrays built from data_ptr() of module-"
                "scope row tensors; invoked via repo's batch_memcpy launcher.\n"
            )
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
    main()
