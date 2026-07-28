"""
Standalone QAIC validation for `_zero_kv_blocks_kernel`.

Source under test:
vllm/v1/worker/utils.py
  - _zero_kv_blocks_kernel  (launched from KVBlockZeroer.zero_block_ids)

Zeroes selected KV-cache blocks in place. Programs are flattened as
(block_index, seg_index, chunk_index):
    chunks         = PAGE_SIZE_EL // BLOCK_SIZE
    work_per_block = N_SEGS * chunks
    block_index    = pid // work_per_block           (skip if >= n_blocks)
    seg_index      = (pid % work_per_block) // chunks
    chunk_index    = (pid % work_per_block) % chunks
    block_id       = block_ids[block_index]
    seg_addr       = seg_addrs[seg_index]  (absolute byte address, int32*)
    offset         = block_id*PAGE_SIZE_EL + chunk_index*BLOCK_SIZE
    ptr[offset + 0..BLOCK_SIZE) = 0

The real launcher (`KVBlockZeroer.zero_block_ids`) needs allocated KV caches
and attention groups, so we replicate the launch inline with a single segment
pointing at one flat int32 buffer (rule 2). We pre-fill the buffer with a
nonzero pattern, then zero a chosen set of block ids.

Config: PAGE_SIZE_EL=8, BLOCK_SIZE=4 (2 chunks/block), N_SEGS=1, 16 total
blocks, zeroing block ids [2, 5, 9]. Integer buffer -> EXACT integer equality.
Reference: pure-PyTorch clone + zero of the same block rows.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_zero_kv_blocks_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.utils import _zero_kv_blocks_kernel  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
PAGE_SIZE_EL = 8
BLOCK_SIZE = 4
N_SEGS = 1
TOTAL_BLOCKS = 16
BLOCK_IDS_LIST = [2, 5, 9]

# One flat int32 buffer holding TOTAL_BLOCKS blocks of PAGE_SIZE_EL each.
BUFFER_BASE = torch.arange(
    1, TOTAL_BLOCKS * PAGE_SIZE_EL + 1, dtype=torch.int32, device=DEVICE
)


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


def pytorch_ref():
    """Pure PyTorch: clone buffer and zero the selected block rows."""
    buf = BUFFER_BASE.clone().cpu()
    for block_id in BLOCK_IDS_LIST:
        start = block_id * PAGE_SIZE_EL
        buf[start : start + PAGE_SIZE_EL] = 0
    return buf


def kernel_impl():
    """Kernel launch only, replicating KVBlockZeroer.zero_block_ids."""
    buf = BUFFER_BASE.clone()
    seg_addrs = torch.tensor([buf.data_ptr()], dtype=torch.uint64, device=DEVICE)
    block_ids = torch.tensor(BLOCK_IDS_LIST, dtype=torch.int64, device=DEVICE)
    n_blocks = len(BLOCK_IDS_LIST)
    grid = (n_blocks * N_SEGS * (PAGE_SIZE_EL // BLOCK_SIZE),)
    _zero_kv_blocks_kernel[grid](
        seg_addrs,
        block_ids,
        n_blocks,
        N_SEGS=N_SEGS,
        PAGE_SIZE_EL=PAGE_SIZE_EL,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return buf


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), "buffer mismatch after zeroing"

        stats = {
            "input_shape": tuple(BUFFER_BASE.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(BUFFER_BASE.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(BUFFER_BASE.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "zeroed_block_ids": BLOCK_IDS_LIST,
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: _zero_kv_blocks_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape (buffer): {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- zeroed block ids: {stats['zeroed_block_ids']}\n",
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
