"""
Standalone QAIC validation for `_apply_write_kernel`.

Source under test:
vllm/v1/worker/gpu/buffer_utils.py
  - _apply_write_kernel  (launched from StagedWriteTensor.apply_write)

Commits a batch of staged (row_index, start, content) writes into a persistent
2D GPU buffer. For staged write `pid`:
  - row_idx  = write_indices[pid]
  - start    = write_starts[pid]
  - cu_start = write_cu_lens[pid-1] if pid>0 else 0
  - cu_end   = write_cu_lens[pid]
  - content_len = cu_end - cu_start
It writes write_contents[cu_start : cu_end] into
buffer[row_idx, start : start + content_len]. This is how deferred
block-table / state row updates are flushed to the GPU in one kernel.

LAUNCH: The public launcher (StagedWriteTensor.apply_write) routes contents
through UVA pinned buffers; we instead replicate the exact launch site from the
source -- `_apply_write_kernel[(n,)](gpu, gpu.stride(0), indices, starts,
contents, cu_lens, BLOCK_SIZE=1024)` -- with plain device tensors so the
validation targets the kernel itself.

Integer-exact validation of the mutated buffer.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_apply_write_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/buffer_utils.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.gpu.buffer_utils import _apply_write_kernel

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
NUM_ROWS = 5
ROW_WIDTH = 16
BLOCK_SIZE = 1024

# Persistent buffer (starts zeroed, like StagedWriteTensor.gpu).
BUFFER = torch.zeros(NUM_ROWS, ROW_WIDTH, dtype=torch.int32, device=DEVICE)

# Staged writes: (row_index, start, content-list).
STAGED = [
    (0, 0, [11, 12, 13]),
    (2, 4, [21, 22, 23, 24, 25]),
    (4, 1, [31, 32]),
    (0, 8, [41, 42, 43, 44]),   # second write into row 0 (different columns)
]
WRITE_INDICES = torch.tensor([s[0] for s in STAGED], dtype=torch.int32, device=DEVICE)
WRITE_STARTS = torch.tensor([s[1] for s in STAGED], dtype=torch.int32, device=DEVICE)
_contents = []
_cu = []
for _, _, c in STAGED:
    _contents.extend(c)
    _cu.append(len(_contents))
WRITE_CONTENTS = torch.tensor(_contents, dtype=torch.int32, device=DEVICE)
WRITE_CU_LENS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
N = len(STAGED)


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


def pytorch_ref(buffer, staged):
    out = buffer.cpu().clone()
    for row_idx, start, content in staged:
        c = torch.tensor(content, dtype=out.dtype)
        out[row_idx, start:start + len(content)] = c
    return out


def kernel_impl(buffer, write_indices, write_starts, write_contents, write_cu_lens, n):
    buffer = buffer.clone()
    _apply_write_kernel[(n,)](
        buffer,
        buffer.stride(0),
        write_indices,
        write_starts,
        write_contents,
        write_cu_lens,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return buffer


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(BUFFER, STAGED)
        ker = kernel_impl(
            BUFFER, WRITE_INDICES, WRITE_STARTS, WRITE_CONTENTS, WRITE_CU_LENS, N
        ).cpu()
        mism = int((ker != ref).sum().item())
        assert mism == 0, f"buffer mismatch count={mism}"
        stats = {
            "buffer_shape": tuple(BUFFER.shape),
            "num_writes": N,
            "dtype": str(BUFFER.dtype),
            "device": str(BUFFER.device),
            "mismatch": mism,
            "max_abs_diff": 0,
            "grid": f"({N},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(BUFFER, STAGED))
        kern_stats = _bench(lambda: kernel_impl(
            BUFFER, WRITE_INDICES, WRITE_STARTS, WRITE_CONTENTS, WRITE_CU_LENS, N
        ))
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
            "Kernel: _apply_write_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"buffer shape: {stats['buffer_shape']}\n")
            lines.append(f"num_writes: {stats['num_writes']}\n")
            lines.append(f"dtype: {stats['dtype']}  device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch: {stats['mismatch']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
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
