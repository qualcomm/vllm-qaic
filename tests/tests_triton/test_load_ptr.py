"""
Standalone QAIC validation for `_load_ptr`.

Source under test:
vllm/v1/worker/gpu/block_table.py
  - _load_ptr  (a @triton.jit DEVICE HELPER, not launched directly)

`_load_ptr(ptr_to_ptr, elem_dtype)` performs pointer plumbing: it loads a raw
address stored in an int64/uint64 array, casts it to a typed Triton pointer
(`tl.pointer_type(elem_dtype)`), and applies `tl.multiple_of(ptr, 16)` alignment
so the compiler knows the dereference is 16-byte aligned. It is used inside
`_gather_block_tables_kernel` / `_compute_slot_mappings_kernel` to dereference
arrays-of-tensor-pointers (each block table is a separate tensor, whose
data_ptr() is packed into a 1D address array).

APPROACH: Because `_load_ptr` is a device helper (never launched on its own),
we write a tiny standalone `@triton.jit` wrapper kernel `_load_ptr_wrapper`
that faithfully reproduces the parent kernels' usage pattern:
  1. Build an int64 array `ptrs` of the data_ptr() addresses of several small
     int32 sub-tensors (exactly like `BlockTables._make_ptr_tensor`).
  2. For grid program `pid`, call `_load_ptr(ptrs + pid, tl.int32)` to load and
     cast the pid-th address to an int32*.
  3. Dereference element `col` of that typed pointer and store the value.
This exercises _load_ptr end-to-end: load address -> cast -> aligned typed
load. pytorch_ref: the value each pointer should point to = sub_tensors[pid][col].

Integer-exact validation of the dereferenced values.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_load_ptr.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/block_table.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.block_table import _load_ptr

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
NUM_PTRS = 4
ROW_LEN = 8
# Column within each sub-tensor to dereference and validate.
COL = 3
# Distinct sub-tensors; their data_ptr() addresses feed the pointer array.
# Keep them alive at module scope so the addresses stay valid.
SUB_TENSORS = [
    torch.arange(i * 100, i * 100 + ROW_LEN, dtype=torch.int32, device=DEVICE)
    for i in range(NUM_PTRS)
]
# uint64 to cover all possible addresses (mirrors _make_ptr_tensor).
PTR_ARRAY = torch.tensor(
    [t.data_ptr() for t in SUB_TENSORS], dtype=torch.uint64, device=DEVICE
)


@triton.jit
def _load_ptr_wrapper(ptr_array, out_ptr, col, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # Load+cast the pid-th raw address into a typed int32 pointer.
    typed_ptr = _load_ptr(ptr_array + pid, tl.int32)
    # Dereference element `col` of that tensor.
    val = tl.load(typed_ptr + col)
    tl.store(out_ptr + pid, val)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(sub_tensors, col):
    return torch.tensor(
        [int(t.cpu()[col].item()) for t in sub_tensors], dtype=torch.int32
    )


def kernel_impl(ptr_array, col, n):
    out = torch.zeros(n, dtype=torch.int32, device=DEVICE)
    _load_ptr_wrapper[(n,)](ptr_array, out, col, BLOCK=1)
    return out


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(SUB_TENSORS, COL)
        ker = kernel_impl(PTR_ARRAY, COL, NUM_PTRS).cpu()
        mism = int((ker != ref).sum().item())
        assert mism == 0, f"dereferenced values mismatch count={mism}"
        stats = {
            "num_ptrs": NUM_PTRS,
            "col": COL,
            "dtype": str(ker.dtype),
            "device": DEVICE,
            "mismatch": mism,
            "max_abs_diff": 0,
            "grid": f"({NUM_PTRS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(SUB_TENSORS, COL))
        kern_stats = _bench(lambda: kernel_impl(PTR_ARRAY, COL, NUM_PTRS))
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
            "Kernel: _load_ptr (device helper, via _load_ptr_wrapper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"num_ptrs: {stats['num_ptrs']}  col: {stats['col']}\n")
            lines.append(f"dtype: {stats['dtype']}  device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch: {stats['mismatch']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(
                "Note: _load_ptr is a pointer-plumbing device helper; validated "
                "via a faithful wrapper that builds an int64 address array of "
                "sub-tensors and dereferences the loaded typed pointer.\n"
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
