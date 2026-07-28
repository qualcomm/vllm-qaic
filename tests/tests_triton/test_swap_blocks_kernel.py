"""
Standalone QAIC validation for `_swap_blocks_kernel`.

Source under test:
vllm/v1/kv_offload/cpu/swap_blocks_triton.py
  - _swap_blocks_kernel  (@triton.jit)
  - swap_blocks_batch    (public launcher)

`_swap_blocks_kernel` is a batched raw-address memcpy: given three parallel
arrays `src_addrs`, `dst_addrs`, `sizes` (one entry per "job"/block), each
program strides over the jobs and, for its job, reinterprets the stored int64
values as int64 device pointers and copies `sizes[job] // 8` 64-bit words from
src to dst in BYTES_PER_CHUNK-sized chunks. It is a pure data movement kernel
(no arithmetic).

We do NOT use the public `swap_blocks_batch` launcher because it hardcodes
`.to("cuda", ...)` and falls back to a C++ op for small batches — neither is
appropriate on QAIC. Instead we import the real `@triton.jit` kernel unchanged
and drive it with a minimal launch wrapper that builds address tensors from
`data_ptr()` of per-block int64 buffers on the QAIC device (this mirrors the
exact args the source launcher passes). The kernel body under test is
unmodified.

Config: N_JOBS blocks of int64 data (8-byte aligned so sizes are word
multiples), BYTES_PER_CHUNK = 64. Pure copy => EXACT integer equality of the
destination vs source.
Reference: pure PyTorch tensor copy of each block.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_swap_blocks_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/kv_offload/cpu/swap_blocks_triton.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (  # noqa: E402
    NUM_SMS,
    _swap_blocks_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
N_JOBS = 8  # number of blocks to copy
BLOCK_WORDS = 32  # int64 words per block (=> 256 bytes each)
BYTES_PER_CHUNK = 64  # 8 words per chunk
DTYPE = torch.int64

# Per-block source data (distinct values) and zero-initialized destinations.
# Each block is its own contiguous tensor so data_ptr() is a stable base addr.
SRC_BLOCKS = [
    torch.randint(
        -(2**40), 2**40, (BLOCK_WORDS,), dtype=DTYPE, device=DEVICE
    )
    for _ in range(N_JOBS)
]


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
    """Pure PyTorch per-block copy."""
    return [blk.cpu().clone() for blk in SRC_BLOCKS]


def kernel_impl():
    """Launch only: allocate dst blocks, build address arrays, run the kernel."""
    dst_blocks = [torch.zeros_like(b) for b in SRC_BLOCKS]
    src_addrs = torch.tensor(
        [b.data_ptr() for b in SRC_BLOCKS], dtype=torch.int64, device=DEVICE
    )
    dst_addrs = torch.tensor(
        [b.data_ptr() for b in dst_blocks], dtype=torch.int64, device=DEVICE
    )
    sizes = torch.tensor(
        [b.numel() * b.element_size() for b in SRC_BLOCKS],
        dtype=torch.int64,
        device=DEVICE,
    )
    n = src_addrs.numel()
    grid = (min(NUM_SMS, n),)
    _swap_blocks_kernel[grid](
        src_addrs,
        dst_addrs,
        sizes,
        n,
        BYTES_PER_CHUNK=BYTES_PER_CHUNK,
    )
    return dst_blocks


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_blocks = pytorch_ref()
        ker_blocks = kernel_impl()

        ker_cpu = [b.cpu() for b in ker_blocks]
        max_abs_diff = 0
        for r, k in zip(ref_blocks, ker_cpu):
            assert torch.equal(r, k), "block mismatch in swap_blocks copy"
            max_abs_diff = max(max_abs_diff, int((r - k).abs().max().item()))

        stats = {
            "input_shape": (N_JOBS, BLOCK_WORDS),
            "output_shape": (N_JOBS, BLOCK_WORDS),
            "in_dtype": str(DTYPE),
            "out_dtype": str(ker_blocks[0].dtype),
            "device": str(SRC_BLOCKS[0].device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": 0.0,
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: _swap_blocks_kernel (raw-address batched memcpy)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
