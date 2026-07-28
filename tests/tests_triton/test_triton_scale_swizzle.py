"""
Standalone QAIC validation for `triton_scale_swizzle`.

Source under test:
vllm/model_executor/layers/quantization/qutlass_utils.py
  - triton_scale_swizzle  (@triton.jit)
  - triton_mx_block_rearrange  (launcher)

Rearranges an MX/NVFP4 block-scale tensor (1-byte elements, E8M0) from
row-major layout into the hardware block-scaled "swizzle" layout used by Tmem
(see NVIDIA d-block-scaling-factors-layout docs). This is a pure permutation
of scale BYTES with zero-padding: rows are padded to a multiple of 128 and
cols to a multiple of 4.

Exact index mapping (per 128x4 tile, tile coords (pid_row, pid_col)):
  For a local row r in [0,128) and local col c in [0,4):
    global_row = pid_row*128 + r ; global_col = pid_col*4 + c
    value      = scale[global_row, global_col]   (0 if out of the real bounds)
    dest       = (r % 32) * 16 + (r // 32) * 4 + c            # within-tile
    block_off  = pid_col * (128*4) + pid_row * output_block_stride
    output_flat[block_off + dest] = value
  where output_block_stride = 128 * 4 * (padded_cols // 4).

The reference replicates this mapping byte-for-byte; comparison is EXACT
(integer/byte equality), since no arithmetic is performed on the values.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_triton_scale_swizzle.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/qutlass_utils.py"
)

# Real (unpadded) scale-tensor dims. rows->padded to 128, cols->padded to 4.
ROWS = 100
COLS = 8
DEVICE = "qaic"

_IS_CHILD = os.environ.get("SWIZZLE_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.quantization.qutlass_utils import (
        triton_mx_block_rearrange,
    )

    torch.manual_seed(42)
    # 1-byte scale codes (uint8), row-major contiguous.
    X = torch.randint(0, 256, (ROWS, COLS), dtype=torch.uint8, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _cdiv(a, b):
    return (a + b - 1) // b


def pytorch_ref(x):
    """Replicate the swizzle index mapping exactly, in pure PyTorch."""
    x = x.cpu()
    rows, cols = x.shape
    n_row_blocks = _cdiv(rows, 128)
    n_col_blocks = _cdiv(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    output_block_stride = 128 * 4 * (padded_cols // 4)
    flat = torch.zeros(padded_rows * padded_cols, dtype=torch.uint8)

    for pid_row in range(n_row_blocks):
        for pid_col in range(n_col_blocks):
            block_off = pid_col * (128 * 4) + pid_row * output_block_stride
            for r in range(128):
                gr = pid_row * 128 + r
                for c in range(4):
                    gc = pid_col * 4 + c
                    if gr < rows and gc < cols:
                        val = int(x[gr, gc].item())
                    else:
                        val = 0
                    dest = (r % 32) * 16 + (r // 32) * 4 + c
                    flat[block_off + dest] = val

    return flat.reshape(padded_rows, padded_cols)


def kernel_impl(x):
    out = triton_mx_block_rearrange(x.contiguous())
    return out.view(torch.uint8).cpu()


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

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X)
        kern_out = kernel_impl(X)

        ref_out = ref_out.cpu()
        kern_out = kern_out.cpu()

        exact = bool(torch.equal(ref_out, kern_out))
        assert exact, "swizzle byte layout mismatch"

        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kern_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kern_out.dtype),
            "device": str(X.device),
            "exact_match": exact,
            "num_mismatch": int((ref_out != kern_out).sum().item()),
        }
        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X))
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
        lines = [
            f"{timestamp}\n",
            "Kernel: triton_scale_swizzle\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- x shape: {stats['input_shape']} ({stats['in_dtype']})\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(
                f"- out shape: {stats['output_shape']} ({stats['out_dtype']})\n"
            )
            lines.append(f"- exact byte match: {stats['exact_match']}\n")
            lines.append(f"- num_mismatch: {stats['num_mismatch']}\n")
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


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("SWIZZLE_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, SWIZZLE_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: triton_scale_swizzle\n"
            f"Kernel file: {KERNEL_FILE_PATH}\n"
            f"Device target: QAIC (device='{DEVICE}')\n"
            "Status: FAILURE\n\n"
            "Error:\n"
            f"Child killed by signal (exit {proc.returncode}) during "
            "Triton->Hexagon compile/execution.\n"
            "\n------------------------------------\n\n"
        )
    sys.exit(proc.returncode if proc.returncode >= 0 else 1)


if __name__ == "__main__":
    _run_with_crash_guard()
