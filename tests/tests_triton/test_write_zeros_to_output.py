"""
Standalone QAIC validation for the `write_zeros_to_output` device helper.

Source under test:
vllm/model_executor/layers/fused_moe/fused_moe.py
  - write_zeros_to_output  (@triton.jit device function)

`write_zeros_to_output` is a Triton *device* function (called from inside the
fused MoE GEMM kernels). It materializes a zero-filled [BLOCK_SIZE_M,
BLOCK_SIZE_N] accumulator tile and stores it back into the output matrix C for
the tokens of a block whose assigned expert is not present on the current
expert-parallel rank (expert id == -1). Because it is not itself a launchable
kernel, we wrap it in a tiny `@triton.jit` kernel that provides the block
metadata (offs_token / token_mask) and stores the zeros.

Reference: a zero tile of shape (BLOCK_SIZE_M, BLOCK_SIZE_N).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_write_zeros_to_output.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/fused_moe.py"

# Global inputs
BLOCK_SIZE_M = 8
BLOCK_SIZE_N = 16
DEVICE = "qaic"

_IS_CHILD = os.environ.get("WZ_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import write_zeros_to_output
    from vllm.triton_utils import tl, triton

    torch.manual_seed(42)
    # Start from ones so a correct kernel must overwrite every element with 0.
    C = torch.ones(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=torch.float32, device=DEVICE)

    @triton.jit
    def _write_zeros_wrapper(
        c_ptr,
        stride_cm,
        stride_cn,
        N,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        compute_type: tl.constexpr,
    ):
        # One block, pid_n == 0. All rows are valid tokens.
        offs_token = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        token_mask = offs_token < BLOCK_SIZE_M
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            0,  # pid_n
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )


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


def pytorch_ref(c):
    """Pure PyTorch reference: the output tile is fully zeroed."""
    return torch.zeros_like(c.cpu())


def kernel_impl(c):
    c = c.clone()
    _write_zeros_wrapper[(1,)](
        c,
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        tl.float32,
    )
    return c


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(C)
        kernel_out = kernel_impl(C)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(C.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(C.dtype),
            "device": str(C.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "grid": ("(1,)", f"BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}"),
        }

        pt_stats = _bench(lambda: pytorch_ref(C))
        kern_stats = _bench(lambda: kernel_impl(C))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: write_zeros_to_output\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- C shape: {stats['input_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid'][0]}\n")
            lines.append(f"- block: {stats['grid'][1]}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n"
                )
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("WZ_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, WZ_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: write_zeros_to_output\n"
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
