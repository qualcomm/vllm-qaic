"""
Standalone QAIC validation for `solve_tril_16x16_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/solve_tril.py
  - solve_tril_16x16_kernel  (Inverts a strictly-lower-triangular 16x16 block
    matrix (I+A)^-1 via a sequential recurrence.)

Launcher `solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype)`
dispatches to this exact kernel when A.shape[-1] == 16.

A is [B, T, H, BT] and must be strictly lower triangular within each BTxBT
block (A.triu() == 0, including the diagonal). The kernel computes
Ai = (I + A)^-1 for each BTxBT block.

Reference: for a single chunk (T == BT), the block is A[0, :, 0, :] viewed
as a BTxBT matrix where A[i, j] is token i's causal weight on token j.
M = I + A is unit lower-triangular; (I + A)^-1 is exactly
torch.linalg.solve_triangular(M, I, upper=False, unitriangular=True).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_solve_tril_16x16_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/solve_tril.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# solve_tril_16x16_kernel is wrapped in @triton.autotune. Triton's generic
# autotuner benchmarking path (triton.testing.do_bench) calls
# Event.elapsed_time() across two independently-created torch.qaic Events,
# which raises "expected other to be a torch.Event object" on this QAIC
# backend/torch_qaic version. This is a benchmarking-harness incompatibility,
# unrelated to kernel correctness, so we shim do_bench to just execute the
# candidate config once and report a constant timing. Autotune then always
# selects the (only, since key=["BT"]-scoped) fastest-reported config, and
# the kernel itself still executes for real on qaic hardware.
def _qaic_safe_do_bench(fn, *args, **kwargs):
    fn()
    quantiles = kwargs.get("quantiles")
    if quantiles is not None:
        return [1.0] * len(quantiles) if len(quantiles) > 1 else 1.0
    return 1.0


_triton_testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril  # noqa: E402

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
B, T, H, BT = 1, 16, 1, 16

torch.manual_seed(42)
# A is logically [B, H, T, BT] per (b, h): rows = token position within the
# chunk, cols = position within the chunk's BTxBT block. torch.tril operates
# on the LAST TWO dims, so build with shape (B, H, T, BT), apply tril there,
# then permute back to the storage layout (B, T, H, BT).
_raw = torch.randn(B, H, T, BT, dtype=torch.float32, device=DEVICE)
# Enforce strictly lower triangular (diagonal must also be zero) within the
# BTxBT block, per the docstring precondition A.triu() == 0.
_tril = torch.tril(_raw, diagonal=-1)
INPUT = _tril.permute(0, 2, 1, 3).contiguous()
WEIGHT = None
BIAS = None


def pytorch_ref(A):
    """Pure PyTorch reference implementation for `solve_tril_16x16_kernel`.

    Computes Ai = (I + A)^-1 for the single BTxBT block via exact
    unit-lower-triangular matrix inversion.
    """
    A_cpu = A.detach().cpu()
    b, t, h, bt = A_cpu.shape
    out = torch.zeros_like(A_cpu)
    eye = torch.eye(bt, dtype=A_cpu.dtype)
    for ib in range(b):
        for ih in range(h):
            # Single chunk: T == BT, block is A[ib, :, ih, :].
            block = A_cpu[ib, :, ih, :]
            M = eye + block
            Ai_block = torch.linalg.solve_triangular(
                M, eye, upper=False, unitriangular=True
            )
            out[ib, :, ih, :] = Ai_block
    return out


def kernel_impl(A):
    return solve_tril(A, cu_seqlens=None, chunk_indices=None, output_dtype=torch.float32)


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


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(INPUT)
        kernel_out = kernel_impl(INPUT)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-8)).mean().item()
        stats = {
            "input_shape": tuple(INPUT.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(INPUT.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(INPUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "relative_error": rel_err,
        }

        pt_stats = _bench(lambda: pytorch_ref(INPUT))
        kern_stats = _bench(lambda: kernel_impl(INPUT))
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
            "Kernel: solve_tril_16x16_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- A shape: {stats['input_shape']} (B={B}, T={T}, H={H}, BT={BT})\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- Ai shape: {stats['output_shape']}\n")
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- relative_error: {stats['relative_error']}\n")
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
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
