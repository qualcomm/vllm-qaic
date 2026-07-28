"""
Standalone QAIC validation for `scaled_mm_kernel`.

Source under test:
vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
  - scaled_mm_kernel  (@triton.jit)
  - triton_scaled_mm  (launcher)

Fallback Triton scaled matmul for compressed-tensors static/dynamic scaled
INT8/FP8 GEMM. Computes:
    C = (A @ B) * scale_a * scale_b + bias
where A is (M, K), B is (K, N), scale_a is per-tensor (1,1) or per-row (M,1),
scale_b is per-tensor (1,1) or per-column (N,1), and bias is optional (N,).
The accumulator is int32 for integer inputs and fp32 for float inputs.

We validate the per-tensor scale path (scale_a, scale_b scalars) with int8
inputs and an fp32 bias. Reference: pure PyTorch
`(A.float() @ B.float()) * scale_a * scale_b + bias`. The int32 dot product is
exact, so compare the fp32 output at rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_scaled_mm_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/compressed_tensors/"
    "triton_scaled_mm.py"
)

M = 64
N = 64
K = 64
DEVICE = "qaic"

_IS_CHILD = os.environ.get("SCALED_MM_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    torch.manual_seed(42)
    A = torch.randint(-100, 100, (M, K), dtype=torch.int8, device=DEVICE)
    B = torch.randint(-100, 100, (K, N), dtype=torch.int8, device=DEVICE)
    # Per-tensor scalar scales.
    SCALE_A = torch.tensor([[0.0123]], dtype=torch.float32, device=DEVICE)
    SCALE_B = torch.tensor([[0.0456]], dtype=torch.float32, device=DEVICE)
    BIAS = torch.randn(N, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(a, b, scale_a, scale_b, bias):
    """Pure PyTorch scaled matmul: (A @ B) * sa * sb + bias."""
    a = a.to(torch.float32).cpu()
    b = b.to(torch.float32).cpu()
    sa = scale_a.to(torch.float32).cpu()
    sb = scale_b.to(torch.float32).cpu()
    out = (a @ b) * sa * sb.t()  # sa: (1,1) or (M,1); sb: (1,1) or (N,1) -> .t()
    out = out + bias.to(torch.float32).cpu()
    return out


def kernel_impl(a, b, scale_a, scale_b, bias):
    c = triton_scaled_mm(
        a, b, scale_a, scale_b, out_dtype=torch.float32, bias=bias
    )
    return c.to(torch.float32)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_c = pytorch_ref(A, B, SCALE_A, SCALE_B, BIAS)
        kern_c = kernel_impl(A, B, SCALE_A, SCALE_B, BIAS)

        ref_c = ref_c.cpu()
        kern_c = kern_c.cpu()

        torch.testing.assert_close(kern_c, ref_c, rtol=1e-3, atol=1e-3)

        diff = (kern_c - ref_c).abs()
        rel = diff / (ref_c.abs() + 1e-6)
        stats = {
            "A_shape": tuple(A.shape),
            "B_shape": tuple(B.shape),
            "C_shape": tuple(kern_c.shape),
            "in_dtype": str(A.dtype),
            "device": str(A.device),
            "scale_mode": "per-tensor (scalar)",
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "max_rel_err": rel.max().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(A, B, SCALE_A, SCALE_B, BIAS))
        kern_stats = _bench(lambda: kernel_impl(A, B, SCALE_A, SCALE_B, BIAS))
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
            "Kernel: scaled_mm_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- A shape: {stats['A_shape']} ({stats['in_dtype']})\n")
            lines.append(f"- B shape: {stats['B_shape']}\n")
            lines.append(f"- scale mode: {stats['scale_mode']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- C shape: {stats['C_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- max_rel_err: {stats['max_rel_err']}\n")
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

    if os.environ.get("SCALED_MM_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, SCALED_MM_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: scaled_mm_kernel\n"
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
