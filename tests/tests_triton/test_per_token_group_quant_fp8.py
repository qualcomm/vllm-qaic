"""
Standalone QAIC validation for `_per_token_group_quant_fp8`.

Source under test:
vllm/model_executor/layers/quantization/utils/fp8_utils.py
  - _per_token_group_quant_fp8  (@triton.jit)
  - per_token_group_quant_fp8   (launcher; Triton fallback path)

The kernel quantizes a 2D tensor to FP8 with a per-token-group (row-major)
absmax scale. For each contiguous group of `group_size` elements along the
last dim it computes:
    absmax = max(|y|)
    y_s    = max(absmax, eps) * (1 / fp8_max)          (non-UE8M0 path)
    y_q    = clamp(y / y_s, fp8_min, fp8_max)  cast to fp8
Outputs: y_q (M, N) fp8 codes and y_s (M, N/group_size) fp32 scales in
row-major layout.

We validate the NON-UE8M0 path (use_ue8m0=False). Comparison choice: fp8 is a
low-precision format, so we (a) dequantize the kernel codes with the kernel
scales and compare against the reference dequantized-from-reference values at
rtol/atol=1e-2, and (b) compare the fp32 scale tensors directly at 1e-3.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_per_token_group_quant_fp8.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
)

# Global inputs (tiny). x: (TOKENS, HIDDEN), grouped every GROUP_SIZE along dim -1.
TOKENS = 8
HIDDEN = 64
GROUP_SIZE = 32
EPS = 1e-10
DEVICE = "qaic"

_IS_CHILD = os.environ.get("PTGQ_FP8_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm.platforms import current_platform

    torch.manual_seed(42)
    X = torch.randn(TOKENS, HIDDEN, dtype=torch.float32, device=DEVICE)
    FP8_DTYPE = current_platform.fp8_dtype()
    FP8_MIN, FP8_MAX = get_fp8_min_max()


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


def pytorch_ref(x):
    """Pure PyTorch per-token-group absmax FP8 quantization (non-UE8M0).

    Returns (y_q_float, y_s):
      * y_q_float: fp8 codes cast to float32, shape (TOKENS, HIDDEN)
      * y_s:       fp32 per-group scales, shape (TOKENS, HIDDEN/GROUP_SIZE)
    """
    x = x.cpu()
    M, N = x.shape
    G = N // GROUP_SIZE
    y_q = torch.zeros(M, N, dtype=torch.float32)
    y_s = torch.zeros(M, G, dtype=torch.float32)
    eps_t = torch.tensor(EPS, dtype=torch.float32)
    for m in range(M):
        for g in range(G):
            grp = x[m, g * GROUP_SIZE : (g + 1) * GROUP_SIZE]
            absmax = torch.maximum(grp.abs().max(), eps_t)
            scale = absmax * (1.0 / FP8_MAX)
            q = torch.clamp(grp / scale, FP8_MIN, FP8_MAX)
            y_q[m, g * GROUP_SIZE : (g + 1) * GROUP_SIZE] = (
                q.to(FP8_DTYPE).to(torch.float32)
            )
            y_s[m, g] = scale
    return y_q, y_s


def kernel_impl(x):
    x_q, x_s = per_token_group_quant_fp8(
        x.contiguous(),
        GROUP_SIZE,
        eps=EPS,
        column_major_scales=False,
        use_ue8m0=False,
    )
    return x_q.to(torch.float32), x_s.to(torch.float32)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_q, ref_s = pytorch_ref(X)
        kern_q, kern_s = kernel_impl(X)

        ref_q = ref_q.cpu()
        ref_s = ref_s.cpu()
        kern_q = kern_q.cpu()
        kern_s = kern_s.cpu()

        # Dequantize both using their own stored scales, then compare.
        G = HIDDEN // GROUP_SIZE
        ref_deq = ref_q * ref_s.repeat_interleave(GROUP_SIZE, dim=1)
        kern_deq = kern_q * kern_s.repeat_interleave(GROUP_SIZE, dim=1)

        torch.testing.assert_close(kern_deq, ref_deq, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(kern_s, ref_s, rtol=1e-3, atol=1e-3)

        diff_v = (kern_deq - ref_deq).abs()
        diff_s = (kern_s - ref_s).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "yq_shape": tuple(kern_q.shape),
            "ys_shape": tuple(kern_s.shape),
            "dtype": str(X.dtype),
            "fp8_dtype": str(FP8_DTYPE),
            "device": str(X.device),
            "max_abs_diff_deq": diff_v.max().item(),
            "mean_abs_diff_deq": diff_v.mean().item(),
            "max_abs_diff_scale": diff_s.max().item(),
            "groups": G,
        }
        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X))
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
            "Kernel: _per_token_group_quant_fp8\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- x shape: {stats['input_shape']}\n")
            lines.append(
                f"- tokens={TOKENS}, hidden={HIDDEN}, group_size={GROUP_SIZE}\n"
            )
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- fp8_dtype: {stats['fp8_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- y_q shape: {stats['yq_shape']}\n")
            lines.append(f"- y_s shape: {stats['ys_shape']} (row-major)\n")
            lines.append(f"- max_abs_diff (dequant): {stats['max_abs_diff_deq']}\n")
            lines.append(f"- mean_abs_diff (dequant): {stats['mean_abs_diff_deq']}\n")
            lines.append(f"- max_abs_diff (scale): {stats['max_abs_diff_scale']}\n")
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


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("PTGQ_FP8_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, PTGQ_FP8_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: _per_token_group_quant_fp8\n"
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
