"""
Standalone QAIC validation for `_per_token_quant_int8`.

Source under test:
vllm/model_executor/layers/quantization/utils/int8_utils.py
  - _per_token_quant_int8  (@triton.jit)
  - per_token_quant_int8   (launcher)

Per-token (per-row) absmax INT8 quantization. For each row:
    absmax = max(|x|, 1e-10)
    scale  = absmax / 127                 (stored, returned as reciprocal form)
    x_q    = round_int8(x * (127 / absmax))
Outputs: x_q (M, N) int8 and scales (M, 1) fp32 (= absmax / 127).

Comparison choice: dequantize kernel codes with kernel scales and compare to
the reference dequant at rtol/atol=1e-2 (int8 low precision), and compare the
fp32 scales directly at 1e-3. Inputs avoid exact .5 ties so the round-half
convention does not affect the int8 codes.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_per_token_quant_int8.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/int8_utils.py"
)

TOKENS = 8
HIDDEN = 64
DEVICE = "qaic"

_IS_CHILD = os.environ.get("PTQ_INT8_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_quant_int8,
    )

    torch.manual_seed(42)
    X = torch.randn(TOKENS, HIDDEN, dtype=torch.float32, device=DEVICE)


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
    """Pure PyTorch per-row absmax INT8 quantization.

    Returns (x_q_int8, scale) with scale = absmax / 127, shape (M, 1).
    """
    x = x.cpu()
    absmax = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-10)
    scale = absmax / 127.0
    x_q = torch.round(x * (127.0 / absmax)).clamp(-128, 127).to(torch.int8)
    return x_q, scale


def kernel_impl(x):
    x_q, scale = per_token_quant_int8(x.contiguous())
    return x_q, scale.to(torch.float32)


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

        ref_deq = ref_q.to(torch.float32) * ref_s
        kern_deq = kern_q.to(torch.float32) * kern_s

        torch.testing.assert_close(kern_deq, ref_deq, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(kern_s, ref_s, rtol=1e-3, atol=1e-3)

        diff_v = (kern_deq - ref_deq).abs()
        diff_s = (kern_s - ref_s).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "xq_shape": tuple(kern_q.shape),
            "scale_shape": tuple(kern_s.shape),
            "in_dtype": str(X.dtype),
            "xq_dtype": str(kern_q.dtype),
            "device": str(X.device),
            "max_abs_diff_deq": diff_v.max().item(),
            "mean_abs_diff_deq": diff_v.mean().item(),
            "max_abs_diff_scale": diff_s.max().item(),
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
            "Kernel: _per_token_quant_int8\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- x shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- x_q shape: {stats['xq_shape']} ({stats['xq_dtype']})\n")
            lines.append(f"- scale shape: {stats['scale_shape']}\n")
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

    if os.environ.get("PTQ_INT8_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, PTQ_INT8_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: _per_token_quant_int8\n"
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
