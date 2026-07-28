"""
Standalone QAIC validation for `_quantize_pad_fp8_kernel`.

Source under test:
vllm/kernels/triton/qkv_padded_fp8_quant.py
  - _quantize_pad_fp8_kernel  (@triton.jit)
  - quantize_fp8_pad_head_dim_triton  (launcher)

Stride-aware FP8 quantization for ViT attention. Reads a (possibly
non-contiguous) 3D (S, H, D) tensor directly via its 3D strides, quantizes to
FP8 using a single per-tensor scalar `scale`:
    x_q = clamp(x / scale, fp8_min, fp8_max)  cast to fp8
and writes a fresh CONTIGUOUS output of shape (S, H, padded_D) where padded_D
is D rounded up to a multiple of 16 (for cuDNN). Columns in [D, padded_D) are
never written by the real-data mask, so they stay zero.

To exercise the stride path we build an interleaved QKV buffer (S, 3*H*D) and
slice out the Q view (non-contiguous in the head/dim axes), with D=40 so
padding to 48 is required.

Comparison choice: dequantize kernel fp8 codes with the same scalar scale and
compare against the reference dequant over the REAL columns [0, D) at
rtol/atol=1e-2 (fp8 low precision); the padding columns [D, padded_D) are
checked to be EXACTLY zero.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_quantize_pad_fp8_kernel.txt")
KERNEL_FILE_PATH = "vllm/kernels/triton/qkv_padded_fp8_quant.py"

S = 8       # tokens
H = 4       # heads
D = 40      # head_dim (not a multiple of 16 -> padded to 48)
PADDED_D = 48
SCALE_VAL = 0.05
DEVICE = "qaic"

_IS_CHILD = os.environ.get("QPAD_FP8_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.kernels.triton.qkv_padded_fp8_quant import (
        quantize_fp8_pad_head_dim_triton,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm.platforms import current_platform

    torch.manual_seed(42)
    FP8_DTYPE = current_platform.fp8_dtype()
    FP8_MIN, FP8_MAX = get_fp8_min_max()

    # Interleaved QKV buffer: (S, 3, H, D). Slicing [:, 0] gives a
    # non-contiguous (S, H, D) Q view exercising the 3D stride path.
    QKV = torch.randn(S, 3, H, D, dtype=torch.float32, device=DEVICE)
    Q_VIEW = QKV[:, 0]  # (S, H, D), non-contiguous
    SCALE = torch.tensor(SCALE_VAL, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(q_view, scale):
    """Pure PyTorch stride-aware FP8 quant with head_dim padding.

    Returns fp8 codes cast to float32, shape (S, H, PADDED_D), with padding
    columns zero.
    """
    q = q_view.to(torch.float32).cpu().contiguous()  # (S, H, D)
    scale_v = float(scale.item())
    out = torch.zeros(S, H, PADDED_D, dtype=torch.float32)
    q_scaled = torch.clamp(q / scale_v, FP8_MIN, FP8_MAX)
    q_fp8 = q_scaled.to(FP8_DTYPE).to(torch.float32)  # (S, H, D)
    out[:, :, :D] = q_fp8
    return out


def kernel_impl(q_view, scale):
    out = quantize_fp8_pad_head_dim_triton(q_view, scale, skip_scale=False)
    # out shape (S, H, PADDED_D) fp8; view as float for comparison.
    return out.to(torch.float32)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(Q_VIEW, SCALE)
        kern_out = kernel_impl(Q_VIEW, SCALE)

        ref_out = ref_out.cpu()
        kern_out = kern_out.cpu()

        # Dequantize real columns with the scalar scale and compare.
        ref_deq = ref_out[:, :, :D] * SCALE_VAL
        kern_deq = kern_out[:, :, :D] * SCALE_VAL
        torch.testing.assert_close(kern_deq, ref_deq, rtol=1e-2, atol=1e-2)

        # Padding must be exactly zero.
        pad_zero = bool((kern_out[:, :, D:] == 0).all())
        assert pad_zero, "padding region is not zero"

        diff = (kern_deq - ref_deq).abs()
        stats = {
            "qkv_shape": tuple(QKV.shape),
            "q_view_shape": tuple(Q_VIEW.shape),
            "q_view_contig": bool(Q_VIEW.is_contiguous()),
            "output_shape": tuple(kern_out.shape),
            "fp8_dtype": str(FP8_DTYPE),
            "device": str(QKV.device),
            "D": D,
            "padded_D": PADDED_D,
            "scale": SCALE_VAL,
            "pad_zero": pad_zero,
            "max_abs_diff_deq": diff.max().item(),
            "mean_abs_diff_deq": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(Q_VIEW, SCALE))
        kern_stats = _bench(lambda: kernel_impl(Q_VIEW, SCALE))
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
            "Kernel: _quantize_pad_fp8_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- qkv buffer shape: {stats['qkv_shape']}\n")
            lines.append(
                f"- q view shape: {stats['q_view_shape']} "
                f"(contiguous={stats['q_view_contig']})\n"
            )
            lines.append(f"- D={stats['D']}, padded_D={stats['padded_D']}\n")
            lines.append(f"- scale: {stats['scale']}\n")
            lines.append(f"- fp8_dtype: {stats['fp8_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- padding all-zero: {stats['pad_zero']}\n")
            lines.append(f"- max_abs_diff (dequant): {stats['max_abs_diff_deq']}\n")
            lines.append(f"- mean_abs_diff (dequant): {stats['mean_abs_diff_deq']}\n")
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

    if os.environ.get("QPAD_FP8_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, QPAD_FP8_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: _quantize_pad_fp8_kernel\n"
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
