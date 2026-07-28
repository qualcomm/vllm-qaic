"""
Standalone QAIC validation for `_silu_mul_per_token_group_quant_fp8_colmajor`.

Source under test:
vllm/model_executor/layers/quantization/utils/fp8_utils.py
  - _silu_mul_per_token_group_quant_fp8_colmajor  (@triton.jit)
  - silu_mul_per_token_group_quant_fp8_colmajor   (launcher)

Fused SiLU-and-mul + per-token-group block-FP8 quantization (group size 128).
Input y is [M, N]; the first half is the SiLU gate, the second half is the
multiplicand:
    N_2   = N // 2
    act   = silu(y[:, :N_2])            # x * sigmoid(x), fp32
    prod  = act * y[:, N_2:]            # [M, N_2]
Then, per group of GROUP_SIZE(=128) columns:
    absmax = max(|prod_group|, eps)
    scale  = absmax / fp8_max           # (use_ue8m0=False path)
    y_q    = clamp(prod_group / scale, fp8_min, fp8_max).to(fp8)
Scales are written COLUMN-MAJOR (returned shape (M, N_2//GROUP), stride(0)==1).

This is the HAS_CLAMP=False path (clamp_limit=None). We validate the
dequantized output (y_q * scale) against a pure-PyTorch reference (fp8 rounding
tolerance) and the scale values, and confirm the scale tensor is column-major.

FP8 note: the kernel casts to the platform fp8 dtype in-kernel. If in-kernel
fp8 casts are unsupported on the QAIC backend the kernel_impl call will surface
that; the reference and comparison remain faithful to the source semantics.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_silu_mul_per_token_group_quant_fp8_colmajor.txt"
)
KERNEL_FILE_PATH = "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402,E501
    silu_mul_per_token_group_quant_fp8_colmajor,
)
from vllm.platforms import current_platform  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
# Constraints from the launcher: M % 128 == 0, N % 256 == 0, M % 8 == 0.
GROUP_SIZE = 128
M = 128
N = 256          # N_2 = 128 -> exactly one group per row
N_2 = N // 2
EPS = 1e-10
Y = torch.randn(M, N, dtype=torch.float32, device=DEVICE)

FP8_DTYPE = current_platform.fp8_dtype()
_FINFO = torch.finfo(FP8_DTYPE)
FP8_MIN = -224.0 if current_platform.is_fp8_fnuz() else _FINFO.min
FP8_MAX = 224.0 if current_platform.is_fp8_fnuz() else _FINFO.max


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


def pytorch_ref(y):
    """Pure PyTorch silu-and-mul + per-group absmax FP8 quant (non-UE8M0).

    Returns (y_q float [M, N_2], y_s float [M, G]).
    """
    y = y.cpu().to(torch.float32)
    act = y[:, :N_2]
    mul = y[:, N_2:]
    silu = act * torch.sigmoid(act)
    prod = silu * mul  # [M, N_2]

    G = N_2 // GROUP_SIZE
    y_q = torch.zeros(M, N_2, dtype=torch.float32)
    y_s = torch.zeros(M, G, dtype=torch.float32)
    eps_t = torch.tensor(EPS, dtype=torch.float32)
    for m in range(M):
        for g in range(G):
            grp = prod[m, g * GROUP_SIZE:(g + 1) * GROUP_SIZE]
            absmax = torch.maximum(grp.abs().max(), eps_t)
            scale = absmax * (1.0 / FP8_MAX)
            q = torch.clamp(grp / scale, FP8_MIN, FP8_MAX)
            y_q[m, g * GROUP_SIZE:(g + 1) * GROUP_SIZE] = (
                q.to(FP8_DTYPE).to(torch.float32)
            )
            y_s[m, g] = scale
    return y_q, y_s


def kernel_impl(y):
    y_q, y_s = silu_mul_per_token_group_quant_fp8_colmajor(
        y.contiguous(), use_ue8m0=False, eps=EPS, clamp_limit=None
    )
    return y_q.to(torch.float32), y_s


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_q, ref_s = pytorch_ref(Y)
        kern_q, kern_s_cm = kernel_impl(Y)

        ref_q = ref_q.cpu()
        ref_s = ref_s.cpu()
        kern_q = kern_q.cpu()

        cm_stride = tuple(kern_s_cm.stride())
        is_colmajor = cm_stride[0] == 1
        kern_s = kern_s_cm.to(torch.float32).cpu()

        ref_deq = ref_q * ref_s.repeat_interleave(GROUP_SIZE, dim=1)
        kern_deq = kern_q * kern_s.repeat_interleave(GROUP_SIZE, dim=1)

        torch.testing.assert_close(kern_deq, ref_deq, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(kern_s, ref_s, rtol=1e-3, atol=1e-3)
        assert is_colmajor, f"scale tensor not column-major: stride={cm_stride}"

        diff_v = (kern_deq - ref_deq).abs()
        diff_s = (kern_s - ref_s).abs()
        stats = {
            "input_shape": tuple(Y.shape),
            "output_shape": tuple(kern_q.shape),
            "ys_shape": tuple(kern_s.shape),
            "ys_stride": cm_stride,
            "is_colmajor": is_colmajor,
            "in_dtype": str(Y.dtype),
            "fp8_dtype": str(FP8_DTYPE),
            "out_dtype": str(FP8_DTYPE),
            "device": str(Y.device),
            "max_abs_diff": diff_v.max().item(),
            "mean_abs_diff": diff_v.mean().item(),
            "max_abs_diff_scale": diff_s.max().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(Y))
        kern_stats = _bench(lambda: kernel_impl(Y))
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
            "Kernel: _silu_mul_per_token_group_quant_fp8_colmajor\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- y shape: {stats['input_shape']} (N_2={N_2})\n")
            lines.append(f"- M={M}, N={N}, group_size={GROUP_SIZE}, eps={EPS}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- fp8_dtype: {stats['fp8_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- y_q shape: {stats['output_shape']}\n")
            lines.append(f"- y_s shape: {stats['ys_shape']}\n")
            lines.append(f"- y_s stride (colmajor): {stats['ys_stride']}\n")
            lines.append(f"- is_colmajor: {stats['is_colmajor']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff (dequant): {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff (dequant): {stats['mean_abs_diff']}\n")
            lines.append(f"- max_abs_diff (scale): {stats['max_abs_diff_scale']}\n")
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
