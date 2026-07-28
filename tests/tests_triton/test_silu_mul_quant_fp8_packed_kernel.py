"""
Standalone QAIC validation for `_silu_mul_quant_fp8_packed_kernel`.

Source under test:
vllm/model_executor/layers/quantization/utils/fp8_utils.py
  - _silu_mul_quant_fp8_packed_kernel  (fused SiLU-and-multiply of a gated
    activation followed by per-group UE8M0 FP8 quantization, with the group
    scale exponents bit-packed 4-per-int32). Launched via the public wrapper
    `silu_mul_quant_fp8_packed_triton`.

Exact source math (per group of GROUP_SIZE cols, no clamp path):
    act = x[:, :N/2];  mul = x[:, N/2:]
    y   = (act * sigmoid(act)) * mul            (SiLU-gate then multiply)
    y   = round_to_bf16(y)                      (kernel rounds through bf16)
    absmax = max(|y|) over the group
    scale  = 2^ceil(log2(max(absmax/fp8_max, 1e-10)))   (UE8M0, power of two)
    y_q    = clamp(y / scale, fp8_min, fp8_max) -> float8_e4m3fn
    packed_scale byte(pack_idx) = clamp(exp + 127, 0, 255)   (int32, 4 groups/word)

Because the FP8 output alone is not comparable to a dense reference, we
DEQUANTIZE the kernel output (y_q * scale, unpacking the int32 scales) and
compare against the identical pure-PyTorch quant->dequant pipeline. Both apply
the same power-of-two UE8M0 scale and the same FP8 rounding, so this is a
faithful numeric check. Tolerance is slightly relaxed (rtol/atol 1e-2) to allow
any device-vs-CPU FP8 tie-breaking difference in the final cast.

Config tested: M=8, N=512 (N/2=256), group_size=128 (2 groups/row,
1 packed word), no clamp_limit. Input dtype bf16, output fp8_e4m3fn.
Reference: pure PyTorch SiLU-mul + UE8M0 FP8 quant + dequant.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_silu_mul_quant_fp8_packed_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/quantization/utils/fp8_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: E402
    silu_mul_quant_fp8_packed_triton,
)

torch.manual_seed(42)

M = 8
N = 512
N_2 = N // 2
GROUP_SIZE = 128
GROUPS_PER_ROW = N_2 // GROUP_SIZE
FP8_DTYPE = torch.float8_e4m3fn
_finfo = torch.finfo(FP8_DTYPE)
FP8_MIN, FP8_MAX = _finfo.min, _finfo.max

# bf16 input; second half kept moderate so SiLU-gate * mul stays in fp8 range.
INPUT = (torch.randn(M, N, dtype=torch.float32, device=DEVICE) * 2.0).to(
    torch.bfloat16
).contiguous()


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


def pytorch_ref(x):
    """Pure PyTorch SiLU-mul + UE8M0 FP8 quant, returned as dequantized fp32."""
    act = x[:, :N_2].float()
    mul = x[:, N_2:].float()
    y = (act * torch.sigmoid(act)) * mul
    y = y.to(torch.bfloat16).to(torch.float32)  # kernel rounds through bf16

    yg = y.view(M, GROUPS_PER_ROW, GROUP_SIZE)
    absmax = yg.abs().amax(dim=-1)  # [M, groups]
    scale_raw = torch.clamp(absmax / FP8_MAX, min=1e-10)
    exp = torch.ceil(torch.log2(scale_raw))
    scale = torch.exp2(exp)  # [M, groups]
    yq = torch.clamp(yg / scale[..., None], FP8_MIN, FP8_MAX).to(FP8_DTYPE)
    dequant = (yq.float() * scale[..., None]).reshape(M, N_2)
    return dequant


def _dequant_kernel_output(output_q, scale_packed):
    """Reconstruct fp32 from the FP8 output and packed int32 UE8M0 scales."""
    oq = output_q.float().cpu()
    sp = scale_packed.cpu().to(torch.int64)
    out = torch.empty(M, N_2, dtype=torch.float32)
    for g in range(GROUPS_PER_ROW):
        pack = g // 4
        pidx = g % 4
        byte = (sp[:, pack] >> (pidx * 8)) & 0xFF  # [M]
        exp = byte.float() - 127.0
        scale = torch.exp2(exp)
        cols = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        out[:, cols] = oq[:, cols] * scale[:, None]
    return out


def kernel_impl(x):
    """Kernel launch only. Returns (output_q fp8, packed int32 scales)."""
    return silu_mul_quant_fp8_packed_triton(x, group_size=GROUP_SIZE)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_dequant = pytorch_ref(INPUT).cpu()
        output_q, scale_packed = kernel_impl(INPUT)
        kernel_dequant = _dequant_kernel_output(output_q, scale_packed)

        torch.testing.assert_close(
            kernel_dequant, ref_dequant, rtol=1e-2, atol=1e-2
        )

        diff = (kernel_dequant - ref_dequant).abs()
        rel_error = (diff / (ref_dequant.abs() + 1e-8)).mean().item()
        stats = {
            "input_shape": tuple(INPUT.shape),
            "output_shape": tuple(output_q.shape),
            "in_dtype": str(INPUT.dtype),
            "out_dtype": str(output_q.dtype),
            "device": str(INPUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": rel_error,
        }

        pt_stats = _bench(lambda: pytorch_ref(INPUT))
        kern_stats = _bench(lambda: kernel_impl(INPUT))
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
            "Kernel: _silu_mul_quant_fp8_packed_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output_q shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff (dequantized): {stats['max_abs_diff']}\n",
                f"- mean_abs_diff (dequantized): {stats['mean_abs_diff']}\n",
                f"- rel_error (dequantized): {stats['rel_error']}\n",
                "Timing:\n",
                f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
            ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
