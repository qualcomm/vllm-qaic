"""
Standalone QAIC validation for `_nvfp4_quant_dequant_kernel`.

Source under test:
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
  - _nvfp4_quant_dequant_kernel  (@triton.jit)
  - launcher: _triton_nvfp4_quant_dequant(x, global_scale, block_size)

Emulates NVFP4 quantize-then-dequantize per row / per block of `block_size`
elements:
    vec_max      = max(|x|) over the block
    scale        = clamp(global_scale * vec_max / FP4_MAX(=6), -448, 448)
                   -> cast to float8_e4m3fn -> f32   (UE8M0-style block scale
                      but stored as fp8 e4m3, matching the source)
    output_scale = global_scale / scale             (0 if scale == 0)
    scaled_x     = clamp(x * output_scale, -6, 6)
    fp4_val      = round_to_E2M1(scaled_x)           (nearest grid value)
    result       = fp4_val * (scale / global_scale)  (dequantized)

FLOAT/quant kernel. Comparison choice (NOT executing on device): the reference
recomputes the identical block scale (fp8 e4m3 roundtrip is deterministic) and
E2M1 rounding, then compares the round-tripped fp32 values at rtol/atol=1e-3.
The E2M1 grid values are exact, so at non-tie inputs this is effectively exact.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _triton_nvfp4_quant_dequant,
    FLOAT4_E2M1_MAX_RECIPROCAL,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_nvfp4_quant_dequant_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
)

DEVICE = "qaic"
torch.manual_seed(42)

M = 4
BLOCK_SIZE = 16
NUM_BLOCKS = 2
K = BLOCK_SIZE * NUM_BLOCKS  # 32

X = torch.randn(M, K, dtype=torch.float32, device=DEVICE)
GLOBAL_SCALE = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _round_to_fp4(a):
    """Round abs values to E2M1 grid (matches _round_to_fp4 / cast_to_fp4)."""
    sign = torch.where(a < 0.0, -1.0, 1.0)
    a = a.abs()
    result = torch.where(a > 5.0, torch.tensor(6.0), torch.tensor(0.0))
    result = torch.where((a >= 3.5) & (a <= 5.0), torch.tensor(4.0), result)
    result = torch.where((a > 2.5) & (a < 3.5), torch.tensor(3.0), result)
    result = torch.where((a >= 1.75) & (a <= 2.5), torch.tensor(2.0), result)
    result = torch.where((a > 1.25) & (a < 1.75), torch.tensor(1.5), result)
    result = torch.where((a >= 0.75) & (a <= 1.25), torch.tensor(1.0), result)
    result = torch.where((a > 0.25) & (a < 0.75), torch.tensor(0.5), result)
    return result * sign


def pytorch_ref(x, global_scale):
    x = x.cpu().to(torch.float32)
    gs = float(global_scale.cpu().item())
    x_blk = x.reshape(M, NUM_BLOCKS, BLOCK_SIZE)
    vec_max = x_blk.abs().amax(dim=-1)  # (M, NUM_BLOCKS)
    scale = gs * (vec_max * FLOAT4_E2M1_MAX_RECIPROCAL)
    scale = torch.clamp(scale, -448.0, 448.0)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    output_scale = torch.where(scale == 0.0, torch.tensor(0.0), gs / scale)
    scaled = torch.clamp(x_blk * output_scale.unsqueeze(-1), -6.0, 6.0)
    fp4 = _round_to_fp4(scaled)
    dequant_scale = (scale / gs).unsqueeze(-1)
    result = fp4 * dequant_scale
    return result.reshape(M, K)


def kernel_impl(x, global_scale):
    return _triton_nvfp4_quant_dequant(x, global_scale, BLOCK_SIZE)


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X, GLOBAL_SCALE)
        kernel_out = kernel_impl(X, GLOBAL_SCALE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(X.dtype),
            "device": DEVICE,
            "block_size": BLOCK_SIZE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, GLOBAL_SCALE))
        kern_stats = _bench(lambda: kernel_impl(X, GLOBAL_SCALE))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _nvfp4_quant_dequant_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- x shape: {stats['input_shape']} dtype {stats['dtype']}\n",
                f"- global_scale: {float(GLOBAL_SCALE.cpu().item())}\n",
                f"- block_size: {stats['block_size']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (quant->dequant fp32, rtol/atol=1e-3):\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
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
