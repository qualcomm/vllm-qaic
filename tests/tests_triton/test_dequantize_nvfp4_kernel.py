"""
Standalone QAIC validation for `_dequantize_nvfp4_kernel`.

Source under test:
vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py
  - _dequantize_nvfp4_kernel  (@triton.jit)
  - launcher: _triton_dequantize_nvfp4(tensor_fp4, tensor_sf, global_scale,
              dtype, block_size)

Dequantizes packed NVFP4 (E2M1, 2 nibbles/byte) tensors back to higher
precision. Layout (swizzle=False), per row / per block of `block_size`
elements:
    value = E2M1_decode(nibble) * (sf_block_fp8_e4m3 -> f32) * global_scale
Nibble order matches `break_fp4_bytes`: within each byte the LOW nibble is the
even-index element and the HIGH nibble is the odd-index element; the kernel
reproduces this via `tl.interleave(low, high)`. Per-block scales are stored as
float8_e4m3fn (the source bitcasts the raw scale byte to `tl.float8e4nv`); the
global scale is a single fp32 scalar.

FLOAT/quant kernel. Comparison choice (NOT executing on device): we build a
layout-valid packed buffer + fp8 block scales ourselves and compare the
DEQUANTIZED fp32 output against a pure-PyTorch dequant at rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    _triton_dequantize_nvfp4,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dequantize_nvfp4_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
)

DEVICE = "qaic"
torch.manual_seed(42)

_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

M = 4
BLOCK_SIZE = 16
NUM_BLOCKS = 2
K = BLOCK_SIZE * NUM_BLOCKS       # 32
PACKED_K = K // 2                 # 16

# Random packed nibbles (each byte = two 4-bit E2M1 codes).
TENSOR_FP4 = torch.randint(0, 256, (M, PACKED_K), dtype=torch.uint8, device=DEVICE)
# Per-block scales stored as float8_e4m3fn: (M, NUM_BLOCKS).
_SF_F32 = (torch.rand(M, NUM_BLOCKS, dtype=torch.float32) * 2.0 + 0.25)
TENSOR_SF = _SF_F32.to(torch.float8_e4m3fn).to(DEVICE)
GLOBAL_SCALE = torch.tensor([0.5], dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _e2m1_decode(code: int) -> float:
    mag = _E2M1_MAG[code & 7]
    return -mag if (code & 8) else mag


def pytorch_ref(tensor_fp4, tensor_sf, global_scale):
    fp4 = tensor_fp4.cpu()
    # Decode fp8 block scales to fp32.
    sf = tensor_sf.cpu().to(torch.float32)
    gs = float(global_scale.cpu().item())
    out = torch.empty(M, K, dtype=torch.float32)
    half_block = BLOCK_SIZE // 2  # bytes per block
    for r in range(M):
        for b in range(NUM_BLOCKS):
            scale = float(sf[r, b].item()) * gs
            byte_base = b * half_block
            for i in range(half_block):
                byte = int(fp4[r, byte_base + i].item())
                low = byte & 0x0F
                high = (byte >> 4) & 0x0F
                out[r, b * BLOCK_SIZE + 2 * i] = _e2m1_decode(low) * scale
                out[r, b * BLOCK_SIZE + 2 * i + 1] = _e2m1_decode(high) * scale
    return out


def kernel_impl(tensor_fp4, tensor_sf, global_scale):
    return _triton_dequantize_nvfp4(
        tensor_fp4,
        tensor_sf,
        global_scale,
        torch.float32,
        BLOCK_SIZE,
    )


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
        ref_out = pytorch_ref(TENSOR_FP4, TENSOR_SF, GLOBAL_SCALE)
        kernel_out = kernel_impl(TENSOR_FP4, TENSOR_SF, GLOBAL_SCALE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "fp4_shape": tuple(TENSOR_FP4.shape),
            "sf_shape": tuple(TENSOR_SF.shape),
            "output_shape": tuple(kernel_out.shape),
            "device": DEVICE,
            "block_size": BLOCK_SIZE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(TENSOR_FP4, TENSOR_SF, GLOBAL_SCALE))
        kern_stats = _bench(lambda: kernel_impl(TENSOR_FP4, TENSOR_SF, GLOBAL_SCALE))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _dequantize_nvfp4_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- tensor_fp4 shape: {stats['fp4_shape']} (uint8 packed E2M1)\n",
                f"- tensor_sf shape: {stats['sf_shape']} (float8_e4m3fn block scales)\n",
                f"- global_scale: {float(GLOBAL_SCALE.cpu().item())}\n",
                f"- block_size: {stats['block_size']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (dequantized fp32, rtol/atol=1e-3):\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
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
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
