"""
Standalone QAIC validation for `_dequant_gather_slots_kernel`.

Source under test:
vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py
  - _dequant_gather_slots_kernel
  - launcher: dequant_gather_slots(out, cache, indices, cache_block_size)

Dequantizes scattered FP8 UE8M0 KV-cache slots into a flat BF16 workspace.
For each gathered slot index (one program per slot):
  * invalid slot (index < 0)  -> output row is all zeros.
  * otherwise the 512-wide output row is:
      - first 448 dims: dequantized FP8. Split into 7 blocks of 64 elements;
        each block b has a UE8M0 byte scale s_b, output = fp8_byte -> float *
        2^(s_b - 127), stored bf16.
      - last 64 dims: copied directly from the token's bf16 region.
Cache block layout (block_size tokens, head_bytes = 576 + 8 = 584):
    [0, bs*576):        token data (448 fp8 bytes + 128 bf16 bytes each)
    [bs*576, +bs*8):    UE8M0 scale bytes (7 real + 1 pad per token)

Config tested: num_blocks=4, block_size=8, 5 gathered slots (one invalid -1).
Reference: pure PyTorch reading the same raw cache bytes and applying the
identical dequant math. Compared with assert_close (bf16 tolerance).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dequant_gather_slots_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/xpu/xpu_sparse_decode_fp8.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import (  # noqa: E402
    dequant_gather_slots,
    OUTPUT_DIM,
    TOKEN_FP8_DIM,
    TOKEN_BF16_DIM,
    TOKEN_SCALE_DIM,
    QUANT_BLOCK_SIZE,
    TOKEN_DATA_SIZE,
)

torch.manual_seed(42)

NUM_BLOCKS = 4
BLOCK_SIZE = 8
# head_bytes per token-block row: token data (576) + scales (8) = 584
HEAD_BYTES = TOKEN_DATA_SIZE + TOKEN_SCALE_DIM  # 576 + 8 = 584
N_QUANT_BLOCKS = 7  # 448 / 64
FP8_DTYPE = torch.float8_e4m3fn

TOTAL_SLOTS = 5
# indices into [0, NUM_BLOCKS*BLOCK_SIZE); include one invalid (-1)
INDICES = torch.tensor([3, 17, -1, 8, 25], dtype=torch.int32, device=DEVICE)


def _build_cache_flat():
    """Flat-correct cache construction matching the kernel's byte addressing.

    The kernel addresses scales at block_base + block_size*token_data_size +
    pos*scale_dim (the scale region begins after ALL token data in the block),
    and block_stride = cache.stride(0). We lay the cache out flat to match this
    exact byte addressing, then view it as [num_blocks, block_size, head_bytes].
    """
    total_tokens = NUM_BLOCKS * BLOCK_SIZE
    fp8_vals = (torch.randn(total_tokens, TOKEN_FP8_DIM) * 2.0).to(FP8_DTYPE)
    fp8_bytes = fp8_vals.view(torch.uint8)
    scale_bytes = torch.randint(
        120, 135, (total_tokens, TOKEN_SCALE_DIM), dtype=torch.uint8
    )
    bf16_vals = torch.randn(total_tokens, TOKEN_BF16_DIM, dtype=torch.bfloat16)
    bf16_bytes = bf16_vals.view(torch.uint8)

    block_stride = BLOCK_SIZE * HEAD_BYTES  # 8 * 584
    cache_flat = torch.zeros(NUM_BLOCKS * block_stride, dtype=torch.uint8)
    for blk in range(NUM_BLOCKS):
        block_base = blk * block_stride
        for pos in range(BLOCK_SIZE):
            tok = blk * BLOCK_SIZE + pos
            td = block_base + pos * TOKEN_DATA_SIZE
            cache_flat[td : td + TOKEN_FP8_DIM] = fp8_bytes[tok]
            cache_flat[td + TOKEN_FP8_DIM : td + TOKEN_DATA_SIZE] = bf16_bytes[tok]
            sc = block_base + BLOCK_SIZE * TOKEN_DATA_SIZE + pos * TOKEN_SCALE_DIM
            cache_flat[sc : sc + TOKEN_SCALE_DIM] = scale_bytes[tok]
    cache = cache_flat.view(NUM_BLOCKS, BLOCK_SIZE, HEAD_BYTES)
    return cache, fp8_vals, scale_bytes, bf16_vals


CACHE, FP8_VALS, SCALE_BYTES, BF16_VALS = _build_cache_flat()
CACHE = CACHE.to(DEVICE)


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


def pytorch_ref(indices):
    """Pure PyTorch dequant+gather reading the same raw cache bytes."""
    idx = indices.cpu()
    out = torch.zeros(TOTAL_SLOTS, OUTPUT_DIM, dtype=torch.bfloat16)
    fp8 = FP8_VALS.cpu().float()  # [total_tokens, 448]
    scales = SCALE_BYTES.cpu()  # [total_tokens, 8]
    bf16 = BF16_VALS.cpu()  # [total_tokens, 64]
    for i in range(TOTAL_SLOTS):
        slot = int(idx[i].item())
        if slot < 0:
            continue  # zeros
        # dequant fp8 portion, 7 blocks of 64
        row = torch.zeros(OUTPUT_DIM, dtype=torch.float32)
        for b in range(N_QUANT_BLOCKS):
            s = float(2.0 ** (int(scales[slot, b].item()) - 127))
            lo = b * QUANT_BLOCK_SIZE
            hi = lo + QUANT_BLOCK_SIZE
            row[lo:hi] = fp8[slot, lo:hi] * s
        out_row = row.to(torch.bfloat16)
        # bf16 portion copied directly
        out_row[TOKEN_FP8_DIM:OUTPUT_DIM] = bf16[slot]
        out[i] = out_row
    return out


def kernel_impl(indices):
    """Kernel launch only."""
    out = torch.empty(TOTAL_SLOTS, OUTPUT_DIM, dtype=torch.bfloat16, device=CACHE.device)
    dequant_gather_slots(out, CACHE, indices, BLOCK_SIZE)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(INDICES)
        kernel_out = kernel_impl(INDICES)

        ref_cpu = ref_out.cpu().float()
        kernel_cpu = kernel_out.cpu().float()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-2, atol=1e-2)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(INDICES.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(INDICES.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(INDICES.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(INDICES))
        kern_stats = _bench(lambda: kernel_impl(INDICES))
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
            "Kernel: _dequant_gather_slots_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- indices shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(
                f"- cache: [{NUM_BLOCKS}, {BLOCK_SIZE}, {HEAD_BYTES}] uint8\n"
            )
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output (dequantized bf16 rows):\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
