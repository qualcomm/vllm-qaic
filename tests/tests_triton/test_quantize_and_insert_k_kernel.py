"""
Standalone QAIC validation for `quantize_and_insert_k_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/cache_utils.py
  - quantize_and_insert_k_kernel  (quantize bf16 K [num_tokens, 512] to UE8M0
    FP8 for the first 448 dims + store the last 64 dims as raw bf16, then insert
    into the DeepseekV4 paged K-cache byte layout). Launched via the public
    wrapper `quantize_and_insert_k_cache`.

Cache block byte layout (block_size=64 tokens), all in a uint8 buffer:
  - [0, 64*576): token data, each token = 448 fp8 bytes + 128 bf16 bytes
  - [64*576, 64*576 + 64*8): scales, each token 8 uint8 (7 real + 1 pad)
Per 64-element quant block:
    block_max = max(|x|, 1e-4)
    scale = 2^ceil(log2(block_max/448))              (UE8M0 power of two)
    fp8   = clamp(x/scale, -448, 448) -> float8_e4m3fn
    stored_scale_byte = clamp(exp + 127, 0, 255)

Validation: run the kernel to fill the cache, then DECODE the cache back to
fp32 with a pure-PyTorch reader of the byte layout, and compare against a pure
PyTorch quantize->dequantize of the same K (identical UE8M0 + fp8 algorithm).
Only non-padded tokens (slot != -1) are compared.

Config tested: num_tokens=10 (2 padded), 512 dims, block_size=64, UE8M0.
Reference: pure PyTorch UE8M0 FP8 quant/dequant + bf16 passthrough.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_quantize_and_insert_k_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/cache_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.cache_utils import (  # noqa: E402
    quantize_and_insert_k_cache,
)

torch.manual_seed(42)

FP8_DIM = 448
BF16_DIM = 64
INPUT_DIM = 512
SCALE_DIM = 8
QUANT_BLOCK = 64
N_QUANT_BLOCKS = 7
FP8_MAX = 448.0
TOKEN_DATA_SIZE = FP8_DIM + BF16_DIM * 2  # 576
BLOCK_SIZE = 64
NUM_BLOCKS = 4
# padded to a multiple of TOKEN_DATA_SIZE
BLOCK_BYTES = (
    (BLOCK_SIZE * TOKEN_DATA_SIZE + BLOCK_SIZE * SCALE_DIM + TOKEN_DATA_SIZE - 1)
    // TOKEN_DATA_SIZE
) * TOKEN_DATA_SIZE

NUM_TOKENS = 10
FP8_DTYPE = torch.float8_e4m3fn

K = (torch.randn(NUM_TOKENS, INPUT_DIM, dtype=torch.float32, device=DEVICE) * 3.0).to(
    torch.bfloat16
)
_slots = torch.randperm(NUM_BLOCKS * BLOCK_SIZE, device=DEVICE)[:NUM_TOKENS]
SLOT_MAPPING = _slots.to(torch.int64)
SLOT_MAPPING[2] = -1
SLOT_MAPPING[7] = -1


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


def _quant_dequant_token(x_row):
    """UE8M0 FP8 quant+dequant of the first 448 dims; bf16 passthrough of last 64.

    Returns a fp32 (512,) reconstruction. Pure PyTorch.
    """
    x = x_row.float()
    out = torch.empty(INPUT_DIM, dtype=torch.float32)
    for g in range(N_QUANT_BLOCKS):
        blk = x[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK]
        block_max = torch.clamp(blk.abs().max(), min=1e-4)
        exp = torch.ceil(torch.log2(block_max / FP8_MAX))
        scale = torch.exp2(exp)
        q = torch.clamp(blk / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        out[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK] = q.float() * scale
    # bf16 passthrough of dims [448:512]
    out[FP8_DIM:INPUT_DIM] = x_row[FP8_DIM:INPUT_DIM].to(torch.bfloat16).float()
    return out


def pytorch_ref(k, slot_mapping):
    """Pure PyTorch expected reconstruction for every non-padded token."""
    slots = slot_mapping.cpu()
    k_cpu = k.cpu()
    recon = torch.zeros(NUM_TOKENS, INPUT_DIM, dtype=torch.float32)
    for tok in range(NUM_TOKENS):
        if int(slots[tok].item()) < 0:
            continue
        recon[tok] = _quant_dequant_token(k_cpu[tok])
    return recon


def _decode_cache(k_cache, slot_mapping):
    """Pure PyTorch reader of the packed uint8 cache -> fp32 [num_tokens, 512]."""
    cache = k_cache.cpu()
    slots = slot_mapping.cpu()
    recon = torch.zeros(NUM_TOKENS, INPUT_DIM, dtype=torch.float32)
    for tok in range(NUM_TOKENS):
        slot = int(slots[tok].item())
        if slot < 0:
            continue
        b = slot // BLOCK_SIZE
        p = slot % BLOCK_SIZE
        tok_data = cache[b, p * TOKEN_DATA_SIZE:(p + 1) * TOKEN_DATA_SIZE]
        fp8_bytes = tok_data[0:FP8_DIM].contiguous()
        fp8_vals = fp8_bytes.view(FP8_DTYPE).float()
        bf16_bytes = tok_data[FP8_DIM:TOKEN_DATA_SIZE].contiguous()
        bf16_vals = bf16_bytes.view(torch.bfloat16).float()
        scale_base = BLOCK_SIZE * TOKEN_DATA_SIZE + p * SCALE_DIM
        scale_bytes = cache[b, scale_base:scale_base + SCALE_DIM]
        for g in range(N_QUANT_BLOCKS):
            exp = float(scale_bytes[g].item()) - 127.0
            scale = 2.0 ** exp
            recon[tok, g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK] = (
                fp8_vals[g * QUANT_BLOCK:(g + 1) * QUANT_BLOCK] * scale
            )
        recon[tok, FP8_DIM:INPUT_DIM] = bf16_vals
    return recon


def kernel_impl(k, slot_mapping):
    """Kernel launch only. Returns the filled uint8 cache."""
    k_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_BYTES, dtype=torch.uint8, device=k.device
    )
    quantize_and_insert_k_cache(
        k, k_cache, slot_mapping, block_size=BLOCK_SIZE, is_ue8m0=True
    )
    return k_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_recon = pytorch_ref(K, SLOT_MAPPING)
        k_cache = kernel_impl(K, SLOT_MAPPING)
        kernel_recon = _decode_cache(k_cache, SLOT_MAPPING)

        # Compare only non-padded tokens.
        valid = (SLOT_MAPPING.cpu() >= 0)
        rr = ref_recon[valid]
        kr = kernel_recon[valid]
        torch.testing.assert_close(kr, rr, rtol=1e-2, atol=1e-2)

        diff = (kr - rr).abs()
        stats = {
            "input_shape": tuple(K.shape),
            "output_shape": tuple(k_cache.shape),
            "in_dtype": str(K.dtype),
            "out_dtype": str(k_cache.dtype),
            "device": str(K.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(K, SLOT_MAPPING))
        kern_stats = _bench(lambda: kernel_impl(K, SLOT_MAPPING))
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
            "Kernel: quantize_and_insert_k_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- K shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- cache shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff (decoded): {stats['max_abs_diff']}\n",
                f"- mean_abs_diff (decoded): {stats['mean_abs_diff']}\n",
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
