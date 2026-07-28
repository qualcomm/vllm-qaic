"""
Standalone QAIC validation for `_indexer_k_quant_and_cache_kernel`.

Source under test:
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
  - _indexer_k_quant_and_cache_kernel
  - launcher: indexer_k_quant_and_cache_triton(k, kv_cache, slot_mapping,
        quant_block_size, scale_fmt)

Quantizes newly computed indexer keys to FP8 (one scalar scale per token,
optionally rounded to a UE8M0 power-of-2) and scatters them into the sparse-MLA
indexer paged KV cache at the slot given by `slot_mapping` (slot < 0 skipped):

    amax  = max_d |k[token]|
    scale = max(1e-4, amax) / (224 if fnuz else 448)
    scale = 2^ceil(log2(scale))               (only if scale_fmt == "ue8m0")
    k_fp8 = (k / scale) as fp8
    cache_value[slot] = k_fp8 ;  cache_scale[slot] = scale

Cache layout (per block, block_size tokens):
    [0 : block_size*head_dim)         fp8 values (1 byte each)
    [block_size*head_dim : +bs*4)     float32 per-token scales
We use block_size == 1 -> LAYOUT == "NORMAL" (no SHUFFLE head/block tiling); the
SHUFFLE tiled-layout branch is not exercised (documented simplification).

FLOAT/quant kernel. Comparison choice (NOT executing on device): the reference
recomputes the scalar scale, quantizes to fp8 and DEQUANTIZES. After the kernel
runs we read the fp8 bytes + stored scale back from the cache and dequantize;
we compare dequantized K (rtol/atol=1e-3) and the stored scales.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    indexer_k_quant_and_cache_triton,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_indexer_k_quant_and_cache_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 4
HEAD_DIM = 32
BLOCK_SIZE = 1  # NORMAL layout
NUM_BLOCKS = 8
SCALE_FMT = "ue8m0"

try:
    FP8_DTYPE = current_platform.fp8_dtype()
except Exception:
    FP8_DTYPE = torch.float8_e4m3fn
IS_FNUZ = FP8_DTYPE == torch.float8_e4m3fnuz
FP8_DIV = 224.0 if IS_FNUZ else 448.0

K = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
# distinct slots, one skipped (-1)
SLOT_MAPPING = torch.tensor([2, 5, -1, 0], dtype=torch.int64, device=DEVICE)


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


def _make_cache():
    # [num_blocks, block_size, head_dim + 4] uint8
    return torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE
    )


def pytorch_ref(k, slot_mapping):
    """Return dict slot -> (dequant_value[head_dim], scale) for written slots."""
    k = k.cpu().to(torch.float32)
    slot_mapping = slot_mapping.cpu()
    out = {}
    for t in range(NUM_TOKENS):
        slot = int(slot_mapping[t].item())
        if slot < 0:
            continue
        amax = k[t].abs().max()
        scale = torch.clamp(amax, min=1e-4) / FP8_DIV
        if SCALE_FMT == "ue8m0":
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        s = float(scale.item())
        q = (k[t] / s).to(FP8_DTYPE).to(torch.float32)
        deq = q * s
        out[slot] = (deq, s)
    return out


def kernel_impl(k, slot_mapping):
    kv_cache = _make_cache()
    indexer_k_quant_and_cache_triton(
        k, kv_cache, slot_mapping, quant_block_size=HEAD_DIM, scale_fmt=SCALE_FMT
    )
    # read back
    flat = kv_cache.view(NUM_BLOCKS, -1)
    value = flat[:, : BLOCK_SIZE * HEAD_DIM].view(FP8_DTYPE)
    scale = flat[:, BLOCK_SIZE * HEAD_DIM :].view(torch.float32)
    out = {}
    for slot in range(NUM_BLOCKS * BLOCK_SIZE):
        blk = slot // BLOCK_SIZE
        s = float(scale[blk, slot % BLOCK_SIZE].item())
        deq = value[blk].to(torch.float32).reshape(BLOCK_SIZE, HEAD_DIM)[
            slot % BLOCK_SIZE
        ] * s
        out[slot] = (deq.cpu(), s)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(K, SLOT_MAPPING)
        k_all = kernel_impl(K, SLOT_MAPPING)

        max_v = 0.0
        sum_v = 0.0
        n = 0
        max_s = 0.0
        for slot, (ref_deq, ref_s) in ref.items():
            k_deq, k_s = k_all[slot]
            torch.testing.assert_close(k_deq, ref_deq, rtol=1e-3, atol=1e-3)
            d = (k_deq - ref_deq).abs()
            max_v = max(max_v, float(d.max().item()))
            sum_v += float(d.sum().item())
            n += d.numel()
            max_s = max(max_s, abs(k_s - ref_s))
        assert max_s <= 1e-6, f"scale mismatch {max_s}"

        stats = {
            "input_shape": tuple(K.shape),
            "cache_shape": (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4),
            "in_dtype": str(K.dtype),
            "fp8_dtype": str(FP8_DTYPE),
            "device": DEVICE,
            "num_written_slots": len(ref),
            "q_max_abs_diff": max_v,
            "q_mean_abs_diff": (sum_v / n) if n else 0.0,
            "scale_max_abs_diff": max_s,
        }

        pt_stats = _bench(lambda: pytorch_ref(K, SLOT_MAPPING))
        kern_stats = _bench(lambda: kernel_impl(K, SLOT_MAPPING))
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
            "Kernel: _indexer_k_quant_and_cache_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- k shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- slot_mapping: {SLOT_MAPPING.cpu().tolist()}\n",
                f"- head_dim={HEAD_DIM}, block_size={BLOCK_SIZE} (NORMAL layout), "
                f"scale_fmt={SCALE_FMT}, fp8={stats['fp8_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (dequantized-value + scale comparison, rtol/atol=1e-3):\n",
                f"- cache shape: {stats['cache_shape']}\n",
                f"- written slots: {stats['num_written_slots']}\n",
                f"- q dequant max_abs_diff: {stats['q_max_abs_diff']}\n",
                f"- q dequant mean_abs_diff: {stats['q_mean_abs_diff']}\n",
                f"- scale max_abs_diff: {stats['scale_max_abs_diff']}\n",
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
