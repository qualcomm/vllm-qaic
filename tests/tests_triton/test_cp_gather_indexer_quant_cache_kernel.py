"""
Standalone QAIC validation for `_cp_gather_indexer_quant_cache_kernel`.

Source under test:
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
  - _cp_gather_indexer_quant_cache_kernel
  - launcher: cp_gather_indexer_k_quant_cache_triton(k_cache, k_fp8, k_scale,
        block_table, cu_seqlen, token_to_seq)

Gathers FP8-quantized indexer keys + their float32 scales out of the paged
indexer cache into contiguous per-token buffers. Per output token `tid`:

    b            = token_to_seq[tid]
    batch_offset = tid - cu_seqlen[b]
    block_id     = block_table[b, batch_offset // block_size]
    (block_size == 1 -> NORMAL layout: pos_in_block = 0)
    k_fp8[tid]   = cache_value[block_id]         (raw fp8 copy)
    k_scale[tid] = cache_scale[block_id]         (float32 copy)

The kernel is a pure gather (no dequant inside). We first populate the cache with
`indexer_k_quant_and_cache_triton`, then gather and verify. Comparison choice
(NOT executing on device): compare the gathered fp8 values (as float) and the
gathered float scales directly against a PyTorch gather from the same cache
tensors, rtol/atol = 1e-3 (values are byte-for-byte copies, so this is tight).
block_size == 1 selects the NORMAL layout; the SHUFFLE tiled branch is not
exercised (documented simplification).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    cp_gather_indexer_k_quant_cache_triton,
    indexer_k_quant_and_cache_triton,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_cp_gather_indexer_quant_cache_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"

DEVICE = "qaic"
torch.manual_seed(42)

HEAD_DIM = 32
BLOCK_SIZE = 1  # NORMAL layout
NUM_BLOCKS = 16
SCALE_FMT = "ue8m0"

# Two batches: batch0 has 2 tokens, batch1 has 3 tokens -> 5 gathered tokens.
CU_SEQLEN = torch.tensor([0, 2, 5], dtype=torch.int32, device=DEVICE)
NUM_TOKENS = int(CU_SEQLEN[-1].item())
TOKEN_TO_SEQ = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=DEVICE)
MAX_BLOCKS_PER_SEQ = 4
# physical block per (batch, logical pos)
BLOCK_TABLE = torch.tensor(
    [[3, 7, 0, 0], [10, 4, 12, 0]], dtype=torch.int32, device=DEVICE
)

try:
    FP8_DTYPE = current_platform.fp8_dtype()
except Exception:
    FP8_DTYPE = torch.float8_e4m3fn


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


def _build_populated_cache():
    """Fill a paged cache by quantizing random K into the physical blocks that
    the gather will read from."""
    cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4, dtype=torch.uint8, device=DEVICE
    )
    # Determine which physical slots are referenced.
    slots = []
    for tid in range(NUM_TOKENS):
        b = int(TOKEN_TO_SEQ[tid].item())
        off = tid - int(CU_SEQLEN[b].item())
        block_id = int(BLOCK_TABLE[b, off // BLOCK_SIZE].item())
        slots.append(block_id * BLOCK_SIZE)  # pos_in_block == 0
    k = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    slot_mapping = torch.tensor(slots, dtype=torch.int64, device=DEVICE)
    indexer_k_quant_and_cache_triton(
        k, cache, slot_mapping, quant_block_size=HEAD_DIM, scale_fmt=SCALE_FMT
    )
    return cache


def pytorch_ref(cache):
    """Gather raw stored value/scale from the cache for each token."""
    flat = cache.view(NUM_BLOCKS, -1)
    value = flat[:, : BLOCK_SIZE * HEAD_DIM].view(FP8_DTYPE)
    scale = flat[:, BLOCK_SIZE * HEAD_DIM :].view(torch.float32)
    out_val = torch.empty(NUM_TOKENS, HEAD_DIM, dtype=torch.float32)
    out_scale = torch.empty(NUM_TOKENS, dtype=torch.float32)
    for tid in range(NUM_TOKENS):
        b = int(TOKEN_TO_SEQ[tid].item())
        off = tid - int(CU_SEQLEN[b].item())
        block_id = int(BLOCK_TABLE[b, off // BLOCK_SIZE].item())
        out_val[tid] = value[block_id].to(torch.float32).reshape(
            BLOCK_SIZE, HEAD_DIM
        )[0]
        out_scale[tid] = scale[block_id, 0]
    return out_val, out_scale


def kernel_impl(cache):
    k_fp8 = torch.empty(NUM_TOKENS, HEAD_DIM, dtype=FP8_DTYPE, device=DEVICE)
    k_scale = torch.zeros(NUM_TOKENS, 4, dtype=torch.uint8, device=DEVICE)
    cp_gather_indexer_k_quant_cache_triton(
        cache,
        k_fp8,
        k_scale,
        BLOCK_TABLE,
        CU_SEQLEN,
        TOKEN_TO_SEQ,
    )
    return k_fp8.to(torch.float32).cpu(), k_scale.view(torch.float32).reshape(-1).cpu()


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cache = _build_populated_cache()
        ref_val, ref_scale = pytorch_ref(cache)
        k_val, k_scale = kernel_impl(cache)

        torch.testing.assert_close(k_val, ref_val, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_scale, ref_scale, rtol=1e-3, atol=1e-3)

        v_diff = (k_val - ref_val).abs()
        s_diff = (k_scale - ref_scale).abs()
        stats = {
            "cache_shape": (NUM_BLOCKS, BLOCK_SIZE, HEAD_DIM + 4),
            "k_fp8_shape": tuple(k_val.shape),
            "k_scale_shape": tuple(k_scale.shape),
            "fp8_dtype": str(FP8_DTYPE),
            "device": DEVICE,
            "val_max_abs_diff": v_diff.max().item(),
            "val_mean_abs_diff": v_diff.mean().item(),
            "scale_max_abs_diff": s_diff.max().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(cache))
        kern_stats = _bench(lambda: kernel_impl(cache))
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
            "Kernel: _cp_gather_indexer_quant_cache_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- cache shape: {stats['cache_shape']} (populated via quant kernel)\n",
                f"- cu_seqlen: {CU_SEQLEN.cpu().tolist()}, token_to_seq: {TOKEN_TO_SEQ.cpu().tolist()}\n",
                f"- head_dim={HEAD_DIM}, block_size={BLOCK_SIZE} (NORMAL layout), fp8={stats['fp8_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (gathered value + scale comparison, rtol/atol=1e-3):\n",
                f"- k_fp8 shape: {stats['k_fp8_shape']}\n",
                f"- k_scale shape: {stats['k_scale_shape']}\n",
                f"- gathered value max_abs_diff: {stats['val_max_abs_diff']}\n",
                f"- gathered value mean_abs_diff: {stats['val_mean_abs_diff']}\n",
                f"- gathered scale max_abs_diff: {stats['scale_max_abs_diff']}\n",
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
