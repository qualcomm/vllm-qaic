"""
Standalone QAIC validation for `_sparse_attn_decode_ragged_kernel`.

Source under test:
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
  - _sparse_attn_decode_ragged_kernel  (ragged sparse MLA *decode* attention
    for DeepSeek-V4, over a block-paged fp8_ds_mla KV cache, with the
    query/key head split into a no-position-embedding (NoPE) part and a
    rotary (RoPE) part, plus an optional second "extra" cache and an optional
    per-head attention sink.)

For each decode query (one per sequence) and query head, the kernel walks the
sequence's ragged list of selected KV slots (`main_indices[main_indptr[q] :
main_indptr[q+1]]`), gathers each selected token from the packed fp8_ds_mla
cache, dequantizes it, computes online-softmax attention over just those
selected positions, and writes the weighted latent value back out.

Packed fp8_ds_mla cache layout (per KV block of `block_size` tokens):
  * A data region of `block_size * 576` bytes: for each token, 576 contiguous
    bytes = 448 FP8 (e4m3) NoPE bytes followed by 64 bfloat16 RoPE values
    (128 bytes). Token `p`'s data starts at byte offset `p * 576`.
  * A scale region right after it (`block_size * 576` byte offset), 8 bytes
    per token; byte `g` is the e8m0 exponent for NoPE dim-group `g`
    (dim // 64). Dequantized NoPE = fp8_value * 2**(exp - 127).
Value == Key (MLA latent), so the output is the softmax-weighted sum of the
full [NoPE(448) + RoPE(64)] gathered vectors, written split into the NoPE
slice [0:448) and RoPE slice [448:512) of each output row.

NoPE/RoPE score combination:
  The kernel computes `scores = q_nope @ k_nope^T + q_rope @ k_rope^T` (two
  disjoint dim ranges summed), then multiplies by `scale`. Because the NoPE
  and RoPE dims are disjoint and concatenated, this is exactly the full
  scaled q . k dot product; the reference gathers both halves and sums their
  dot products explicitly for faithfulness.

Branches exercised / skipped:
  * HAS_ATTN_SINK = False  (no attention sink; skipped, documented here).
  * HAS_EXTRA     = False  (single "main" cache only; the "extra"/topk cache
    path is skipped, documented here).
  * IS_FNUZ follows the platform default (OCP e4m3fn), matching the fp8 dtype
    used to build the cache.

The cache stores identical FP8/bf16 bytes that BOTH the kernel and the
reference dequantize, so quantization rounding cancels; with e8m0 exponent
127 (scale factor 1.0) the NoPE dequant is exact. Remaining error is only the
kernel's bf16 probability accumulation, so rtol/atol=1e-3 is used.

Reference: pure PyTorch gather + split-dim scaled dot product + softmax.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_sparse_attn_decode_ragged_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.platforms import current_platform  # noqa: E402
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (  # noqa: E402
    _rocm_sparse_attn_decode_ragged_triton,
)

# ---------------------------------------------------------------------------
# DSv4 sparse MLA fixed dims (enforced by _validate_dsv4_sparse_dims).
# ---------------------------------------------------------------------------
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
TOKEN_DATA_BYTES = 576  # 448 fp8 NoPE + 64 bf16 RoPE (128 bytes)
TOKEN_SCALE_BYTES = 8  # one e8m0 exponent per 64-dim NoPE group

# ---------------------------------------------------------------------------
# Global inputs (shared by both implementations)
# ---------------------------------------------------------------------------
DEVICE = "qaic"

NUM_SEQS = 2  # one decode query per sequence
NUM_HEADS = 2
NUM_BLOCKS = 2
BLOCK_SIZE = 8
NUM_ROWS = NUM_BLOCKS * BLOCK_SIZE  # total cache slots
TOPK = 6  # selected slots per query
SCALE = 1.0 / math.sqrt(HEAD_DIM)

_FP8_DTYPE = current_platform.fp8_dtype()  # e4m3fn on non-FNUZ platforms

torch.manual_seed(42)


def _build_cache():
    """Build the packed fp8_ds_mla cache and a parallel float 'k_ref' tensor.

    Returns:
        cache:  uint8 tensor [NUM_BLOCKS, BLOCK_SIZE, TOKEN_DATA_BYTES +
                TOKEN_SCALE_BYTES], packed exactly as the kernel indexes it.
        k_ref:  float32 tensor [NUM_ROWS, HEAD_DIM], the dequantized latent
                vectors (NoPE fp8-rounded + RoPE bf16-rounded) used by the
                pure-PyTorch reference.
    """
    block_bytes = BLOCK_SIZE * TOKEN_DATA_BYTES + BLOCK_SIZE * TOKEN_SCALE_BYTES
    cache = torch.zeros(NUM_BLOCKS, block_bytes, dtype=torch.uint8)
    k_ref = torch.zeros(NUM_ROWS, HEAD_DIM, dtype=torch.float32)

    for blk in range(NUM_BLOCKS):
        for pos in range(BLOCK_SIZE):
            row = blk * BLOCK_SIZE + pos
            nope_f = 0.1 * torch.randn(NOPE_DIM, dtype=torch.float32)
            rope_f = 0.1 * torch.randn(ROPE_DIM, dtype=torch.float32)

            # NoPE: quantize to fp8 (scale factor 1.0 -> e8m0 exponent 127).
            nope_fp8 = nope_f.to(_FP8_DTYPE)
            nope_bytes = nope_fp8.view(torch.uint8)  # [448]
            # RoPE: stored raw as bfloat16 (2 bytes each).
            rope_bf16 = rope_f.to(torch.bfloat16)
            rope_bytes = rope_bf16.view(torch.uint8)  # [128]

            data_start = pos * TOKEN_DATA_BYTES
            cache[blk, data_start : data_start + NOPE_DIM] = nope_bytes
            cache[blk, data_start + NOPE_DIM : data_start + TOKEN_DATA_BYTES] = (
                rope_bytes
            )

            # Scale region: 8 e8m0 exponent bytes per token, all 127 (== 1.0).
            scale_start = BLOCK_SIZE * TOKEN_DATA_BYTES + pos * TOKEN_SCALE_BYTES
            cache[blk, scale_start : scale_start + TOKEN_SCALE_BYTES] = 127

            # Reference dequantized latent: fp8->fp32 NoPE (scale 1), bf16 RoPE.
            k_ref[row, :NOPE_DIM] = nope_fp8.to(torch.float32)
            k_ref[row, NOPE_DIM:] = rope_bf16.to(torch.float32)

    cache = cache.reshape(NUM_BLOCKS, BLOCK_SIZE, block_bytes // BLOCK_SIZE)
    return cache, k_ref


CACHE_CPU, K_REF_CPU = _build_cache()
CACHE = CACHE_CPU.to(DEVICE)
K_REF = K_REF_CPU  # kept on CPU for the reference

# q: [num_seqs, num_heads, head_dim], bf16 (matmul dtype).
Q = (0.1 * torch.randn(NUM_SEQS, NUM_HEADS, HEAD_DIM, dtype=torch.float32)).to(
    dtype=torch.bfloat16, device=DEVICE
)

# Ragged selected slots: TOPK distinct rows per query, plus indptr.
_sel = torch.stack([torch.randperm(NUM_ROWS)[:TOPK] for _ in range(NUM_SEQS)])
MAIN_INDICES = _sel.reshape(-1).to(dtype=torch.int32, device=DEVICE)
MAIN_INDPTR = torch.arange(
    0, (NUM_SEQS + 1) * TOPK, TOPK, dtype=torch.int32, device=DEVICE
)
SEL_CPU = _sel  # [num_seqs, topk] for the reference


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


def pytorch_ref(q, k_ref, sel, scale):
    """Pure PyTorch ragged sparse MLA decode attention (no sink, no extra).

    Pure PyTorch only -- no custom / Triton / vLLM / QAIC kernel calls.
    """
    q = q.float().cpu()
    num_seqs, num_heads, head_dim = q.shape
    out = torch.zeros(num_seqs, num_heads, head_dim, dtype=torch.float32)

    for s in range(num_seqs):
        slots = sel[s].long()  # [topk]
        k_sel = k_ref[slots]  # [topk, head_dim]
        k_nope = k_sel[:, :NOPE_DIM]
        k_rope = k_sel[:, NOPE_DIM:]
        for h in range(num_heads):
            q_nope = q[s, h, :NOPE_DIM]
            q_rope = q[s, h, NOPE_DIM:]
            # NoPE + RoPE contributions summed (== full-width q.k), scaled.
            score = scale * (
                q_nope @ k_nope.transpose(0, 1) + q_rope @ k_rope.transpose(0, 1)
            )  # [topk]
            m = score.max()
            p = torch.exp(score - m)
            p = p / p.sum()
            # Value == Key latent: weighted sum of full gathered vectors.
            out[s, h] = p @ k_sel
    return out


def kernel_impl(q, cache, main_indices, main_indptr, scale):
    """Kernel wrapper: launch only (no sink, no extra cache)."""
    return _rocm_sparse_attn_decode_ragged_triton(
        q=q,
        main_cache=cache,
        main_indices=main_indices,
        main_indptr=main_indptr,
        scale=scale,
        attn_sink=None,
        nope_head_dim=NOPE_DIM,
        rope_head_dim=ROPE_DIM,
        extra_cache=None,
        extra_indices=None,
        extra_indptr=None,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(Q, K_REF, SEL_CPU, SCALE)
        kernel_out = kernel_impl(Q, CACHE, MAIN_INDICES, MAIN_INDPTR, SCALE)

        kernel_cpu = kernel_out.float().cpu()

        torch.testing.assert_close(kernel_cpu, ref_out, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_out).abs()
        stats = {
            "q_shape": tuple(Q.shape),
            "cache_shape": tuple(CACHE.shape),
            "main_indices_shape": tuple(MAIN_INDICES.shape),
            "main_indptr": MAIN_INDPTR.cpu().tolist(),
            "out_shape": tuple(kernel_cpu.shape),
            "in_dtype": str(Q.dtype),
            "out_dtype": "torch.float32 (cast from bf16)",
            "fp8_dtype": str(_FP8_DTYPE),
            "device": str(Q.device),
            "scale": SCALE,
            "topk": TOPK,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_out.abs().max() + 1e-8)).item(),
            "grid": (f"({NUM_SEQS},1)", "BLOCK_H=16,BLOCK_K=16"),
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, K_REF, SEL_CPU, SCALE))
        kern_stats = _bench(
            lambda: kernel_impl(Q, CACHE, MAIN_INDICES, MAIN_INDPTR, SCALE)
        )
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
            "Kernel: _sparse_attn_decode_ragged_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Branches: HAS_ATTN_SINK=False, HAS_EXTRA=False (both skipped)\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- cache shape: {stats['cache_shape']}\n")
            lines.append(f"- main_indices shape: {stats['main_indices_shape']}\n")
            lines.append(f"- main_indptr: {stats['main_indptr']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- fp8 dtype: {stats['fp8_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- scale: {stats['scale']}\n")
            lines.append(f"- topk: {stats['topk']}\n\n")
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid'][0]}\n")
            lines.append(f"- blocks: {stats['grid'][1]}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- out shape: {stats['out_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err: {stats['rel_err']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
