"""
Standalone QAIC validation for `_fused_inv_rope_fp8_quant_per_head`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py
  - _fused_inv_rope_fp8_quant_per_head  (@triton.jit)
  - launcher: fused_inv_rope_fp8_quant(o, positions, cos_sin_cache, n_groups,
              heads_per_group, nope_dim, rope_dim, quant_group_size,
              tma_aligned_scales=False)

Applies INVERSE (reverse) RoPE to the last `rope_dim` dims of each head's
attention output, then block-scaled FP8 (E4M3) quantization with per-quant-
group power-of-2 (UE8M0-valued) scales.

INVERSE interleaved (GPT-J adjacent-pair) RoPE — read from the kernel body:
for a rope element at head-local offset o (o >= rope_abs_start), with partner
p = o ^ 1, pair index i = (o - rope_abs_start) >> 1, cos = cache[pos, i],
sin = cache[pos, HALF_ROPE + i]:
    even local index:  x'_even = x_even * cos + x_odd  * sin
    odd  local index:  x'_odd  = x_odd  * cos - x_even * sin
This is the sign-flipped-sin (inverse) of forward RoPE. Partner uses the
ORIGINAL (unrotated) neighbor value, matching the kernel (it loads x_partner
from the input buffer).

Block-scaled FP8 quant, per QUANT_GROUP_SIZE chunk of the head:
    absmax = max(|x'|) over the chunk, clamped to eps=1e-10
    scale  = 2^ceil(log2(absmax / fp8_max))         (power-of-2 UE8M0 value)
    q      = clamp(x' / scale, -fp8_max, fp8_max) -> float8_e4m3fn

We use the UE8M0/FP32-scale path (tma_aligned_scales=False), which stores the
power-of-2 scale as fp32 (the SM100 TMA INT32-packed variant is not exercised
here). FLOAT/fp8 kernel: we compare the DEQUANTIZED values (fp8 * scale) at
rtol/atol=1e-2 and the per-block scales at rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_inv_rope_fp8_quant_per_head.txt")
KERNEL_FILE_PATH = (
    "vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py"
)

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 4
N_GROUPS = 1
HEADS_PER_GROUP = 2
NUM_HEADS = N_GROUPS * HEADS_PER_GROUP  # 2
NOPE_DIM = 16
ROPE_DIM = 16
HEAD_DIM = NOPE_DIM + ROPE_DIM          # 32
QUANT_GROUP_SIZE = 16
CHUNKS_PER_HEAD = HEAD_DIM // QUANT_GROUP_SIZE  # 2
HALF_ROPE = ROPE_DIM // 2               # 8
ROPE_START = NOPE_DIM % QUANT_GROUP_SIZE  # 0
ROPE_ABS_START = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START  # 16
D = HEADS_PER_GROUP * HEAD_DIM          # 64
NUM_SCALE_BLOCKS = D // QUANT_GROUP_SIZE  # 4
MAX_POS = 16
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# bf16 attention output so the fp32 load in the kernel is exact.
O = (torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32) * 2.0).to(
    torch.bfloat16
).to(DEVICE)
POSITIONS = torch.tensor([0, 3, 7, 12], dtype=torch.int64, device=DEVICE)
COS_SIN_CACHE = torch.randn(MAX_POS, ROPE_DIM, dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(o, positions, cos_sin_cache):
    o = o.cpu().to(torch.float32)
    positions = positions.cpu()
    cache = cos_sin_cache.cpu().to(torch.float32)

    # Output layout mirrors the kernel/launcher: fp8[t, g, hh*HEAD_DIM + ...]
    deq = torch.zeros(NUM_TOKENS, N_GROUPS, D, dtype=torch.float32)
    scales = torch.zeros(NUM_TOKENS, N_GROUPS, NUM_SCALE_BLOCKS, dtype=torch.float32)

    for t in range(NUM_TOKENS):
        pos = int(positions[t].item())
        for h in range(NUM_HEADS):
            g = h // HEADS_PER_GROUP
            hh = h % HEADS_PER_GROUP
            x = o[t, h].clone()
            x_orig = x.clone()
            # Inverse RoPE on [ROPE_ABS_START, HEAD_DIM).
            for local in range(ROPE_DIM):
                o_idx = ROPE_ABS_START + local
                p_idx = o_idx ^ 1
                i = local >> 1
                cos = float(cache[pos, i].item())
                sin = float(cache[pos, HALF_ROPE + i].item())
                xo = float(x_orig[o_idx].item())
                xp = float(x_orig[p_idx].item())
                if local % 2 == 0:
                    x[o_idx] = xo * cos + xp * sin
                else:
                    x[o_idx] = xo * cos - xp * sin
            # Per-quant-group block-scaled FP8 quant.
            for c in range(CHUNKS_PER_HEAD):
                chunk = x[c * QUANT_GROUP_SIZE:(c + 1) * QUANT_GROUP_SIZE]
                absmax = torch.clamp(chunk.abs().max(), min=1e-10)
                scale_raw = absmax / FP8_MAX
                scale = float(torch.exp2(torch.ceil(torch.log2(scale_raw))).item())
                q = torch.clamp(chunk / scale, -FP8_MAX, FP8_MAX).to(
                    torch.float8_e4m3fn
                ).to(torch.float32)
                base = hh * HEAD_DIM + c * QUANT_GROUP_SIZE
                deq[t, g, base:base + QUANT_GROUP_SIZE] = q * scale
                scales[t, g, hh * CHUNKS_PER_HEAD + c] = scale
    return deq, scales


def kernel_impl(o, positions, cos_sin_cache):
    return fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=N_GROUPS,
        heads_per_group=HEADS_PER_GROUP,
        nope_dim=NOPE_DIM,
        rope_dim=ROPE_DIM,
        quant_group_size=QUANT_GROUP_SIZE,
        tma_aligned_scales=False,
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
        ref_deq, ref_scales = pytorch_ref(O, POSITIONS, COS_SIN_CACHE)
        fp8_out, scale_out = kernel_impl(O, POSITIONS, COS_SIN_CACHE)

        # fp8_out: (num_tokens, n_groups, d); scale_out: (num_tokens, n_groups,
        # num_scale_blocks).
        fp8_cpu = fp8_out.cpu().to(torch.float32)
        scale_cpu = scale_out.cpu().to(torch.float32)

        # Dequantize kernel fp8 with kernel scales, block by block.
        k_deq = torch.zeros_like(ref_deq)
        for t in range(NUM_TOKENS):
            for g in range(N_GROUPS):
                for blk in range(NUM_SCALE_BLOCKS):
                    scl = float(scale_cpu[t, g, blk].item())
                    base = blk * QUANT_GROUP_SIZE
                    k_deq[t, g, base:base + QUANT_GROUP_SIZE] = (
                        fp8_cpu[t, g, base:base + QUANT_GROUP_SIZE] * scl
                    )

        torch.testing.assert_close(scale_cpu, ref_scales, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_deq, ref_deq, rtol=1e-2, atol=1e-2)

        deq_diff = (k_deq - ref_deq).abs()
        scl_diff = (scale_cpu - ref_scales).abs()
        stats = {
            "o_shape": tuple(O.shape),
            "fp8_shape": tuple(fp8_out.shape),
            "scale_shape": tuple(scale_out.shape),
            "device": DEVICE,
            "deq_max_abs_diff": deq_diff.max().item(),
            "deq_mean_abs_diff": deq_diff.mean().item(),
            "scale_max_abs_diff": scl_diff.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(O, POSITIONS, COS_SIN_CACHE))
        kern_stats = _bench(lambda: kernel_impl(O, POSITIONS, COS_SIN_CACHE))
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
            "Kernel: _fused_inv_rope_fp8_quant_per_head\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: UE8M0/FP32-scale path (tma_aligned_scales=False); SM100\n"
            "      INT32-packed variant not exercised here.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- o shape: {stats['o_shape']} bf16\n",
                f"- nope_dim={NOPE_DIM}, rope_dim={ROPE_DIM}, "
                f"quant_group_size={QUANT_GROUP_SIZE}\n",
                f"- n_groups={N_GROUPS}, heads_per_group={HEADS_PER_GROUP}\n",
                f"- device: {stats['device']}\n\n",
                "Output (dequant rtol/atol=1e-2, scales rtol/atol=1e-3):\n",
                f"- fp8 shape: {stats['fp8_shape']} float8_e4m3fn\n",
                f"- scale shape: {stats['scale_shape']} fp32 (power-of-2)\n",
                f"- deq max_abs_diff: {stats['deq_max_abs_diff']}\n",
                f"- deq mean_abs_diff: {stats['deq_mean_abs_diff']}\n",
                f"- scale max_abs_diff: {stats['scale_max_abs_diff']}\n",
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
