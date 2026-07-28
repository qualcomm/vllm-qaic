"""
Standalone QAIC validation for `_fused_indexer_q_rope_mxfp4_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
  - _fused_indexer_q_rope_mxfp4_kernel
  - helpers _quantize_mxfp4_pair / _fp32x2_to_fp4x2
  - launcher: fused_indexer_q_rope_quant(..., use_fp4=True)

MXFP4 variant of the fused indexer-Q RoPE+quant: applies GPT-J interleaved RoPE
(to the RoPE blocks) then quantizes each MXFP4_BLOCK_SIZE(=32)-element block to
packed MXFP4 (E2M1, 4-bit float, 2 nibbles/byte) with a per-block UE8M0 scale.

Per (token, head), block-wise (block = 32 elements, laid out as 16 low/high
pairs -> low nibble = even-index value, high nibble = odd-index value):
  amax  = max(|x_lo|, |x_hi|) over the block, clamped to >= 6*2^-126
  log2r = clamp(ceil(log2(amax/6.0)), -127, 127)
  scale = 2^log2r ;  ue8m0 = log2r + 127
  nibble = E2M1_round_to_nearest_even(value / scale)   (satfinite clamp to +/-6)
RoPE blocks first apply interleaved RoPE (with a bf16 roundtrip) to the pair.
weights_out = weights * softmax_scale * head_scale   (q_scale NOT folded here).

E2M1 grid magnitudes: {0, .5, 1, 1.5, 2, 3, 4, 6}.

FLOAT/quant kernel. Comparison choice (NOT executing on device): the reference
recomputes the per-block UE8M0 scale exactly (deterministic power-of-2) and the
E2M1 nibbles. We compare (a) the per-block UE8M0 scale bytes EXACTLY, and (b) the
DEQUANTIZED Q values (unpack kernel nibbles -> E2M1 magnitude * per-block scale)
with rtol/atol = 1e-3, plus the folded weights_out. At E2M1 grid tie-points the
hardware round-to-nearest-even could in principle pick the adjacent code; the
reference implements round-to-nearest-even to match `cvt.rn.satfinite`.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
    fused_indexer_q_rope_quant,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_indexer_q_rope_mxfp4_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 3
NUM_HEADS = 2
BLOCK = MXFP4_BLOCK_SIZE  # 32
HALF_ROT_DIM = 16          # one MXFP4 block worth of cos/sin pairs
ROT_DIM = 2 * HALF_ROT_DIM  # 32
NOPE_DIM = 32               # one nope block
HEAD_DIM = NOPE_DIM + ROT_DIM  # 64
SOFTMAX_SCALE = 0.5
HEAD_SCALE = 1.25

# E2M1 grid magnitudes (code 0..7 -> magnitude); sign bit is code|8.
_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

POSITIONS = torch.tensor([0, 3, 7], dtype=torch.int64, device=DEVICE)
INDEX_Q = torch.randn(
    NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE
)
MAX_POS = 16
COS_SIN_CACHE = torch.randn(
    MAX_POS, ROT_DIM, dtype=torch.float32, device=DEVICE
)
INDEX_WEIGHTS = torch.randn(
    NUM_TOKENS, NUM_HEADS, dtype=torch.float32, device=DEVICE
)


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


def _e2m1_encode_rne(v: float) -> int:
    """Round a normalized fp32 value to an E2M1 nibble (round-to-nearest-even,
    satfinite clamp). Returns a 4-bit code (0..15)."""
    sign = 8 if v < 0 else 0
    a = abs(v)
    if a >= 6.0:
        return sign | 7
    # find nearest grid magnitude
    best_code = 0
    best_dist = abs(a - _E2M1_MAG[0])
    for code in range(1, 8):
        d = abs(a - _E2M1_MAG[code])
        if d < best_dist - 1e-12:
            best_dist = d
            best_code = code
        elif abs(d - best_dist) <= 1e-12:
            # tie -> round to even (even code)
            if (code % 2 == 0) and (best_code % 2 == 1):
                best_code = code
                best_dist = d
    return sign | best_code


def _e2m1_decode(code: int) -> float:
    mag = _E2M1_MAG[code & 7]
    return -mag if (code & 8) else mag


def _quant_block_pair(x_lo, x_hi):
    """Given two length-HALF_BLOCK fp32 tensors (even/odd values of a block),
    return (packed_bytes[HALF], ue8m0_byte, deq_lo, deq_hi)."""
    amax = torch.maximum(x_lo.abs().max(), x_hi.abs().max())
    amax = torch.maximum(amax, torch.tensor(6.0 * (2.0 ** -126)))
    log2r = torch.ceil(torch.log2(amax * (1.0 / 6.0)))
    log2r = torch.clamp(log2r, -127.0, 127.0)
    scale = torch.exp2(log2r)
    ue8m0 = int((log2r + 127.0).item()) & 0xFF
    inv = 1.0 / float(scale.item())

    half = x_lo.shape[0]
    packed = torch.empty(half, dtype=torch.uint8)
    deq_lo = torch.empty(half, dtype=torch.float32)
    deq_hi = torch.empty(half, dtype=torch.float32)
    for i in range(half):
        lo_code = _e2m1_encode_rne(float(x_lo[i].item()) * inv)
        hi_code = _e2m1_encode_rne(float(x_hi[i].item()) * inv)
        packed[i] = (lo_code & 0xF) | ((hi_code & 0xF) << 4)
        deq_lo[i] = _e2m1_decode(lo_code) * float(scale.item())
        deq_hi[i] = _e2m1_decode(hi_code) * float(scale.item())
    return packed, ue8m0, deq_lo, deq_hi


def pytorch_ref(positions, index_q, cos_sin_cache, index_weights):
    positions = positions.cpu()
    q = index_q.cpu().to(torch.float32)
    cache = cos_sin_cache.cpu().to(torch.float32)
    weights = index_weights.cpu().to(torch.float32)

    num_scale_blocks = HEAD_DIM // BLOCK
    half_block = BLOCK // 2
    num_nope_blocks = NOPE_DIM // BLOCK
    num_rope_blocks = ROT_DIM // BLOCK

    packed = torch.empty(
        NUM_TOKENS, NUM_HEADS, HEAD_DIM // 2, dtype=torch.uint8
    )
    scales = torch.empty(
        NUM_TOKENS, NUM_HEADS, num_scale_blocks, dtype=torch.uint8
    )
    # dequantized full head-dim values (interleaved back)
    deq = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    w_out = torch.empty(NUM_TOKENS, NUM_HEADS, dtype=torch.float32)

    for t in range(NUM_TOKENS):
        pos = int(positions[t].item())
        for h in range(NUM_HEADS):
            row = q[t, h]
            # NoPE blocks
            for b in range(num_nope_blocks):
                base = b * BLOCK
                x_lo = row[base + torch.arange(half_block) * 2]
                x_hi = row[base + torch.arange(half_block) * 2 + 1]
                pk, ue8, dlo, dhi = _quant_block_pair(x_lo, x_hi)
                packed[t, h, base // 2 : base // 2 + half_block] = pk
                scales[t, h, b] = ue8
                deq[t, h, base + torch.arange(half_block) * 2] = dlo
                deq[t, h, base + torch.arange(half_block) * 2 + 1] = dhi
            # RoPE blocks
            rot = row[NOPE_DIM:]
            for b in range(num_rope_blocks):
                pair_off = b * half_block + torch.arange(half_block)
                cos_b = cache[pos, pair_off]
                sin_b = cache[pos, pair_off + HALF_ROT_DIM]
                x_even = rot[pair_off * 2]
                x_odd = rot[pair_off * 2 + 1]
                r_even = x_even * cos_b - x_odd * sin_b
                r_odd = x_odd * cos_b + x_even * sin_b
                r_even = r_even.to(torch.bfloat16).to(torch.float32)
                r_odd = r_odd.to(torch.bfloat16).to(torch.float32)
                pk, ue8, dlo, dhi = _quant_block_pair(r_even, r_odd)
                rope_byte_off = (NOPE_DIM + b * BLOCK) // 2
                packed[t, h, rope_byte_off : rope_byte_off + half_block] = pk
                scales[t, h, num_nope_blocks + b] = ue8
                base = NOPE_DIM + b * BLOCK
                deq[t, h, base + torch.arange(half_block) * 2] = dlo
                deq[t, h, base + torch.arange(half_block) * 2 + 1] = dhi

            w_out[t, h] = float(weights[t, h].item()) * (SOFTMAX_SCALE * HEAD_SCALE)
    return packed, scales, deq, w_out


def _dequant_kernel_output(packed, scales):
    """Unpack kernel nibble bytes + per-block UE8M0 scales -> fp32 head-dim."""
    packed = packed.cpu()
    scales = scales.cpu()
    half_block = BLOCK // 2
    num_nope_blocks = NOPE_DIM // BLOCK
    num_rope_blocks = ROT_DIM // BLOCK
    deq = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    for t in range(NUM_TOKENS):
        for h in range(NUM_HEADS):
            # NoPE blocks
            for b in range(num_nope_blocks):
                base = b * BLOCK
                scl = 2.0 ** (int(scales[t, h, b].item()) - 127)
                for i in range(half_block):
                    byte = int(packed[t, h, base // 2 + i].item())
                    deq[t, h, base + 2 * i] = _e2m1_decode(byte & 0xF) * scl
                    deq[t, h, base + 2 * i + 1] = _e2m1_decode((byte >> 4) & 0xF) * scl
            # RoPE blocks
            for b in range(num_rope_blocks):
                base = NOPE_DIM + b * BLOCK
                scl = 2.0 ** (int(scales[t, h, num_nope_blocks + b].item()) - 127)
                rope_byte_off = base // 2
                for i in range(half_block):
                    byte = int(packed[t, h, rope_byte_off + i].item())
                    deq[t, h, base + 2 * i] = _e2m1_decode(byte & 0xF) * scl
                    deq[t, h, base + 2 * i + 1] = _e2m1_decode((byte >> 4) & 0xF) * scl
    return deq


def kernel_impl(positions, index_q, cos_sin_cache, index_weights):
    (packed, scale_int32), w_out = fused_indexer_q_rope_quant(
        positions,
        index_q,
        cos_sin_cache,
        index_weights,
        SOFTMAX_SCALE,
        HEAD_SCALE,
        use_fp4=True,
    )
    # scale_int32 has shape (T, H) where each int32 packs HEAD_DIM//BLOCK ue8m0
    # bytes. Reinterpret back to per-block uint8 for dequant.
    num_scale_blocks = HEAD_DIM // BLOCK
    scale_bytes = scale_int32.cpu().contiguous().view(torch.uint8).reshape(
        NUM_TOKENS, NUM_HEADS, num_scale_blocks
    )
    return packed, scale_bytes, w_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_packed, ref_scales, ref_deq, ref_w = pytorch_ref(
            POSITIONS, INDEX_Q, COS_SIN_CACHE, INDEX_WEIGHTS
        )
        k_packed, k_scales, k_w = kernel_impl(
            POSITIONS, INDEX_Q, COS_SIN_CACHE, INDEX_WEIGHTS
        )
        k_scales = k_scales.cpu()
        k_w = k_w.cpu().to(torch.float32)

        # (a) per-block UE8M0 scale bytes EXACT.
        scale_mm = int((k_scales != ref_scales).sum().item())
        assert scale_mm == 0, f"UE8M0 scale byte mismatch={scale_mm}"

        # (b) dequantized values.
        k_deq = _dequant_kernel_output(k_packed, k_scales)
        torch.testing.assert_close(k_deq, ref_deq, rtol=1e-3, atol=1e-3)

        # (c) folded weights.
        torch.testing.assert_close(k_w, ref_w, rtol=1e-3, atol=1e-3)

        q_diff = (k_deq - ref_deq).abs()
        w_diff = (k_w - ref_w).abs()
        stats = {
            "input_shape": tuple(INDEX_Q.shape),
            "packed_shape": tuple(k_packed.shape),
            "scales_shape": tuple(k_scales.shape),
            "weights_shape": tuple(k_w.shape),
            "in_dtype": str(INDEX_Q.dtype),
            "device": DEVICE,
            "scale_byte_mismatch": scale_mm,
            "q_max_abs_diff": q_diff.max().item(),
            "q_mean_abs_diff": q_diff.mean().item(),
            "w_max_abs_diff": w_diff.max().item(),
            "w_mean_abs_diff": w_diff.mean().item(),
        }
        pt_stats = _bench(
            lambda: pytorch_ref(
                POSITIONS, INDEX_Q, COS_SIN_CACHE, INDEX_WEIGHTS
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(POSITIONS, INDEX_Q, COS_SIN_CACHE, INDEX_WEIGHTS)
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
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _fused_indexer_q_rope_mxfp4_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- index_q shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- head_dim={HEAD_DIM} (nope={NOPE_DIM}, rot={ROT_DIM}), block={BLOCK}\n",
                f"- softmax_scale={SOFTMAX_SCALE}, head_scale={HEAD_SCALE}\n",
                f"- device: {stats['device']}\n\n",
                "Output (UE8M0 scales EXACT + dequant/weights rtol/atol=1e-3):\n",
                f"- packed shape: {stats['packed_shape']} (uint8, 2 E2M1 nibbles/byte)\n",
                f"- scales shape: {stats['scales_shape']} (uint8 UE8M0)\n",
                f"- UE8M0 scale byte mismatches: {stats['scale_byte_mismatch']}\n",
                f"- q dequant max_abs_diff: {stats['q_max_abs_diff']}\n",
                f"- q dequant mean_abs_diff: {stats['q_mean_abs_diff']}\n",
                f"- weights max_abs_diff: {stats['w_max_abs_diff']}\n",
                f"- weights mean_abs_diff: {stats['w_mean_abs_diff']}\n",
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
