"""
Standalone QAIC validation for `_fused_indexer_q_rope_quant_kernel` (FP8 UE8M0).

Source under test:
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
  - _fused_indexer_q_rope_quant_kernel
  - launcher: fused_indexer_q_rope_quant(..., use_fp4=False)

Fuses GPT-J interleaved RoPE + FP8 (UE8M0, power-of-2 scale) quantization for the
sparse-indexer query, folding the per-token/per-head quant scale into the index
weights.

Per (token, head):
  - GPT-J interleaved RoPE on the LAST rope_dim dims (leading nope dims pass
    through unchanged):
        x_even = q[nope + 2i], x_odd = q[nope + 2i + 1]
        r_even = x_even*cos - x_odd*sin ; r_odd = x_odd*cos + x_even*sin
    then bf16 roundtrip (fp32 -> bf16 -> fp32) for parity.
  - amax over |nope|, |r_even|, |r_odd|
    scale = 2^ceil(log2(max(amax, 1e-4) / 448.0))          (UE8M0, power of 2)
    q_fp8 = round(value / scale) as float8_e4m3fn
  - weights_out = weights * scale * softmax_scale * head_scale   (q_scale folded)

FLOAT/quant kernel. Comparison choice (NOT executing on device, so fp8 is
represented numerically): we recompute the UE8M0 scale in the reference, then
compare (a) the DEQUANTIZED q values  q_fp8.float() * scale  and (b) the folded
weights_out, both with rtol/atol = 1e-3. Because the kernel and reference use the
identical power-of-2 scale and the same bf16-roundtrip RoPE math, the fp8 codes
coincide and the dequantized tensors match tightly.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    fused_indexer_q_rope_quant,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_indexer_q_rope_quant_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 3
NUM_HEADS = 2
HALF_ROT_DIM = 4
ROT_DIM = 2 * HALF_ROT_DIM  # 8
NOPE_DIM = 8
HEAD_DIM = NOPE_DIM + ROT_DIM  # 16
FP8_MAX = 448.0
SOFTMAX_SCALE = 0.5
HEAD_SCALE = 1.25

POSITIONS = torch.tensor([0, 3, 7], dtype=torch.int64, device=DEVICE)
INDEX_Q = torch.randn(
    NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE
)
# cos_sin_cache: [max_pos, ROT_DIM] = [cos(HALF) | sin(HALF)]
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


def pytorch_ref(positions, index_q, cos_sin_cache, index_weights):
    positions = positions.cpu()
    q = index_q.cpu().to(torch.float32)
    cache = cos_sin_cache.cpu().to(torch.float32)
    weights = index_weights.cpu().to(torch.float32)

    q_fp8 = torch.empty(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    scale = torch.empty(NUM_TOKENS, NUM_HEADS, dtype=torch.float32)
    w_out = torch.empty(NUM_TOKENS, NUM_HEADS, dtype=torch.float32)

    for t in range(NUM_TOKENS):
        pos = int(positions[t].item())
        cos = cache[pos, :HALF_ROT_DIM]
        sin = cache[pos, HALF_ROT_DIM:ROT_DIM]
        for h in range(NUM_HEADS):
            row = q[t, h]
            nope = row[:NOPE_DIM]
            rot = row[NOPE_DIM:]
            x_even = rot[0::2]
            x_odd = rot[1::2]
            r_even = x_even * cos - x_odd * sin
            r_odd = x_odd * cos + x_even * sin
            # bf16 roundtrip parity
            r_even = r_even.to(torch.bfloat16).to(torch.float32)
            r_odd = r_odd.to(torch.bfloat16).to(torch.float32)

            amax = torch.maximum(r_even.abs().max(), r_odd.abs().max())
            if NOPE_DIM > 0:
                amax = torch.maximum(amax, nope.abs().max())
            raw = torch.clamp(amax, min=1e-4) / FP8_MAX
            s = torch.exp2(torch.ceil(torch.log2(raw)))
            scale[t, h] = s

            # quantize
            nope_q = (nope / s).to(torch.float8_e4m3fn).to(torch.float32)
            re_q = (r_even / s).to(torch.float8_e4m3fn).to(torch.float32)
            ro_q = (r_odd / s).to(torch.float8_e4m3fn).to(torch.float32)
            q_fp8[t, h, :NOPE_DIM] = nope_q
            q_fp8[t, h, NOPE_DIM + torch.arange(HALF_ROT_DIM) * 2] = re_q
            q_fp8[t, h, NOPE_DIM + torch.arange(HALF_ROT_DIM) * 2 + 1] = ro_q

            w_out[t, h] = float(weights[t, h].item()) * float(s.item()) * (
                SOFTMAX_SCALE * HEAD_SCALE
            )
    return q_fp8, scale, w_out


def kernel_impl(positions, index_q, cos_sin_cache, index_weights):
    q_fp8, w_out = fused_indexer_q_rope_quant(
        positions,
        index_q,
        cos_sin_cache,
        index_weights,
        SOFTMAX_SCALE,
        HEAD_SCALE,
        use_fp4=False,
    )
    return q_fp8, w_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_q_fp8, ref_scale, ref_w = pytorch_ref(
            POSITIONS, INDEX_Q, COS_SIN_CACHE, INDEX_WEIGHTS
        )
        k_q_fp8, k_w = kernel_impl(POSITIONS, INDEX_Q, COS_SIN_CACHE, INDEX_WEIGHTS)
        k_q_fp8 = k_q_fp8.cpu().to(torch.float32)
        k_w = k_w.cpu().to(torch.float32)

        # Dequantize both using the reference UE8M0 scale, then compare.
        deq_ref = ref_q_fp8 * ref_scale.unsqueeze(-1)
        deq_k = k_q_fp8 * ref_scale.unsqueeze(-1)

        torch.testing.assert_close(deq_k, deq_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_w, ref_w, rtol=1e-3, atol=1e-3)

        q_diff = (deq_k - deq_ref).abs()
        w_diff = (k_w - ref_w).abs()
        stats = {
            "input_shape": tuple(INDEX_Q.shape),
            "q_fp8_shape": tuple(k_q_fp8.shape),
            "weights_shape": tuple(k_w.shape),
            "in_dtype": str(INDEX_Q.dtype),
            "q_fp8_dtype": "torch.float8_e4m3fn",
            "device": DEVICE,
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
            "Kernel: _fused_indexer_q_rope_quant_kernel (FP8 UE8M0)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- index_q shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- head_dim={HEAD_DIM} (nope={NOPE_DIM}, rot={ROT_DIM}), "
                f"softmax_scale={SOFTMAX_SCALE}, head_scale={HEAD_SCALE}\n",
                f"- device: {stats['device']}\n\n",
                "Output (dequantized-value + folded-weight comparison, rtol/atol=1e-3):\n",
                f"- q_fp8 shape: {stats['q_fp8_shape']} dtype {stats['q_fp8_dtype']}\n",
                f"- weights_out shape: {stats['weights_shape']}\n",
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
