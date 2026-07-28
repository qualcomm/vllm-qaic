"""
Standalone QAIC validation for `_xpu_qnorm_rope_kernel`.

Source under test:
vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py
  - _xpu_qnorm_rope_kernel

This is the only genuinely float/numeric kernel in this batch. It applies:
  * Q: per-head RMSNorm (no weight) over the full HEAD_DIM, then GPT-J
       interleaved RoPE on the last ROPE_DIM dims.
  * KV: GPT-J interleaved RoPE on the last ROPE_DIM dims (no norm), written to
       kv_out.

We launch ONLY the Triton kernel (grid=(num_tokens, num_heads+1)), replicated
from the launch site in xpu_qnorm_rope_kv_fp8_insert(); the FP8 quant+insert
tail of that wrapper is intentionally skipped (out of scope for this kernel).

RoPE convention (matching source exactly -- GPT-J interleaved):
  For pair i in [0, HALF_ROPE):
    even index = NOPE_DIM + 2*i,  odd index = NOPE_DIM + 2*i + 1
    cos_i = cos_sin_cache[pos, i]           (first HALF_ROPE cols = cos)
    sin_i = cos_sin_cache[pos, HALF_ROPE+i] (second HALF_ROPE cols = sin)
    new_even = even*cos_i - odd*sin_i
    new_odd  = even*sin_i + odd*cos_i
  Q's rope inputs are the RMS-normalized values (even/odd * rms).

dtype float32; validated with rtol/atol = 1e-3.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    HALF_ROPE,
    HEAD_DIM,
    NOPE_DIM,
    ROPE_DIM,
    _xpu_qnorm_rope_kernel,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_xpu_qnorm_rope_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py"
KERNEL_NAME = "_xpu_qnorm_rope_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
NUM_TOKENS = 3
NUM_HEADS = 2
EPS = 1e-6
MAX_POS = 16

Q = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
KV = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
POSITIONS = torch.tensor([1, 4, 9], dtype=torch.int64, device=DEVICE)
# cos_sin_cache: [MAX_POS, ROPE_DIM]; first HALF_ROPE cols cos, second sin.
_angles = torch.randn(MAX_POS, HALF_ROPE, dtype=torch.float32, device=DEVICE)
COS_SIN_CACHE = torch.cat([torch.cos(_angles), torch.sin(_angles)], dim=1).contiguous()


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


def pytorch_ref(q, kv, positions, cos_sin_cache):
    q = q.cpu().clone().to(torch.float32)
    kv = kv.cpu().clone().to(torch.float32)
    positions = positions.cpu()
    cos_sin_cache = cos_sin_cache.cpu().to(torch.float32)

    q_out = q.clone()
    kv_out = kv.clone()

    for t in range(NUM_TOKENS):
        pos = int(positions[t].item())
        cos = cos_sin_cache[pos, :HALF_ROPE]          # [HALF_ROPE]
        sin = cos_sin_cache[pos, HALF_ROPE:ROPE_DIM]  # [HALF_ROPE]

        # ----- Q: per-head RMSNorm + GPT-J RoPE -----
        for h in range(NUM_HEADS):
            vec = q[t, h]
            sq_sum = torch.sum(vec * vec)
            rms = torch.rsqrt(sq_sum / HEAD_DIM + EPS)
            normed = vec * rms
            q_out[t, h, :NOPE_DIM] = normed[:NOPE_DIM]
            even = normed[NOPE_DIM + torch.arange(HALF_ROPE) * 2]
            odd = normed[NOPE_DIM + torch.arange(HALF_ROPE) * 2 + 1]
            new_even = even * cos - odd * sin
            new_odd = even * sin + odd * cos
            q_out[t, h, NOPE_DIM + torch.arange(HALF_ROPE) * 2] = new_even
            q_out[t, h, NOPE_DIM + torch.arange(HALF_ROPE) * 2 + 1] = new_odd

        # ----- KV: GPT-J RoPE only (full copy then rotate rope dims) -----
        kv_out[t] = kv[t].clone()
        even = kv[t, NOPE_DIM + torch.arange(HALF_ROPE) * 2]
        odd = kv[t, NOPE_DIM + torch.arange(HALF_ROPE) * 2 + 1]
        new_even = even * cos - odd * sin
        new_odd = even * sin + odd * cos
        kv_out[t, NOPE_DIM + torch.arange(HALF_ROPE) * 2] = new_even
        kv_out[t, NOPE_DIM + torch.arange(HALF_ROPE) * 2 + 1] = new_odd

    return q_out, kv_out


def kernel_impl(q, kv, positions, cos_sin_cache):
    q = q.clone()  # kernel mutates q in place
    kv_out = torch.empty_like(kv)
    grid = (NUM_TOKENS, NUM_HEADS + 1)
    _xpu_qnorm_rope_kernel[grid](
        q,
        kv,
        kv_out,
        positions,
        cos_sin_cache,
        EPS,
        NUM_TOKENS,
        num_heads=NUM_HEADS,
        HEAD_DIM=HEAD_DIM,
        ROPE_DIM=ROPE_DIM,
        NOPE_DIM=NOPE_DIM,
        HALF_ROPE=HALF_ROPE,
    )
    return q, kv_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_q, ref_kv = pytorch_ref(Q, KV, POSITIONS, COS_SIN_CACHE)
        ker_q, ker_kv = kernel_impl(Q, KV, POSITIONS, COS_SIN_CACHE)

        ref_q, ref_kv = ref_q.cpu(), ref_kv.cpu()
        ker_q, ker_kv = ker_q.cpu(), ker_kv.cpu()

        torch.testing.assert_close(ker_q, ref_q, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_kv, ref_kv, rtol=1e-3, atol=1e-3)

        q_diff = (ker_q - ref_q).abs()
        kv_diff = (ker_kv - ref_kv).abs()
        stats = {
            "num_tokens": NUM_TOKENS,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "rope_dim": ROPE_DIM,
            "nope_dim": NOPE_DIM,
            "eps": EPS,
            "q_shape": tuple(Q.shape),
            "kv_shape": tuple(KV.shape),
            "dtype": str(Q.dtype),
            "device": str(Q.device),
            "q_max_abs_diff": q_diff.max().item(),
            "q_mean_abs_diff": q_diff.mean().item(),
            "kv_max_abs_diff": kv_diff.max().item(),
            "kv_mean_abs_diff": kv_diff.mean().item(),
            "grid": f"({NUM_TOKENS}, {NUM_HEADS + 1})",
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, KV, POSITIONS, COS_SIN_CACHE))
        kern_stats = _bench(lambda: kernel_impl(Q, KV, POSITIONS, COS_SIN_CACHE))
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
            f"Kernel: {KERNEL_NAME}\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            _timing_keys = {
                "pytorch_latency_ms",
                "kernel_latency_ms",
                "speedup_kernel_over_pytorch",
            }
            for k, v in stats.items():
                if k in _timing_keys:
                    continue
                lines.append(f"- {k}: {v}\n")
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
    sys.exit(0 if main() == "SUCCESS" else 1)
