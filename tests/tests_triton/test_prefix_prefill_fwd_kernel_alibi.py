"""
Standalone QAIC validation for `_fwd_kernel_alibi` (Triton prefix prefill+ALiBi).

Source under test:
vllm/v1/attention/ops/prefix_prefill.py
  - _fwd_kernel_alibi  (context/prefix-prefill attention identical to
    _fwd_kernel but adding an ALiBi positional bias to the attention scores.
    Attends new query tokens against a paged KV cache holding the prefix
    context AND against the new K/V tokens themselves; always causal.)

SIMPLE configuration: single sequence, no GQA, float32 paged cache (no FP8),
causal. The kernel adds, per (query token q at absolute pos ctx_len+i, key at
absolute pos p), the bias:

    alibi = alibi_slope * (p - (ctx_len + i))

which is <= 0 for causal (p <= ctx_len+i) positions and masks p > ctx_len+i to
-inf. We reproduce this exact convention with pure PyTorch matmul + softmax.

Reference: pure PyTorch attention over [cached prefix ; new K/V] with a causal
mask and the additive ALiBi bias slope*(k_pos - q_pos).
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prefix_prefill_fwd_kernel_alibi.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/prefix_prefill.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.prefix_prefill import (  # noqa: E402
    context_attention_fwd,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
NUM_HEADS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 32
QUERY_LEN = 16          # number of new tokens
CTX_LEN = 16            # number of cached prefix tokens
SEQ_LEN = CTX_LEN + QUERY_LEN
BLOCK_SIZE = 16         # paged-cache physical block size (power of 2)
X = 8                   # K-cache packing factor (HEAD_DIM % X == 0)
MAX_BLOCKS = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
NUM_BLOCKS = MAX_BLOCKS
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)

torch.manual_seed(42)

Q = torch.randn(QUERY_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
K = torch.randn(QUERY_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
V = torch.randn(QUERY_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)

CACHED_K = torch.randn(CTX_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
CACHED_V = torch.randn(CTX_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)

# k_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
# v_cache: [num_blocks, num_kv_heads, head_size, block_size]
K_CACHE = torch.zeros(
    NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM // X, BLOCK_SIZE, X,
    dtype=torch.float32, device=DEVICE,
)
V_CACHE = torch.zeros(
    NUM_BLOCKS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
    dtype=torch.float32, device=DEVICE,
)

B_LOC = torch.arange(MAX_BLOCKS, dtype=torch.int32, device=DEVICE).view(1, MAX_BLOCKS)
B_START_LOC = torch.tensor([0, QUERY_LEN], dtype=torch.int32, device=DEVICE)
B_SEQ_LEN = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)
K_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
V_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

# ALiBi slopes, one per query head (standard geometric-ish descending slopes).
ALIBI_SLOPES = torch.tensor(
    [1.0 / (2 ** (i + 1)) for i in range(NUM_HEADS)],
    dtype=torch.float32, device=DEVICE,
)


def _fill_paged_cache():
    for t in range(CTX_LEN):
        blk = int(B_LOC[0, t // BLOCK_SIZE].item())
        slot = t % BLOCK_SIZE
        for h in range(NUM_KV_HEADS):
            for d in range(HEAD_DIM):
                K_CACHE[blk, h, d // X, slot, d % X] = CACHED_K[t, h, d]
                V_CACHE[blk, h, d, slot] = CACHED_V[t, h, d]


_fill_paged_cache()


def pytorch_ref(q, cached_k, cached_v, new_k, new_v, scale, alibi_slopes):
    """Pure PyTorch reference with ALiBi bias.

    score[i, p] = scale * (q_i . k_p) + slope_h * (p - (ctx_len + i)),
    masked to p <= ctx_len + i, followed by softmax over p.
    """
    q = q.float().cpu()
    cached_k = cached_k.float().cpu()
    cached_v = cached_v.float().cpu()
    new_k = new_k.float().cpu()
    new_v = new_v.float().cpu()
    slopes = alibi_slopes.float().cpu()

    ctx_len = cached_k.shape[0]
    q_len = q.shape[0]
    num_heads = q.shape[1]

    full_k = torch.cat([cached_k, new_k], dim=0)
    full_v = torch.cat([cached_v, new_v], dim=0)
    seq_len = full_k.shape[0]

    out = torch.zeros(q_len, num_heads, q.shape[2], dtype=torch.float32)
    q_abs = (ctx_len + torch.arange(q_len)).view(q_len, 1)   # [q_len, 1]
    k_abs = torch.arange(seq_len).view(1, seq_len)           # [1, seq_len]
    mask = k_abs <= q_abs
    for h in range(num_heads):
        qh = q[:, h, :]
        kh = full_k[:, h, :]
        vh = full_v[:, h, :]
        scores = (qh @ kh.transpose(0, 1)) * scale       # [q_len, seq_len]
        alibi = float(slopes[h].item()) * (k_abs - q_abs).float()
        scores = scores + alibi
        scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[:, h, :] = probs @ vh
    return out


def kernel_impl(q, k, v, k_cache, v_cache, b_loc, b_start_loc, b_seq_len,
                k_scale, v_scale, scale, alibi_slopes):
    """Kernel wrapper: launch only (alibi path)."""
    o = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        o,
        "auto",
        k_cache,
        v_cache,
        b_loc,
        b_start_loc,
        b_seq_len,
        SEQ_LEN,
        QUERY_LEN,
        k_scale,
        v_scale,
        alibi_slopes=alibi_slopes,
        sliding_window=None,
        sm_scale=scale,
        skip_decode=False,
        fp8_out_scale=None,
        sinks=None,
        causal=True,
    )
    return o


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(
            Q, CACHED_K, CACHED_V, K, V, SOFTMAX_SCALE, ALIBI_SLOPES
        )
        kernel_out = kernel_impl(
            Q, K, V, K_CACHE, V_CACHE, B_LOC, B_START_LOC, B_SEQ_LEN,
            K_SCALE, V_SCALE, SOFTMAX_SCALE, ALIBI_SLOPES,
        )

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "q_shape": tuple(Q.shape),
            "k_cache_shape": tuple(K_CACHE.shape),
            "v_cache_shape": tuple(V_CACHE.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(Q.dtype),
            "device": str(Q.device),
            "ctx_len": CTX_LEN,
            "query_len": QUERY_LEN,
            "alibi_slopes": ALIBI_SLOPES.cpu().tolist(),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(
            lambda: pytorch_ref(
                Q, CACHED_K, CACHED_V, K, V, SOFTMAX_SCALE, ALIBI_SLOPES
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                Q, K, V, K_CACHE, V_CACHE, B_LOC, B_START_LOC, B_SEQ_LEN,
                K_SCALE, V_SCALE, SOFTMAX_SCALE, ALIBI_SLOPES,
            )
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
            "Kernel: _fwd_kernel_alibi\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- k_cache shape: {stats['k_cache_shape']}\n")
            lines.append(f"- v_cache shape: {stats['v_cache_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- ctx_len: {stats['ctx_len']}\n")
            lines.append(f"- query_len: {stats['query_len']}\n")
            lines.append(f"- alibi_slopes: {stats['alibi_slopes']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
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
