"""
Standalone QAIC validation for `kernel_unified_attention`.

Source under test:
vllm/v1/attention/ops/triton_unified_attention.py
  - kernel_unified_attention  (the main unified paged-attention kernel:
    supports ALiBi, softcap, sliding window, attention sinks, quantized KV,
    and both 2D single-pass and 3D split-softmax layouts)
  - unified_attention         (public Python launcher used here)

We exercise the SIMPLEST configuration: a single sequence, causal attention,
no ALiBi / no softcap / no sliding window / no sinks / no KV quantization,
2D (single-pass) mode. For that configuration the kernel computes standard
causal scaled-dot-product attention over the paged KV layout, so we use
`torch.nn.functional.scaled_dot_product_attention(is_causal=True)` (a standard
PyTorch op, not a custom/triton/vllm kernel) as the pure-PyTorch reference.

KV cache layout fed to the launcher:
  k, v : [num_blocks, block_size, num_kv_heads, head_size]
with a single block holding the whole sequence (block_table = [[0]]).

Reference: pure PyTorch causal scaled_dot_product_attention.
"""

import datetime
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    unified_attention,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_kernel_unified_attention.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

DEVICE = "qaic"
SEQ_LEN = 32
NUM_Q_HEADS = 2
NUM_KV_HEADS = 2  # num_queries_per_kv = 1 (simplest)
HEAD_SIZE = 32
BLOCK_SIZE = SEQ_LEN  # one KV block holds the full sequence
NUM_BLOCKS = 1
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_SIZE)

torch.manual_seed(42)

# Packed query: [num_tokens, num_query_heads, head_size]
Q = torch.randn(SEQ_LEN, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
# Paged KV cache: [num_blocks, block_size, num_kv_heads, head_size]
K_CACHE = torch.randn(
    NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
)
V_CACHE = torch.randn(
    NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
)

CU_SEQLENS_Q = torch.tensor([0, SEQ_LEN], dtype=torch.int32, device=DEVICE)
SEQUSED_K = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)
BLOCK_TABLE = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)  # single block
Q_DESCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
K_DESCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
V_DESCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(q, k_cache, v_cache, scale):
    """Pure PyTorch causal SDPA over the paged KV layout.

    For a single sequence stored contiguously in block 0, the effective K/V
    are the first SEQ_LEN slots. With num_queries_per_kv == 1 no head repeat
    is needed. Computes causal scaled-dot-product attention.
    """
    q = q.float().cpu()  # [S, Hq, D]
    k = k_cache.float().cpu()[0, :SEQ_LEN]  # [S, Hk, D]
    v = v_cache.float().cpu()[0, :SEQ_LEN]  # [S, Hk, D]

    group = NUM_Q_HEADS // NUM_KV_HEADS
    if group > 1:
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

    # [S, H, D] -> [H, S, D]
    q_t = q.transpose(0, 1)
    k_t = k.transpose(0, 1)
    v_t = v.transpose(0, 1)
    o_t = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True, scale=scale)
    return o_t.transpose(0, 1)  # [S, H, D]


def kernel_impl(q, k_cache, v_cache, cu_seqlens_q, seqused_k, block_table, scale):
    """Kernel wrapper: launch only."""
    out = torch.empty_like(q)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=SEQ_LEN,
        seqused_k=seqused_k,
        max_seqlen_k=SEQ_LEN,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),  # no sliding window
        block_table=block_table,
        softcap=0.0,
        q_descale=Q_DESCALE,
        k_descale=K_DESCALE,
        v_descale=V_DESCALE,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(Q, K_CACHE, V_CACHE, SOFTMAX_SCALE)
        kernel_out = kernel_impl(
            Q, K_CACHE, V_CACHE, CU_SEQLENS_Q, SEQUSED_K, BLOCK_TABLE, SOFTMAX_SCALE
        )

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "q_shape": tuple(Q.shape),
            "k_shape": tuple(K_CACHE.shape),
            "v_shape": tuple(V_CACHE.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(Q.dtype),
            "device": str(Q.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
            "softmax_scale": SOFTMAX_SCALE,
        }
        pt_stats = _bench(lambda: pytorch_ref(Q, K_CACHE, V_CACHE, SOFTMAX_SCALE))
        kern_stats = _bench(
            lambda: kernel_impl(
                Q, K_CACHE, V_CACHE, CU_SEQLENS_Q, SEQUSED_K, BLOCK_TABLE, SOFTMAX_SCALE
            )
        )
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
            "Kernel: kernel_unified_attention (2D, causal, no alibi/softcap/swa)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- q shape: {stats['q_shape']}\n",
                f"- k cache shape: {stats['k_shape']}\n",
                f"- v cache shape: {stats['v_shape']}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n",
                f"- softmax_scale: {stats['softmax_scale']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- rel_err: {stats['rel_err']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
