"""
Standalone QAIC validation for `kernel_paged_attention_2d`.

Source under test:
vllm/v1/attention/ops/chunked_prefill_paged_decode.py
  - kernel_paged_attention_2d  (single/multi-query paged attention over a
    block-table-indexed KV cache, supporting alibi, sliding-window, attention
    sinks and FP8 cache; used for the decode portion of chunked-prefill+decode.)

SIMPLE configuration: single decode step (one query token), no GQA, float32
paged cache (no FP8), no alibi, no sliding window, no sinks. The decode query
attends to every cached position [0, seq_len). We launch via the file's public
entry `chunked_prefill_paged_decode` with max_query_len == 1, so the prefill
branch (context_attention_fwd) is skipped and only kernel_paged_attention_2d
runs. (The file's `cdiv_fn` helper is not tested here.)

Reference: pure PyTorch full attention of the decode query over the paged KV.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_kernel_paged_attention_2d.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/chunked_prefill_paged_decode.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.chunked_prefill_paged_decode import (  # noqa: E402
    chunked_prefill_paged_decode,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
NUM_TOKENS = 1          # one decode query token (single sequence)
NUM_QUERY_HEADS = 2
NUM_KV_HEADS = 2
HEAD_SIZE = 32
SEQ_LEN = 16            # cached context length for this sequence
BLOCK_SIZE = 16         # paged-cache physical block size (power of 2)
X = 8                   # K-cache packing factor (HEAD_SIZE % X == 0)
NUM_BLOCKS = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
MAX_QUERY_LEN = 1
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_SIZE)

torch.manual_seed(42)

QUERY = torch.randn(
    NUM_TOKENS, NUM_QUERY_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
)
# Logical cached K/V: [seq_len, num_kv_heads, head_size]
CACHED_K = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
CACHED_V = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)

# key_cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
# value_cache: [num_blocks, num_kv_heads, head_size, block_size]
KEY_CACHE = torch.zeros(
    NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X,
    dtype=torch.float32, device=DEVICE,
)
VALUE_CACHE = torch.zeros(
    NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE,
    dtype=torch.float32, device=DEVICE,
)

# Block table [num_seqs, max_num_blocks_per_seq] (identity mapping).
BLOCK_TABLE = torch.arange(NUM_BLOCKS, dtype=torch.int32, device=DEVICE).view(1, NUM_BLOCKS)
QUERY_START_LOC = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device=DEVICE)
SEQ_LENS = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)
K_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
V_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)


def _fill_paged_cache():
    for t in range(SEQ_LEN):
        blk = int(BLOCK_TABLE[0, t // BLOCK_SIZE].item())
        slot = t % BLOCK_SIZE
        for h in range(NUM_KV_HEADS):
            for d in range(HEAD_SIZE):
                KEY_CACHE[blk, h, d // X, slot, d % X] = CACHED_K[t, h, d]
                VALUE_CACHE[blk, h, d, slot] = CACHED_V[t, h, d]


_fill_paged_cache()


def pytorch_ref(query, cached_k, cached_v, scale):
    """Pure PyTorch full attention of the decode query over cached K/V."""
    query = query.float().cpu()
    cached_k = cached_k.float().cpu()
    cached_v = cached_v.float().cpu()

    num_tokens, num_heads, head_size = query.shape
    out = torch.zeros(num_tokens, num_heads, head_size, dtype=torch.float32)
    for h in range(num_heads):
        qh = query[:, h, :]                # [num_tokens, d]
        kh = cached_k[:, h, :]             # [seq_len, d]
        vh = cached_v[:, h, :]             # [seq_len, d]
        scores = (qh @ kh.transpose(0, 1)) * scale  # [num_tokens, seq_len]
        probs = torch.softmax(scores, dim=-1)
        out[:, h, :] = probs @ vh
    return out


def kernel_impl(query, key_cache, value_cache, block_table, query_start_loc,
                seq_lens, k_scale, v_scale, scale):
    """Kernel wrapper: launch only."""
    output = torch.empty_like(query)
    chunked_prefill_paged_decode(
        query=query,
        key=CACHED_K,          # provides num_kv_heads; unused for decode cache
        value=CACHED_V,
        output=output,
        kv_cache_dtype="auto",
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_seq_len=SEQ_LEN,
        max_query_len=MAX_QUERY_LEN,
        k_scale=k_scale,
        v_scale=v_scale,
        alibi_slopes=None,
        sliding_window=None,
        sm_scale=scale,
        output_scale=None,
        sinks=None,
        causal=True,
    )
    return output


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(QUERY, CACHED_K, CACHED_V, SOFTMAX_SCALE)
        kernel_out = kernel_impl(
            QUERY, KEY_CACHE, VALUE_CACHE, BLOCK_TABLE, QUERY_START_LOC,
            SEQ_LENS, K_SCALE, V_SCALE, SOFTMAX_SCALE,
        )

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "query_shape": tuple(QUERY.shape),
            "key_cache_shape": tuple(KEY_CACHE.shape),
            "value_cache_shape": tuple(VALUE_CACHE.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(QUERY.dtype),
            "device": str(QUERY.device),
            "seq_len": SEQ_LEN,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(QUERY, CACHED_K, CACHED_V, SOFTMAX_SCALE))
        kern_stats = _bench(lambda: kernel_impl(
            QUERY, KEY_CACHE, VALUE_CACHE, BLOCK_TABLE, QUERY_START_LOC,
            SEQ_LENS, K_SCALE, V_SCALE, SOFTMAX_SCALE))
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
            "Kernel: kernel_paged_attention_2d\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- query shape: {stats['query_shape']}\n")
            lines.append(f"- key_cache shape: {stats['key_cache_shape']}\n")
            lines.append(f"- value_cache shape: {stats['value_cache_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- seq_len: {stats['seq_len']}\n\n")
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
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
