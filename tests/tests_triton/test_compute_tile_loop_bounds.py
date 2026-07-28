"""
Standalone QAIC validation for the `compute_tile_loop_bounds` @triton.jit
device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - compute_tile_loop_bounds(context_len, seq_len, cur_batch_query_len,
        q_block_local_idx, segm_idx_or_0, tiles_per_segment_or_0,
        TILE_SIZE, BLOCK_M, BLOCK_Q, num_queries_per_kv, SLIDING_WINDOW,
        USE_MM_PREFIX, IS_3D, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1)

Computes the KV-tile loop bounds (loop_lo, loop_hi) and max_seq_prefix_len.
We validate the SIMPLE causal-only config:
    SLIDING_WINDOW = 0, USE_MM_PREFIX = False, IS_3D = False.

Under that config the source reduces to:
    max_seq_prefix_len = context_len + q_block_local_idx*BLOCK_Q
                         + (BLOCK_M - 1)//num_queries_per_kv + 1
    max_seq_prefix_len = min(max_seq_prefix_len, seq_len)   # causal
    num_tiles = cdiv(max_seq_prefix_len, TILE_SIZE)
    tile_start, tile_end = 0, num_tiles
    loop_lo, loop_hi = 0, num_tiles
    return (loop_lo, loop_hi, max_seq_prefix_len)

Returns three integers -> EXACT integer comparison.

Reference: pure PyTorch/python replication of the causal-only branch.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import compute_tile_loop_bounds

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_tile_loop_bounds.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
# Config (causal only).
CONTEXT_LEN = 32
SEQ_LEN = 64
CUR_BATCH_QUERY_LEN = 20
Q_BLOCK_LOCAL_IDX = 2
TILE_SIZE = 16
BLOCK_M = 16
BLOCK_Q = 4
NUM_QUERIES_PER_KV = 4
SLIDING_WINDOW = 0
USE_MM_PREFIX = False
IS_3D = False

torch.manual_seed(42)


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


def _cdiv(x, y):
    return (x + y - 1) // y


def pytorch_ref():
    """Pure python replication (causal only, no SWA, not 3D)."""
    max_seq_prefix_len = (
        CONTEXT_LEN
        + Q_BLOCK_LOCAL_IDX * BLOCK_Q
        + (BLOCK_M - 1) // NUM_QUERIES_PER_KV
        + 1
    )
    # causal branch: min with seq_len
    max_seq_prefix_len = min(max_seq_prefix_len, SEQ_LEN)
    num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)
    loop_lo, loop_hi = 0, num_tiles
    return torch.tensor([loop_lo, loop_hi, max_seq_prefix_len], dtype=torch.int32)


@triton.jit
def _bounds_launcher(
    out_ptr,
    context_len,
    seq_len,
    cur_batch_query_len,
    q_block_local_idx,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    IS_3D: tl.constexpr,
):
    loop_lo, loop_hi, max_prefix = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        0,  # segm_idx_or_0
        0,  # tiles_per_segment_or_0
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
    )
    tl.store(out_ptr + 0, loop_lo)
    tl.store(out_ptr + 1, loop_hi)
    tl.store(out_ptr + 2, max_prefix)


def kernel_impl():
    out = torch.empty(3, dtype=torch.int32, device=DEVICE)
    _bounds_launcher[(1,)](
        out,
        CONTEXT_LEN,
        SEQ_LEN,
        CUR_BATCH_QUERY_LEN,
        Q_BLOCK_LOCAL_IDX,
        TILE_SIZE=TILE_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_Q=BLOCK_Q,
        num_queries_per_kv=NUM_QUERIES_PER_KV,
        SLIDING_WINDOW=SLIDING_WINDOW,
        USE_MM_PREFIX=USE_MM_PREFIX,
        IS_3D=IS_3D,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()
        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        exact = bool(torch.equal(ref_cpu, kernel_cpu))
        assert exact, f"ref={ref_cpu.tolist()} kernel={kernel_cpu.tolist()}"
        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": "(scalars)",
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": "int32",
            "out_dtype": str(kernel_out.dtype),
            "device": str(kernel_out.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.float().mean().item()),
            "bounds": kernel_cpu.tolist(),
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
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
            "Kernel: compute_tile_loop_bounds (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs (causal-only config):\n",
                f"- context_len={CONTEXT_LEN}, seq_len={SEQ_LEN}, "
                f"cur_batch_query_len={CUR_BATCH_QUERY_LEN}\n",
                f"- q_block_local_idx={Q_BLOCK_LOCAL_IDX}, TILE_SIZE={TILE_SIZE}, "
                f"BLOCK_M={BLOCK_M}, BLOCK_Q={BLOCK_Q}\n",
                f"- num_queries_per_kv={NUM_QUERIES_PER_KV}, "
                f"SLIDING_WINDOW={SLIDING_WINDOW}, USE_MM_PREFIX={USE_MM_PREFIX}, "
                f"IS_3D={IS_3D}\n",
                f"- device: {stats['device']}\n\n",
                "Output (loop_lo, loop_hi, max_seq_prefix_len):\n",
                f"- bounds: {stats['bounds']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact-match comparison)\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
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
