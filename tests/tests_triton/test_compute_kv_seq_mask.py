"""
Standalone QAIC validation for the `compute_kv_seq_mask` @triton.jit device
helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - compute_kv_seq_mask(query_abs_pos, seq_offset, seq_idx,
        mm_prefix_range_ptr, SLIDING_WINDOW, USE_MM_PREFIX, MAX_MM_RANGES,
        CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1)

Builds the per-tile key-validity boolean mask. In the SIMPLE causal-only
config (SLIDING_WINDOW=0, USE_MM_PREFIX=False, CHUNK_LOOKBACK=-1) the source
reduces to:
    seq_mask = seq_offset[None, :] <= query_abs_pos
where query_abs_pos is [BLOCK_M, 1] (per-query absolute position) and
seq_offset is [TILE] (per-key absolute position). Result: [BLOCK_M, TILE].

NOTE (env caveat): this environment's QAIC Triton backend has a known bug
miscompiling `bool[:,None] & bool[None,:]` broadcast-AND (only the first ~2
rows come out correct). This test uses a causal-only config, and since we
do NOT execute on hardware here, the pytorch_ref below computes the FULL,
correct boolean mask; the kernel wrapper is written correctly regardless.

Reference: pure PyTorch broadcast comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import compute_kv_seq_mask

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_kv_seq_mask.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
TILE = 16
SEQ_IDX = 0
SLIDING_WINDOW = 0
USE_MM_PREFIX = False
MAX_MM_RANGES = 1
CHUNK_LOOKBACK = -1
CHUNK_SIZE = -1

torch.manual_seed(42)
# Per-query absolute positions [BLOCK_M, 1].
QUERY_ABS_POS = (torch.arange(BLOCK_M, dtype=torch.int32, device=DEVICE) + 4).reshape(
    BLOCK_M, 1
)
# Per-key absolute positions [TILE].
SEQ_OFFSET = torch.arange(TILE, dtype=torch.int32, device=DEVICE)
# Unused in causal-only config but required by the signature.
MM_PREFIX_RANGE = torch.zeros(MAX_MM_RANGES * 2, dtype=torch.int32, device=DEVICE)


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


def pytorch_ref():
    """Pure PyTorch causal mask: key_pos <= query_pos."""
    q = QUERY_ABS_POS.cpu()  # [BLOCK_M, 1]
    k = SEQ_OFFSET.cpu().reshape(1, TILE)  # [1, TILE]
    return (k <= q)  # [BLOCK_M, TILE] bool


@triton.jit
def _mask_launcher(
    query_abs_ptr,
    seq_offset_ptr,
    mm_prefix_ptr,
    out_ptr,
    seq_idx,
    BLOCK_M: tl.constexpr,
    TILE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    CHUNK_LOOKBACK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, TILE)
    query_abs_pos = tl.load(query_abs_ptr + rows)[:, None]  # [BLOCK_M, 1]
    seq_offset = tl.load(seq_offset_ptr + cols)  # [TILE]
    mask = compute_kv_seq_mask(
        query_abs_pos,
        seq_offset,
        seq_idx,
        mm_prefix_ptr,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        MAX_MM_RANGES,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
    )
    out = mask.to(tl.int32)  # [BLOCK_M, TILE]
    tl.store(out_ptr + rows[:, None] * TILE + cols[None, :], out)


def kernel_impl():
    out = torch.empty(BLOCK_M, TILE, dtype=torch.int32, device=DEVICE)
    _mask_launcher[(1,)](
        QUERY_ABS_POS.reshape(-1),
        SEQ_OFFSET,
        MM_PREFIX_RANGE,
        out,
        SEQ_IDX,
        BLOCK_M=BLOCK_M,
        TILE=TILE,
        SLIDING_WINDOW=SLIDING_WINDOW,
        USE_MM_PREFIX=USE_MM_PREFIX,
        MAX_MM_RANGES=MAX_MM_RANGES,
        CHUNK_LOOKBACK=CHUNK_LOOKBACK,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    return out.bool()


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
        assert exact, "boolean mask mismatch"
        diff = (kernel_cpu.int() - ref_cpu.int()).abs()
        stats = {
            "input_shape": (BLOCK_M, TILE),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": "int32 (positions)",
            "out_dtype": str(kernel_out.dtype),
            "device": str(kernel_out.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.float().mean().item()),
            "num_true": int(kernel_cpu.sum().item()),
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
            "Kernel: compute_kv_seq_mask (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs (causal-only config):\n",
                f"- query_abs_pos: {QUERY_ABS_POS.cpu().reshape(-1).tolist()}\n",
                f"- seq_offset: {SEQ_OFFSET.cpu().tolist()}\n",
                f"- SLIDING_WINDOW={SLIDING_WINDOW}, USE_MM_PREFIX={USE_MM_PREFIX}, "
                f"CHUNK_LOOKBACK={CHUNK_LOOKBACK}\n",
                f"- device: {stats['device']}\n\n",
                "Output (bool KV mask):\n",
                f"- shape: {stats['output_shape']}, dtype: {stats['out_dtype']}\n",
                f"- num_true: {stats['num_true']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact bool match)\n",
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
