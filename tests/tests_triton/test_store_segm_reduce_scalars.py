"""
Standalone QAIC validation for the `store_segm_reduce_scalars` @triton.jit
device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - store_segm_reduce_scalars(segm_max_ptr, segm_expsum_ptr,
        query_offset_0, query_offset_1, segm_idx, M, L,
        query_mask_0, query_mask_1, num_query_heads, NUM_SEGMENTS_PER_SEQ)

Stores per-segment running softmax max (M) and expsum (L) into flat buffers
for `reduce_segments` to combine later. Exact source addressing:
    segm_offset = query_offset_0*(num_query_heads*NUM_SEGMENTS_PER_SEQ)
                  + query_offset_1*NUM_SEGMENTS_PER_SEQ + segm_idx
    store(segm_max_ptr    + segm_offset, M, mask=query_mask_0 & query_mask_1)
    store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)

Here query_offset_0 is a scalar token index, query_offset_1 is a per-row
[BLOCK_M] head-index vector; M, L are [BLOCK_M]. Masked-out rows are left at
their buffer init value.

Reference: pure PyTorch scatter into equivalently-sized buffers.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import store_segm_reduce_scalars

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_store_segm_reduce_scalars.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
NUM_QUERY_HEADS = 8
NUM_SEGMENTS_PER_SEQ = 4
NUM_TOKENS = 4
QUERY_OFFSET_0 = 1  # token index
SEGM_IDX = 2

torch.manual_seed(42)
# Per-row head offsets and softmax scalars.
QUERY_OFFSET_1 = torch.arange(BLOCK_M, dtype=torch.int32, device=DEVICE)
M_VALS = torch.randn(BLOCK_M, dtype=torch.float32, device=DEVICE)
L_VALS = torch.rand(BLOCK_M, dtype=torch.float32, device=DEVICE) + 0.5
# query_mask_0 scalar (token valid), query_mask_1 per-row (head valid).
QUERY_MASK_0 = 1
QUERY_MASK_1 = torch.ones(BLOCK_M, dtype=torch.int32, device=DEVICE)
QUERY_MASK_1[-2:] = 0  # last two rows masked out

BUF_SIZE = NUM_TOKENS * NUM_QUERY_HEADS * NUM_SEGMENTS_PER_SEQ
INIT_VAL = -123.0


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
    """Pure PyTorch scatter matching store_segm_reduce_scalars."""
    segm_max = torch.full((BUF_SIZE,), INIT_VAL, dtype=torch.float32)
    segm_exp = torch.full((BUF_SIZE,), INIT_VAL, dtype=torch.float32)
    offset1 = QUERY_OFFSET_1.cpu().long()
    mask = (QUERY_MASK_0 != 0) & (QUERY_MASK_1.cpu().bool())
    m = M_VALS.cpu()
    l = L_VALS.cpu()
    base = QUERY_OFFSET_0 * (NUM_QUERY_HEADS * NUM_SEGMENTS_PER_SEQ)
    for r in range(BLOCK_M):
        if not bool(mask[r]):
            continue
        off = base + int(offset1[r].item()) * NUM_SEGMENTS_PER_SEQ + SEGM_IDX
        segm_max[off] = m[r]
        segm_exp[off] = l[r]
    return segm_max, segm_exp


@triton.jit
def _store_launcher(
    segm_max_ptr,
    segm_exp_ptr,
    offset1_ptr,
    m_ptr,
    l_ptr,
    mask1_ptr,
    query_offset_0,
    query_mask_0,
    segm_idx,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    query_offset_1 = tl.load(offset1_ptr + rows)
    M = tl.load(m_ptr + rows)
    L = tl.load(l_ptr + rows)
    query_mask_1 = tl.load(mask1_ptr + rows) != 0
    mask0 = query_mask_0 != 0
    store_segm_reduce_scalars(
        segm_max_ptr,
        segm_exp_ptr,
        query_offset_0,
        query_offset_1,
        segm_idx,
        M,
        L,
        mask0,
        query_mask_1,
        num_query_heads,
        NUM_SEGMENTS_PER_SEQ,
    )


def kernel_impl():
    segm_max = torch.full((BUF_SIZE,), INIT_VAL, dtype=torch.float32, device=DEVICE)
    segm_exp = torch.full((BUF_SIZE,), INIT_VAL, dtype=torch.float32, device=DEVICE)
    _store_launcher[(1,)](
        segm_max,
        segm_exp,
        QUERY_OFFSET_1,
        M_VALS,
        L_VALS,
        QUERY_MASK_1,
        QUERY_OFFSET_0,
        QUERY_MASK_0,
        SEGM_IDX,
        num_query_heads=NUM_QUERY_HEADS,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS_PER_SEQ,
        BLOCK_M=BLOCK_M,
    )
    return segm_max, segm_exp


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_max, ref_exp = pytorch_ref()
        ker_max, ker_exp = kernel_impl()

        km, ke = ker_max.cpu(), ker_exp.cpu()
        torch.testing.assert_close(km, ref_max, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ke, ref_exp, rtol=1e-3, atol=1e-3)

        diff = torch.cat([(km - ref_max).abs(), (ke - ref_exp).abs()])
        stats = {
            "input_shape": (BLOCK_M,),
            "output_shape": (BUF_SIZE,),
            "in_dtype": str(M_VALS.dtype),
            "out_dtype": str(ker_max.dtype),
            "device": str(ker_max.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
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
            "Kernel: store_segm_reduce_scalars (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- M/L shape: {stats['input_shape']}\n",
                f"- query_offset_0: {QUERY_OFFSET_0}, segm_idx: {SEGM_IDX}\n",
                f"- query_offset_1: {QUERY_OFFSET_1.cpu().tolist()}\n",
                f"- query_mask_1: {QUERY_MASK_1.cpu().tolist()}\n",
                f"- num_query_heads: {NUM_QUERY_HEADS}, "
                f"NUM_SEGMENTS_PER_SEQ: {NUM_SEGMENTS_PER_SEQ}\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Output (segm_max + segm_expsum buffers):\n",
                f"- buffer shape: {stats['output_shape']}, dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
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
