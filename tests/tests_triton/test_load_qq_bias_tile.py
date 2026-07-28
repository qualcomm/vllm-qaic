"""
Standalone QAIC validation for the `load_qq_bias_tile` @triton.jit device
helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - load_qq_bias_tile(qq_bias_row_ptrs, seq_offset, context_len,
        qq_bias_stride_0)

Loads the query-query bias slice for a tile's key positions (used by
bidirectional / document-mask attention). Exact source:
    key_rel_pos = seq_offset - context_len
    is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
    return load(qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :], other=0.0)

`qq_bias_row_ptrs` is a per-query-row pointer tensor of shape [BLOCK_M, 1]
(base + query_row * stride_0). Adding `key_rel_pos[None, :]` ([1, TILE])
broadcasts to a [BLOCK_M, TILE] gather. Keys whose relative position falls
outside [0, qq_bias_stride_0) are masked to 0.

WRAPPING NOTE: the helper takes a raw *pointer tensor* rather than a plain
value tensor. The launcher constructs that pointer tensor from the qq_bias
base pointer + row offsets exactly as the parent attention kernel does, then
calls the helper on it.

Reference: pure PyTorch gather with the same in-range masking.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import load_qq_bias_tile

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_load_qq_bias_tile.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
TILE = 16
QQ_WIDTH = 12  # qq_bias_stride_0 (columns per query row)
CONTEXT_LEN = 3

torch.manual_seed(42)
# qq_bias matrix: [BLOCK_M query rows, QQ_WIDTH query keys], row-major.
QQ_BIAS = torch.randn(BLOCK_M, QQ_WIDTH, dtype=torch.float32, device=DEVICE)
# Per-key absolute positions; key_rel_pos = seq_offset - context_len spans
# both in-range and out-of-range values.
SEQ_OFFSET = torch.arange(TILE, dtype=torch.int32, device=DEVICE)


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
    """Pure PyTorch gather matching load_qq_bias_tile."""
    qq = QQ_BIAS.cpu()  # [BLOCK_M, QQ_WIDTH]
    seq_offset = SEQ_OFFSET.cpu()
    key_rel_pos = seq_offset - CONTEXT_LEN  # [TILE]
    in_range = (key_rel_pos >= 0) & (key_rel_pos < QQ_WIDTH)  # [TILE]
    safe_idx = key_rel_pos.clamp(0, QQ_WIDTH - 1).long()  # [TILE]
    gathered = qq[:, safe_idx]  # [BLOCK_M, TILE]
    out = torch.where(
        in_range.reshape(1, TILE), gathered, torch.zeros_like(gathered)
    )
    return out


@triton.jit
def _qq_launcher(
    qq_bias_ptr,
    seq_offset_ptr,
    out_ptr,
    context_len,
    qq_bias_stride_0,
    BLOCK_M: tl.constexpr,
    TILE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, TILE)
    # Per-query-row base pointers into qq_bias: [BLOCK_M, 1].
    qq_bias_row_ptrs = qq_bias_ptr + rows[:, None] * qq_bias_stride_0
    seq_offset = tl.load(seq_offset_ptr + cols)  # [TILE]
    tile = load_qq_bias_tile(
        qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
    )  # [BLOCK_M, TILE]
    tl.store(out_ptr + rows[:, None] * TILE + cols[None, :], tile)


def kernel_impl():
    out = torch.empty(BLOCK_M, TILE, dtype=torch.float32, device=DEVICE)
    _qq_launcher[(1,)](
        QQ_BIAS.reshape(-1),
        SEQ_OFFSET,
        out.reshape(-1),
        CONTEXT_LEN,
        QQ_WIDTH,
        BLOCK_M=BLOCK_M,
        TILE=TILE,
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
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(QQ_BIAS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(QQ_BIAS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(kernel_out.device),
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
            "Kernel: load_qq_bias_tile (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- qq_bias shape: {stats['input_shape']} (stride_0={QQ_WIDTH})\n",
                f"- seq_offset: {SEQ_OFFSET.cpu().tolist()}\n",
                f"- context_len: {CONTEXT_LEN}\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Output (gathered qq-bias tile):\n",
                f"- shape: {stats['output_shape']}, dtype: {stats['out_dtype']}\n",
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
