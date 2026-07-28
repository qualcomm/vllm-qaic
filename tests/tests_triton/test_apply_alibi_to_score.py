"""
Standalone QAIC validation for the `apply_alibi_to_score` @triton.jit device
helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - apply_alibi_to_score(S, alibi_slope, seq_offset, context_len,
        query_pos, USE_ALIBI_SQRT)

Adds ALiBi positional bias to raw attention scores. Exact source:
    if USE_ALIBI_SQRT:
        relative_pos = seq_offset - (context_len + query_pos[:, None])
        alibi_offset = where(relative_pos <= 0,
                             -sqrt((-relative_pos).float), 0.0)
    else:
        alibi_offset = seq_offset - context_len
    return S + alibi_slope[:, None] * alibi_offset

Shapes: S [BLOCK_M, TILE], alibi_slope [BLOCK_M], seq_offset [TILE] (per-key
absolute pos), query_pos [BLOCK_M] (per-query relative pos), context_len
scalar. We validate BOTH the linear and sqrt variants.

Reference: pure PyTorch replication of both branches.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import apply_alibi_to_score

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_apply_alibi_to_score.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
TILE = 16
CONTEXT_LEN = 4

torch.manual_seed(42)
SCORES = torch.randn(BLOCK_M, TILE, dtype=torch.float32, device=DEVICE)
ALIBI_SLOPE = (torch.arange(BLOCK_M, dtype=torch.float32, device=DEVICE) + 1) * 0.1
SEQ_OFFSET = torch.arange(TILE, dtype=torch.float32, device=DEVICE)
QUERY_POS = torch.arange(BLOCK_M, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(scores, slope, seq_offset, context_len, query_pos, use_sqrt):
    """Pure PyTorch replication of apply_alibi_to_score."""
    s = scores.cpu()
    slope = slope.cpu()
    seq_offset = seq_offset.cpu()
    query_pos = query_pos.cpu()
    if use_sqrt:
        relative_pos = seq_offset.reshape(1, TILE) - (
            context_len + query_pos.reshape(BLOCK_M, 1)
        )
        alibi_offset = torch.where(
            relative_pos <= 0,
            -torch.sqrt((-relative_pos).clamp_min(0).float()),
            torch.zeros_like(relative_pos),
        )
    else:
        alibi_offset = (seq_offset - context_len).reshape(1, TILE)
    return s + slope.reshape(BLOCK_M, 1) * alibi_offset


@triton.jit
def _alibi_launcher(
    scores_ptr,
    slope_ptr,
    seq_offset_ptr,
    query_pos_ptr,
    out_ptr,
    context_len,
    BLOCK_M: tl.constexpr,
    TILE: tl.constexpr,
    USE_ALIBI_SQRT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, TILE)
    idx = rows[:, None] * TILE + cols[None, :]
    S = tl.load(scores_ptr + idx)  # [BLOCK_M, TILE]
    alibi_slope = tl.load(slope_ptr + rows)  # [BLOCK_M]
    seq_offset = tl.load(seq_offset_ptr + cols)  # [TILE]
    query_pos = tl.load(query_pos_ptr + rows)  # [BLOCK_M]
    res = apply_alibi_to_score(
        S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
    )
    tl.store(out_ptr + idx, res)


def kernel_impl(use_sqrt):
    out = torch.empty(BLOCK_M, TILE, dtype=torch.float32, device=DEVICE)
    _alibi_launcher[(1,)](
        SCORES.reshape(-1),
        ALIBI_SLOPE,
        SEQ_OFFSET,
        QUERY_POS,
        out.reshape(-1),
        CONTEXT_LEN,
        BLOCK_M=BLOCK_M,
        TILE=TILE,
        USE_ALIBI_SQRT=use_sqrt,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        max_diff = 0.0
        mean_diffs = []
        for use_sqrt in (False, True):
            ref = pytorch_ref(
                SCORES, ALIBI_SLOPE, SEQ_OFFSET, CONTEXT_LEN, QUERY_POS, use_sqrt
            )
            ker = kernel_impl(use_sqrt)
            ref_cpu, ker_cpu = ref.cpu(), ker.cpu()
            torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
            diff = (ker_cpu - ref_cpu).abs()
            max_diff = max(max_diff, diff.max().item())
            mean_diffs.append(diff.mean().item())

        stats = {
            "input_shape": tuple(SCORES.shape),
            "output_shape": (BLOCK_M, TILE),
            "in_dtype": str(SCORES.dtype),
            "out_dtype": "torch.float32",
            "device": str(SCORES.device),
            "max_abs_diff": max_diff,
            "mean_abs_diff": sum(mean_diffs) / len(mean_diffs),
        }
        pt_stats = _bench(
            lambda: pytorch_ref(
                SCORES, ALIBI_SLOPE, SEQ_OFFSET, CONTEXT_LEN, QUERY_POS, True
            )
        )
        kern_stats = _bench(lambda: kernel_impl(True))
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
            "Kernel: apply_alibi_to_score (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs (both linear + sqrt variants tested):\n",
                f"- scores shape: {stats['input_shape']}\n",
                f"- alibi_slope: {ALIBI_SLOPE.cpu().tolist()}\n",
                f"- seq_offset: {SEQ_OFFSET.cpu().tolist()}\n",
                f"- query_pos: {QUERY_POS.cpu().tolist()}\n",
                f"- context_len: {CONTEXT_LEN}\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Output (score + ALiBi bias):\n",
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
