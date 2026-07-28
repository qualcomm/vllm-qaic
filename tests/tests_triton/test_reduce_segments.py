"""
Standalone QAIC validation for `reduce_segments`.

Source under test:
vllm/v1/attention/ops/triton_unified_attention.py
  - reduce_segments  (finalizes the 3D split-softmax attention path: combines
    per-segment partial softmax max / expsum / output into the final
    per-(token, head) attention output via a numerically-stable LSE reduction)

In the 3D unified-attention path each program computes attention over a
contiguous *segment* of the KV sequence and stores three partials:
  - segm_output : the unnormalized weighted value sum  (sum_j p_j * V_j)
  - segm_max    : the running softmax max M for that segment
  - segm_expsum : the running exp-sum L for that segment
`reduce_segments` merges the segments of a token/head with the standard
online-softmax combine:
    overall_max     = max_s segm_max[s]
    scaled_expsum   = segm_expsum[s] * exp(segm_max[s] - overall_max)
    overall_expsum  = sum_s scaled_expsum
    out             = sum_s segm_output[s] * exp(segm_max[s] - overall_max)
                      / overall_expsum      (0 if overall_expsum == 0)

We feed synthetic per-segment partials directly (data preparation, not a
kernel call) and validate against this exact formula in pure PyTorch.

Config: seq_len=32, TILE_SIZE=16, NUM_SEGMENTS_PER_SEQ=2 -> all 2 segments
active (act_num_segments == NUM_SEGMENTS_PER_SEQ), HEAD_SIZE==HEAD_SIZE_PADDED
so no padding mask is exercised.

Reference: pure PyTorch LSE-weighted segment combination.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_unified_attention import reduce_segments  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_reduce_segments.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

DEVICE = "qaic"
NUM_TOKENS = 4
NUM_Q_HEADS = 2
HEAD_SIZE = 32
HEAD_SIZE_PADDED = 32
SEQ_LEN = 32
TILE_SIZE = 16
NUM_SEGMENTS_PER_SEQ = 2  # -> act_num_segments == 2 (all active)

torch.manual_seed(42)

# Per-segment partials.
SEGM_OUTPUT = torch.randn(
    NUM_TOKENS, NUM_Q_HEADS, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED,
    dtype=torch.float32, device=DEVICE,
)
SEGM_MAX = torch.randn(
    NUM_TOKENS, NUM_Q_HEADS, NUM_SEGMENTS_PER_SEQ, dtype=torch.float32, device=DEVICE
)
# expsum is strictly positive.
SEGM_EXPSUM = (
    torch.rand(
        NUM_TOKENS, NUM_Q_HEADS, NUM_SEGMENTS_PER_SEQ,
        dtype=torch.float32, device=DEVICE,
    )
    + 0.5
)
SEQ_LENS = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)
CU_SEQLENS_Q = torch.tensor([0, NUM_TOKENS], dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(segm_output, segm_max, segm_expsum):
    """Pure PyTorch LSE-weighted segment combination (matches source math)."""
    so = segm_output.float().cpu()  # [T, H, S, D]
    sm = segm_max.float().cpu()  # [T, H, S]
    se = segm_expsum.float().cpu()  # [T, H, S]

    overall_max = sm.max(dim=-1, keepdim=True).values  # [T, H, 1]
    w = torch.exp(sm - overall_max)  # [T, H, S]
    overall_expsum = (se * w).sum(dim=-1)  # [T, H]
    acc_sum = (so * w.unsqueeze(-1)).sum(dim=-2)  # [T, H, D]
    out = torch.where(
        overall_expsum.unsqueeze(-1) == 0.0,
        torch.zeros_like(acc_sum),
        acc_sum / overall_expsum.unsqueeze(-1),
    )
    return out[..., :HEAD_SIZE]  # [T, H, HEAD_SIZE]


def kernel_impl(segm_output, segm_max, segm_expsum):
    """Kernel wrapper: launch only."""
    out = torch.zeros(
        NUM_TOKENS, NUM_Q_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
    )
    grid = (NUM_TOKENS, NUM_Q_HEADS)
    reduce_segments[grid](
        output_ptr=out,
        segm_output_ptr=segm_output,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=SEQ_LENS,
        num_seqs=1,
        num_query_heads=NUM_Q_HEADS,
        out_scale_inv=1.0,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=0,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        query_start_len_ptr=CU_SEQLENS_Q,
        BLOCK_Q=1,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS_PER_SEQ,
        USE_FP8=False,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(SEGM_OUTPUT, SEGM_MAX, SEGM_EXPSUM)
        kernel_out = kernel_impl(SEGM_OUTPUT, SEGM_MAX, SEGM_EXPSUM)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "segm_output_shape": tuple(SEGM_OUTPUT.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(SEGM_OUTPUT.dtype),
            "device": str(SEGM_OUTPUT.device),
            "num_segments": NUM_SEGMENTS_PER_SEQ,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(SEGM_OUTPUT, SEGM_MAX, SEGM_EXPSUM))
        kern_stats = _bench(lambda: kernel_impl(SEGM_OUTPUT, SEGM_MAX, SEGM_EXPSUM))
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
            "Kernel: reduce_segments (3D split-softmax finalize)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- segm_output shape: {stats['segm_output_shape']}\n",
                f"- num_segments: {stats['num_segments']}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
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
