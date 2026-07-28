"""
Standalone QAIC validation for the `init_softmax_M` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - init_softmax_M(sink_ptr, query_offset_1, query_mask_1, segm_idx_or_0,
        BLOCK_M, USE_SINKS, IS_3D)

Initializes the running row-max accumulator M (shape [BLOCK_M]) for the
online softmax. Exact source semantics:
    M = full([BLOCK_M], -inf)
    if USE_SINKS:
        load_sinks = (not IS_3D) or (segm_idx_or_0 == 0)
        if load_sinks:
            M = load(sink_ptr + query_offset_1, mask=query_mask_1,
                     other=-inf).to(float32)
    return M

So without sinks -> all -inf. With sinks (2D, or 3D segment 0) -> per-head
sink bias where masked-in, else -inf. We validate three configs:
    (USE_SINKS=False), (USE_SINKS=True, IS_3D=False),
    (USE_SINKS=True, IS_3D=True, segm_idx=1) -> should stay -inf.

query_offset_1 / query_mask_1 are per-row (BLOCK_M) tile tensors; the
launcher loads them from pointers. Comparison handles -inf positions
explicitly and compares finite entries with tolerance.

Reference: pure PyTorch replication of the gather-with-mask.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import init_softmax_M

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_init_softmax_M.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
NUM_HEADS = 16

torch.manual_seed(42)
# Per-head sink bias.
SINKS = torch.randn(NUM_HEADS, dtype=torch.float32, device=DEVICE)
# Per-row head offset into sinks.
QUERY_OFFSET_1 = torch.arange(BLOCK_M, dtype=torch.int32, device=DEVICE)
# Mask out the last two rows (simulate padding beyond valid heads).
QUERY_MASK_1 = torch.ones(BLOCK_M, dtype=torch.int32, device=DEVICE)
QUERY_MASK_1[-2:] = 0


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


def pytorch_ref(sinks, offset, mask, segm_idx, use_sinks, is_3d):
    """Pure PyTorch replication of init_softmax_M."""
    sinks = sinks.cpu()
    offset = offset.cpu().long()
    mask = mask.cpu().bool()
    m = torch.full((BLOCK_M,), float("-inf"), dtype=torch.float32)
    if use_sinks:
        load_sinks = (not is_3d) or (segm_idx == 0)
        if load_sinks:
            gathered = torch.where(
                mask, sinks[offset], torch.tensor(float("-inf"))
            )
            m = gathered.to(torch.float32)
    return m


@triton.jit
def _init_M_launcher(
    sink_ptr,
    offset_ptr,
    mask_ptr,
    out_ptr,
    segm_idx_or_0,
    BLOCK_M: tl.constexpr,
    USE_SINKS: tl.constexpr,
    IS_3D: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    query_offset_1 = tl.load(offset_ptr + rows)
    query_mask_1 = tl.load(mask_ptr + rows) != 0
    m = init_softmax_M(
        sink_ptr,
        query_offset_1,
        query_mask_1,
        segm_idx_or_0,
        BLOCK_M,
        USE_SINKS,
        IS_3D,
    )
    tl.store(out_ptr + rows, m)


def kernel_impl(sinks, offset, mask, segm_idx, use_sinks, is_3d):
    out = torch.empty(BLOCK_M, dtype=torch.float32, device=sinks.device)
    _init_M_launcher[(1,)](
        sinks,
        offset,
        mask,
        out,
        segm_idx,
        BLOCK_M=BLOCK_M,
        USE_SINKS=use_sinks,
        IS_3D=is_3d,
    )
    return out


def _compare(ref_cpu, kernel_cpu):
    same_inf = torch.isinf(ref_cpu) == torch.isinf(kernel_cpu)
    finite = ~torch.isinf(ref_cpu)
    torch.testing.assert_close(
        kernel_cpu[finite], ref_cpu[finite], rtol=1e-3, atol=1e-3
    )
    assert bool(same_inf.all()), "-inf mask mismatch"
    diff = (kernel_cpu[finite] - ref_cpu[finite]).abs()
    return diff


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        configs = [
            ("no_sinks", 0, False, False),
            ("sinks_2d", 0, True, False),
            ("sinks_3d_seg1", 1, True, True),  # -> all -inf
        ]
        max_diff = 0.0
        mean_diffs = []
        for name, segm, use_sinks, is_3d in configs:
            ref = pytorch_ref(SINKS, QUERY_OFFSET_1, QUERY_MASK_1, segm, use_sinks, is_3d)
            ker = kernel_impl(SINKS, QUERY_OFFSET_1, QUERY_MASK_1, segm, use_sinks, is_3d)
            diff = _compare(ref.cpu(), ker.cpu())
            if diff.numel() > 0:
                max_diff = max(max_diff, diff.max().item())
                mean_diffs.append(diff.mean().item())

        stats = {
            "input_shape": tuple(SINKS.shape),
            "output_shape": (BLOCK_M,),
            "in_dtype": str(SINKS.dtype),
            "out_dtype": "torch.float32",
            "device": str(SINKS.device),
            "max_abs_diff": max_diff,
            "mean_abs_diff": (sum(mean_diffs) / len(mean_diffs)) if mean_diffs else 0.0,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(SINKS, QUERY_OFFSET_1, QUERY_MASK_1, 0, True, False))
        kern_stats = _bench(
            lambda: kernel_impl(SINKS, QUERY_OFFSET_1, QUERY_MASK_1, 0, True, False))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: init_softmax_M (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- sinks shape: {stats['input_shape']}\n",
                f"- query_offset_1: {QUERY_OFFSET_1.cpu().tolist()}\n",
                f"- query_mask_1: {QUERY_MASK_1.cpu().tolist()}\n",
                f"- BLOCK_M: {BLOCK_M}\n",
                f"- in dtype: {stats['in_dtype']}, device: {stats['device']}\n\n",
                "Configs: no_sinks / sinks_2d / sinks_3d_seg1(all -inf)\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}, dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff (finite): {stats['max_abs_diff']}\n",
                f"- mean_abs_diff (finite): {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
                    "Timing:\n",
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
                ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
