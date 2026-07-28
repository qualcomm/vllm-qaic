"""
Standalone QAIC validation for `_apply_grammar_bitmask_kernel`.

Source under test:
vllm/v1/worker/gpu/structured_outputs.py
  - _apply_grammar_bitmask_kernel (apply a packed structured-output grammar
    bitmask to logits, setting disallowed vocab entries to -inf).

The bitmask is packed 32 vocab entries per int32 word. Bit == 1 means the
token is ALLOWED; bit == 0 means DISALLOWED and the kernel sets that logit to
-inf. `logits_indices` maps each bitmask row to a logits row.

We launch the kernel directly (the production launcher uses CUDA streams +
InputBatch). Float compare on finite entries + exact -inf-mask equality.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.structured_outputs import _apply_grammar_bitmask_kernel

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_apply_grammar_bitmask_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/structured_outputs.py"
KERNEL_NAME = "_apply_grammar_bitmask_kernel"

DEVICE = "qaic"
NUM_LOGITS_ROWS = 4
NUM_MASKS = 3
VOCAB_SIZE = 256
PACKED = (VOCAB_SIZE + 31) // 32
BLOCK_SIZE = 8192

torch.manual_seed(42)
LOGITS = torch.randn(NUM_LOGITS_ROWS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
# Map bitmask row -> logits row (rows 0,1,3 receive masks; row 2 untouched).
LOGITS_INDICES = torch.tensor([0, 1, 3], dtype=torch.int32, device=DEVICE)
# Random packed bitmask: 1 bit == allowed, 0 bit == disallowed.
BITMASK = torch.randint(
    -(2**31), 2**31, (NUM_MASKS, PACKED), dtype=torch.int32, device=DEVICE
)


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
        f"Kernel: {KERNEL_NAME}\n",
        f"Kernel file: {KERNEL_FILE_PATH}\n",
        f"Device target: QAIC (device='{DEVICE}')\n",
        f"Status: {status}\n\n",
    ]
    if status == "SUCCESS":
        for k, v in stats.items():
            lines.append(f"- {k}: {v}\n")
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
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


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


def _unpack_allowed(packed_row):
    """Return a [VOCAB_SIZE] bool tensor where True == allowed (bit==1),
    matching the kernel's `(packed >> arange(32)) & 1`."""
    packed = packed_row.to(torch.int64) & 0xFFFFFFFF
    bits = torch.zeros(PACKED * 32, dtype=torch.bool)
    for w in range(PACKED):
        val = int(packed[w].item())
        for b in range(32):
            bits[w * 32 + b] = bool((val >> b) & 1)
    return bits[:VOCAB_SIZE]


def pytorch_ref():
    logits = LOGITS.cpu().clone().to(torch.float32)
    indices = LOGITS_INDICES.cpu()
    bitmask = BITMASK.cpu()
    for m in range(NUM_MASKS):
        row = int(indices[m].item())
        allowed = _unpack_allowed(bitmask[m])  # True where allowed
        disallowed = ~allowed
        logits[row][disallowed] = float("-inf")
    return logits


def kernel_impl():
    logits = LOGITS.clone()
    grid = (NUM_MASKS, triton.cdiv(VOCAB_SIZE, BLOCK_SIZE))
    _apply_grammar_bitmask_kernel[grid](
        logits,
        logits.stride(0),
        LOGITS_INDICES,
        BITMASK,
        BITMASK.stride(0),
        VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return logits


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        out = kernel_impl()
        ref_cpu = ref.cpu()
        out_cpu = out.cpu()

        same_inf = torch.isinf(ref_cpu) == torch.isinf(out_cpu)
        finite_mask = ~torch.isinf(ref_cpu)
        torch.testing.assert_close(
            out_cpu[finite_mask], ref_cpu[finite_mask], rtol=1e-3, atol=1e-3
        )
        assert bool(same_inf.all()), "-inf mask mismatch"

        diff = (out_cpu[finite_mask] - ref_cpu[finite_mask]).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "bitmask_shape": tuple(BITMASK.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "num_masked_entries": int(torch.isinf(out_cpu).sum().item()),
            "inf_mask_match": bool(same_inf.all()),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
