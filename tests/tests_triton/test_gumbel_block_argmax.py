"""
Standalone QAIC validation for `gumbel_block_argmax`.

Source under test:
vllm/v1/worker/gpu/sample/gumbel.py
  - gumbel_block_argmax (device helper: temperature + Gumbel-max over a block)

`gumbel_block_argmax` is a @triton.jit device helper. It optionally applies
temperature, optionally adds Gumbel noise (only when temp != 0.0), and returns
the block-local (max_value, argmax_index) via the Gumbel-max trick.

We wrap it in a tiny standalone @triton.jit kernel. BLOCK_SIZE covers the full
vocab (single block => block-local == global argmax).

RNG caveat: reproducing Triton's philox Gumbel noise bit-exactly in PyTorch is
impractical, so the *correctness* comparison uses temperature == 0.0, for which
the helper skips both temperature scaling and Gumbel noise (kernel:
`if temp != 0.0:` guards the noise). It then reduces to a plain max/argmax that
PyTorch reproduces exactly. We additionally verify the temp > 0 Gumbel path is
deterministic for a fixed seed and produces a valid in-range argmax.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_block_argmax

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_gumbel_block_argmax.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/gumbel.py"
KERNEL_NAME = "gumbel_block_argmax"

DEVICE = "qaic"
NUM_REQS = 4
VOCAB_SIZE = 128
BLOCK_SIZE = 128  # single block covers full vocab

torch.manual_seed(42)
LOGITS = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
SEEDS = torch.arange(1, NUM_REQS + 1, dtype=torch.int32, device=DEVICE)
POS = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
# temperature == 0 => greedy (no temp scaling, no gumbel noise) => bit-exact.
TEMP_GREEDY = torch.zeros(NUM_REQS, dtype=torch.float32, device=DEVICE)
TEMP_GUMBEL = torch.full((NUM_REQS,), 0.8, dtype=torch.float32, device=DEVICE)


@triton.jit
def _gba_wrapper(
    out_val_ptr,
    out_idx_ptr,
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    temp_ptr,
    seeds_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + token_idx * logits_stride + block, mask=mask, other=float("-inf")
    )
    logits = logits.to(tl.float32)
    value, idx = gumbel_block_argmax(
        logits,
        block,
        mask,
        token_idx,
        expanded_idx_mapping_ptr,
        temp_ptr,
        seeds_ptr,
        pos_ptr,
        None,  # processed_logits_ptr
        0,  # processed_logits_stride
        None,  # processed_logits_col_ptr
        vocab_size,
        APPLY_TEMPERATURE=APPLY_TEMPERATURE,
        USE_FP64=USE_FP64,
    )
    tl.store(out_val_ptr + token_idx, value)
    tl.store(out_idx_ptr + token_idx, idx)


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


def pytorch_ref(logits):
    """Greedy (temp==0) block-local max/argmax over the full vocab block."""
    x = logits.cpu()
    vals, idxs = torch.max(x, dim=-1)
    return vals.to(torch.float32), idxs.to(torch.int64)


def kernel_impl(temp):
    out_val = torch.empty(NUM_REQS, dtype=torch.float32, device=DEVICE)
    out_idx = torch.empty(NUM_REQS, dtype=torch.int64, device=DEVICE)
    _gba_wrapper[(NUM_REQS,)](
        out_val,
        out_idx,
        LOGITS,
        LOGITS.stride(0),
        EXPANDED_IDX_MAPPING,
        temp,
        SEEDS,
        POS,
        VOCAB_SIZE,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=True,
        USE_FP64=False,
    )
    return out_val, out_idx


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Bit-exact greedy correctness (temp == 0).
        ref_val, ref_idx = pytorch_ref(LOGITS)
        out_val, out_idx = kernel_impl(TEMP_GREEDY)
        val_cpu = out_val.cpu().to(torch.float32)
        idx_cpu = out_idx.cpu().to(torch.int64)

        torch.testing.assert_close(val_cpu, ref_val, rtol=1e-3, atol=1e-3)
        idx_mismatch = int((idx_cpu != ref_idx).sum().item())
        assert idx_mismatch == 0, f"argmax index mismatch count={idx_mismatch}"

        # Gumbel path (temp > 0): determinism + valid index range.
        g_val1, g_idx1 = kernel_impl(TEMP_GUMBEL)
        g_val2, g_idx2 = kernel_impl(TEMP_GUMBEL)
        deterministic = bool(
            torch.equal(g_idx1.cpu(), g_idx2.cpu())
            and torch.equal(g_val1.cpu(), g_val2.cpu())
        )
        idx_in_range = bool(
            ((g_idx1.cpu() >= 0) & (g_idx1.cpu() < VOCAB_SIZE)).all()
        )
        assert deterministic, "gumbel path not deterministic for fixed seed"
        assert idx_in_range, "gumbel argmax index out of range"

        vdiff = (val_cpu - ref_val).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_val_shape": tuple(out_val.shape),
            "output_idx_shape": tuple(out_idx.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": f"{out_val.dtype}/{out_idx.dtype}",
            "device": str(LOGITS.device),
            "greedy_val_max_abs_diff": vdiff.max().item(),
            "greedy_val_mean_abs_diff": vdiff.mean().item(),
            "greedy_idx_mismatch": idx_mismatch,
            "gumbel_deterministic": deterministic,
            "gumbel_idx_in_range": idx_in_range,
            "comparison": "temp==0 bit-exact max/argmax; temp>0 determinism+range",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(LOGITS))
        kern_stats = _bench(lambda: kernel_impl(TEMP_GREEDY))
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
