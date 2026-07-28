"""
Standalone QAIC validation for the `_compute_global_lse` device helper.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - _compute_global_lse(...)  (device-side @triton.jit helper).

Given per-block partial maxes and summed exponentials for a single `logit_idx`
(the outputs of _compute_block_max_and_sumexp across the vocab blocks), it merges
them into the global log-sum-exp:
    maxes    = local_max[logit_idx, 0:vocab_num_blocks]  (else -inf)
    sumexps  = local_sumexp[logit_idx, 0:vocab_num_blocks] (else 0)
    global_max = max(maxes)
    global_lse = global_max + log(sum(sumexps * exp(maxes - global_max)))

This equals log(sum over the full vocab of exp(logit)) reconstructed from the
per-block partials. Device-side helper with no launcher, so we wrap it in a
minimal `@triton.jit` launcher over one logit row. Pure float compare against a
pure-PyTorch reference that recomputes the block partials from raw logits then
folds them, i.e. logsumexp(full_logits). No RNG.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_global_lse.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    _compute_global_lse,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
# Emulate a vocab split into VOCAB_NUM_BLOCKS blocks of BLOCK_SIZE each.
VOCAB_BLOCK_SIZE = 8192
VOCAB_NUM_BLOCKS = 5
PADDED_VOCAB_NUM_BLOCKS = 8  # next power of 2
VOCAB = VOCAB_BLOCK_SIZE * VOCAB_NUM_BLOCKS
LOGIT_IDX = 0

# Raw logits for one row, split into blocks; derive the per-block partials the
# same way _compute_block_max_and_sumexp would (that is what the real kernel
# feeds into _compute_global_lse).
_FULL = torch.randn(VOCAB, dtype=torch.float32, device=DEVICE) * 5.0
_blocks = _FULL.view(VOCAB_NUM_BLOCKS, VOCAB_BLOCK_SIZE)
_bmax = _blocks.max(dim=1).values
_bsumexp = torch.exp(_blocks - _bmax[:, None]).sum(dim=1)

# Pad to PADDED_VOCAB_NUM_BLOCKS (padding is masked out inside the helper, but
# we allocate the full padded row as the launcher indexes arange(0, PADDED)).
LOCAL_MAX = torch.full(
    (1, PADDED_VOCAB_NUM_BLOCKS), float("-inf"), dtype=torch.float32, device=DEVICE
)
LOCAL_SUMEXP = torch.zeros(
    (1, PADDED_VOCAB_NUM_BLOCKS), dtype=torch.float32, device=DEVICE
)
LOCAL_MAX[0, :VOCAB_NUM_BLOCKS] = _bmax
LOCAL_SUMEXP[0, :VOCAB_NUM_BLOCKS] = _bsumexp


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
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


def pytorch_ref(local_max, local_sumexp):
    maxes = local_max.cpu().float()[0, :VOCAB_NUM_BLOCKS]
    sumexps = local_sumexp.cpu().float()[0, :VOCAB_NUM_BLOCKS]
    global_max = maxes.max()
    global_lse = global_max + torch.log(
        (sumexps * torch.exp(maxes - global_max)).sum()
    )
    return global_lse.reshape(1)


@triton.jit
def _global_lse_launcher(
    local_max_ptr,
    local_max_stride,
    local_sumexp_ptr,
    local_sumexp_stride,
    out_ptr,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    lse = _compute_global_lse(
        local_max_ptr,
        local_max_stride,
        local_sumexp_ptr,
        local_sumexp_stride,
        logit_idx,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS,
    )
    tl.store(out_ptr + 0, lse)


def kernel_impl(local_max, local_sumexp):
    out = torch.empty(1, dtype=torch.float32, device=local_max.device)
    _global_lse_launcher[(1,)](
        local_max,
        local_max.stride(0),
        local_sumexp,
        local_sumexp.stride(0),
        out,
        LOGIT_IDX,
        VOCAB_NUM_BLOCKS,
        PADDED_VOCAB_NUM_BLOCKS=PADDED_VOCAB_NUM_BLOCKS,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(LOCAL_MAX, LOCAL_SUMEXP)
        kernel_out = kernel_impl(LOCAL_MAX, LOCAL_SUMEXP)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(
            kernel_cpu.float(), ref_cpu.float(), rtol=1e-3, atol=1e-3
        )

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        stats = {
            "input_shape": tuple(LOCAL_MAX.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(LOCAL_MAX.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(LOCAL_MAX.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(LOCAL_MAX, LOCAL_SUMEXP))
        kern_stats = _bench(lambda: kernel_impl(LOCAL_MAX, LOCAL_SUMEXP))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _compute_global_lse (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
    sys.exit(0 if main() == "SUCCESS" else 1)
