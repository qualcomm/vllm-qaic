"""
Standalone QAIC validation for the `_compute_block_max_and_sumexp` device helper.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - _compute_block_max_and_sumexp(logits)  (device-side @triton.jit helper).

Exact source semantics for a 1-D block of `logits`:
    block_max = max(logits, axis=0)
    block_sumexp = sum(exp(logits - block_max))   if block_max > -inf else 0.0
i.e. the numerically-stable per-block partial for a log-sum-exp reduction.

It is a device-side helper with no launcher, so we wrap it in a minimal
`@triton.jit` launcher (`_block_max_sumexp_launcher`) that loads N logits,
calls the helper, and stores the two scalar outputs. Pure float comparison
against a pure-PyTorch reference (no RNG).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_block_max_and_sumexp.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    _compute_block_max_and_sumexp,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
BLOCK = 8192
LOGITS = torch.randn(BLOCK, dtype=torch.float32, device=DEVICE) * 5.0


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


def pytorch_ref(logits):
    logits = logits.cpu().float()
    block_max = logits.max()
    block_sumexp = torch.exp(logits - block_max).sum()
    return torch.stack([block_max, block_sumexp])


@triton.jit
def _block_max_sumexp_launcher(logits_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    logits = tl.load(logits_ptr + offs).to(tl.float32)
    block_max, block_sumexp = _compute_block_max_and_sumexp(logits)
    tl.store(out_ptr + 0, block_max)
    tl.store(out_ptr + 1, block_sumexp)


def kernel_impl(logits):
    out = torch.empty(2, dtype=torch.float32, device=logits.device)
    _block_max_sumexp_launcher[(1,)](logits, out, N=BLOCK)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(LOGITS)
        kernel_out = kernel_impl(LOGITS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(
            kernel_cpu.float(), ref_cpu.float(), rtol=1e-3, atol=1e-3
        )

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(LOGITS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(LOGITS.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(LOGITS))
        kern_stats = _bench(lambda: kernel_impl(LOGITS))
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
            "Kernel: _compute_block_max_and_sumexp (device helper)\n",
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
