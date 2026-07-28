"""
Standalone QAIC validation for the `apply_softcap` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_attention_helpers.py
  - apply_softcap(S, x)

`apply_softcap` is a device-side helper (called from inside the unified
attention kernel), so it cannot be launched directly. We wrap it in a
minimal `@triton.jit` launcher kernel (`_softcap_launcher`) that loads a
tile of scores, calls the helper, and stores the result.

Exact source formula (S = score tensor, x = softcap scalar):
    Sdiv = S / x
    p1 = exp(Sdiv); p2 = exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)          # == x * tanh(S / x)

i.e. the tanh-style logit softcap  softcap * tanh(scores / softcap).

Reference: pure PyTorch  x * tanh(S / x).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import apply_softcap

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_apply_softcap.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_attention_helpers.py"

DEVICE = "qaic"
BLOCK_M = 8
TILE = 16
N = BLOCK_M * TILE
SOFTCAP = 30.0

torch.manual_seed(42)
# Scores span a wide range so softcap has an observable effect.
SCORES = torch.randn(BLOCK_M, TILE, dtype=torch.float32, device=DEVICE) * 20.0


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


def pytorch_ref(scores, softcap):
    """Pure PyTorch tanh-style softcap: softcap * tanh(scores / softcap)."""
    return softcap * torch.tanh(scores / softcap)


@triton.jit
def _softcap_launcher(scores_ptr, out_ptr, softcap, N: tl.constexpr):
    offs = tl.arange(0, N)
    s = tl.load(scores_ptr + offs)
    res = apply_softcap(s, softcap)
    tl.store(out_ptr + offs, res)


def kernel_impl(scores, softcap):
    out = torch.empty_like(scores)
    _softcap_launcher[(1,)](scores.reshape(-1), out.reshape(-1), softcap, N=N)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(SCORES, SOFTCAP)
        kernel_out = kernel_impl(SCORES, SOFTCAP)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(SCORES.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(SCORES.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(SCORES.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(SCORES, SOFTCAP))
        kern_stats = _bench(lambda: kernel_impl(SCORES, SOFTCAP))
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
            "Kernel: apply_softcap (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- scores shape: {stats['input_shape']}\n",
                f"- softcap: {SOFTCAP}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- max_rel_err: {stats['rel_err']}\n",
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
