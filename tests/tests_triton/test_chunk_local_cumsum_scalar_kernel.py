"""
Standalone QAIC validation for `chunk_local_cumsum_scalar_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/cumsum.py
  - chunk_local_cumsum_scalar_kernel  (chunk-local causal cumulative sum of a
    scalar per-head gating/decay signal, reset every `chunk_size` tokens)

For each chunk of `chunk_size` tokens along the time axis, the kernel
computes a causal (inclusive) cumulative sum independently per chunk -- i.e.
NOT a running cumsum across the whole sequence. With `reverse=True`, per
chunk this becomes a suffix ("reverse") cumsum: out[i] = sum(g[i:]) within
that chunk, computed in the Triton source as
    b_o = -cumsum(b_s) + sum(b_s) + b_s
which is algebraically the reverse-inclusive cumsum.

Reference: pure PyTorch per-chunk `torch.cumsum`.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_local_cumsum_scalar_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/cumsum.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402


# chunk_local_cumsum_scalar_kernel is wrapped in @triton.autotune. Triton's
# generic autotuner benchmarking path (triton.testing.do_bench) calls
# Event.elapsed_time() across two independently-created torch.qaic Events,
# which raises "expected other to be a torch.Event object" on this QAIC
# backend/torch_qaic version. This is a benchmarking-harness incompatibility,
# unrelated to kernel correctness, so we shim do_bench to just execute the
# candidate config once and report a constant timing. The kernel itself
# still executes for real on qaic hardware.
def _qaic_safe_do_bench(fn, *args, **kwargs):
    fn()
    quantiles = kwargs.get("quantiles")
    if quantiles is not None:
        return [1.0] * len(quantiles) if len(quantiles) > 1 else 1.0
    return 1.0


_triton_testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.cumsum import (  # noqa: E402
    chunk_local_cumsum_scalar,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"

B = 1
T = 64
H = 2
CHUNK_SIZE = 64  # single chunk (T == chunk_size)
REVERSE = False
OUTPUT_DTYPE = torch.float32

torch.manual_seed(42)
G = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE)


def pytorch_ref(g, chunk_size, reverse=False):
    """Pure PyTorch reference for chunk-local causal cumulative sum.

    g: [B, T, H] (head_first=False). Within each chunk of `chunk_size`
    tokens along dim=1, computes the causal cumsum independently per chunk
    (resets every chunk_size tokens). If reverse, computes the suffix
    (reverse-inclusive) cumsum within each chunk instead.
    """
    B_, T_, H_ = g.shape
    out = torch.empty_like(g, dtype=torch.float32)
    num_chunks = (T_ + chunk_size - 1) // chunk_size
    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, T_)
        chunk = g[:, start:end, :].to(torch.float32)
        fwd = torch.cumsum(chunk, dim=1)
        if reverse:
            total = chunk.sum(dim=1, keepdim=True)
            out[:, start:end, :] = -fwd + total + chunk
        else:
            out[:, start:end, :] = fwd
    return out


def kernel_impl(g, chunk_size, reverse=False):
    return chunk_local_cumsum_scalar(
        g,
        chunk_size,
        reverse=reverse,
        cu_seqlens=None,
        chunk_indices=None,
        head_first=False,
        output_dtype=OUTPUT_DTYPE,
    )


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns dict of latency stats (ms)."""
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


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(G, CHUNK_SIZE, reverse=REVERSE)
        kernel_out = kernel_impl(G, CHUNK_SIZE, reverse=REVERSE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "input_shape": tuple(G.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(G.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(G.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": rel_err,
            "chunk_size": CHUNK_SIZE,
            "reverse": REVERSE,
        }
        pt_stats = _bench(lambda: pytorch_ref(G, CHUNK_SIZE, reverse=REVERSE))
        kern_stats = _bench(lambda: kernel_impl(G, CHUNK_SIZE, reverse=REVERSE))
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
        lines = [
            f"{timestamp}\n",
            "Kernel: chunk_local_cumsum_scalar_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- g shape: {stats['input_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- chunk_size: {stats['chunk_size']}\n")
            lines.append(f"- reverse: {stats['reverse']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_error: {stats['rel_error']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
