"""
Standalone QAIC validation for `chunk_scaled_dot_kkt_fwd_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py
  - chunk_scaled_dot_kkt_fwd_kernel  (beta-scaled, strictly-causal K @ K^T
    per chunk, with optional gating decay.)

For a single chunk (B=1, T=64=chunk_size, Hg=1, H=1, K=32), with no gate
(USE_G=False):
    A[i, j] = beta[i] * (k[i] . k[j])   for j < i  (strictly lower triangular)
    A[i, j] = 0                          otherwise (upper triangle + diagonal)

Reference: pure PyTorch beta-scaled K@K^T with a strict lower-triangular
mask, computed in fp32 then cast to `output_dtype`.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_scaled_dot_kkt_fwd_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py"

DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing  # noqa: E402


def _qaic_safe_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                         quantiles=None, return_mode="mean"):
    """Wall-clock replacement for `triton.testing.do_bench`.

    The stock implementation times kernels via `torch.Event.elapsed_time`,
    which is broken for the QAIC backend in this environment
    (`RuntimeError: expected other to be a torch.Event object`). Triton's
    `@triton.autotune` decorator calls this during its very first config
    search, so every autotuned FLA kernel hits this crash before any of our
    kernel-under-test logic even runs. We swap in a simple `time.perf_counter`
    based timer (device-synced) so autotuning can proceed; this only affects
    *which* config is picked / timed, not kernel correctness.
    """
    import time

    fn()
    torch.qaic.synchronize()
    n_repeat = 3
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        torch.qaic.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    if quantiles is not None:
        import numpy as np

        return list(np.quantile(times, quantiles))
    return sum(times) / len(times)


triton.testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import (  # noqa: E402
    chunk_scaled_dot_kkt_fwd,
)

# ---------------------------------------------------------------------------
# Global inputs (single chunk: B=1, T=64=chunk_size, Hg=1, H=1, K=32)
# ---------------------------------------------------------------------------
B = 1
T = 64
Hg = 1
H = 1
K = 32
CHUNK_SIZE = 64
DTYPE = torch.float32
OUTPUT_DTYPE = torch.float32

torch.manual_seed(42)
KEY = torch.randn(B, T, Hg, K, dtype=DTYPE, device=DEVICE) * 0.1
BETA = torch.rand(B, T, H, dtype=DTYPE, device=DEVICE) * 0.5 + 0.5  # positive-ish


def pytorch_ref(k, beta, chunk_size, output_dtype):
    """Pure PyTorch reference for chunk_scaled_dot_kkt_fwd_kernel (no gate)."""
    b_, t_, hg_, k_ = k.shape
    h_ = beta.shape[-1]
    rep = h_ // hg_
    bt = chunk_size

    kf = k.float()
    betaf = beta.float()

    out = torch.zeros(b_, t_, h_, bt, dtype=torch.float32, device=k.device)
    strict_lower = torch.tril(
        torch.ones(bt, bt, dtype=torch.bool, device=k.device), diagonal=-1
    )

    for bi in range(b_):
        for hh in range(h_):
            hg_idx = hh // rep
            Kb = kf[bi, :, hg_idx, :]  # [T, K]
            betab = betaf[bi, :, hh]  # [T]

            KKt = Kb @ Kb.transpose(0, 1)  # [T, T]
            A = KKt * betab[:, None]
            A = torch.where(strict_lower, A, torch.zeros_like(A))
            out[bi, :, hh, :] = A

    return out.to(output_dtype)


def kernel_impl(k, beta, chunk_size, output_dtype):
    return chunk_scaled_dot_kkt_fwd(
        k, g=None, beta=beta, chunk_size=chunk_size, output_dtype=output_dtype
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
        ref_out = pytorch_ref(KEY, BETA, CHUNK_SIZE, OUTPUT_DTYPE)
        kernel_out = kernel_impl(KEY, BETA, CHUNK_SIZE, OUTPUT_DTYPE)

        ref_cpu = ref_out.cpu().float()
        kernel_cpu = kernel_out.cpu().float()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-6)).mean().item()

        stats = {
            "input_shapes": {"k": tuple(KEY.shape), "beta": tuple(BETA.shape)},
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(KEY.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(KEY.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": rel_err,
        }
        pt_stats = _bench(lambda: pytorch_ref(KEY, BETA, CHUNK_SIZE, OUTPUT_DTYPE))
        kern_stats = _bench(lambda: kernel_impl(KEY, BETA, CHUNK_SIZE, OUTPUT_DTYPE))
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
            "Kernel: chunk_scaled_dot_kkt_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            for name, shape in stats["input_shapes"].items():
                lines.append(f"- {name} shape: {shape}\n")
            lines.append(f"- dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- shape: {stats['output_shape']}\n")
            lines.append(f"- dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err (mean): {stats['rel_err']}\n")
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
