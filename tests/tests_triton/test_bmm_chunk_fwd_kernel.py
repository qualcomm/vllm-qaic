"""
Standalone QAIC validation for `_bmm_chunk_fwd_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/ssd_bmm.py
  - _bmm_chunk_fwd_kernel  (launched via _bmm_chunk_fwd)

Performs a batched (per-chunk) matmul between two per-chunk projections,
computing out = a @ b^T for every chunk. In the SSD (state-space duality)
scan this forms the `C @ B^T` chunk matrices (the attention-like intra-chunk
score matrices). Optionally causal: when causal=True the kernel skips the
strictly-upper-triangular blocks and only the lower triangle (i >= j) is
guaranteed correct. We validate the non-causal path so the full
chunk_size x chunk_size output can be compared.

Inputs a, b: (seqlen, ngroups, k). Output: (nchunks, ngroups, chunk_size,
chunk_size). Small: 1 sequence, chunk_size=16, seqlen=32 (2 chunks), ngroups=1.

Reference: pure PyTorch per-chunk a_chunk @ b_chunk^T.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_bmm_chunk_fwd.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/ssd_bmm.py"

DEVICE = "qaic"
CHUNK_SIZE = 16
SEQLEN = 32
NCHUNKS = SEQLEN // CHUNK_SIZE
NGROUPS = 1
K = 8
CAUSAL = False

torch.manual_seed(42)
A_IN = torch.randn(SEQLEN, NGROUPS, K, dtype=torch.float32, device=DEVICE)
B_IN = torch.randn(SEQLEN, NGROUPS, K, dtype=torch.float32, device=DEVICE)
CU_CHUNK_SEQLENS = torch.tensor(
    [0, CHUNK_SIZE, SEQLEN], dtype=torch.int32, device=DEVICE
)


def pytorch_ref(a, b):
    """Pure PyTorch per-chunk C @ B^T (= a @ b^T per chunk, per group)."""
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    out = torch.zeros(
        NCHUNKS, NGROUPS, CHUNK_SIZE, CHUNK_SIZE,
        dtype=torch.float32, device="cpu",
    )
    cu = CU_CHUNK_SEQLENS.cpu().tolist()
    a_c = a.cpu()
    b_c = b.cpu()
    for c in range(NCHUNKS):
        s, e = cu[c], cu[c + 1]
        n = e - s
        for g in range(NGROUPS):
            ac = a_c[s:e, g, :]  # (n, K)
            bc = b_c[s:e, g, :]  # (n, K)
            out[c, g, :n, :n] = ac @ bc.transpose(-1, -2)
    return out


def kernel_impl(a, b):
    return _bmm_chunk_fwd(
        a, b, CHUNK_SIZE, CU_CHUNK_SEQLENS, causal=CAUSAL, output_dtype=torch.float32
    )


def _log(text):
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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(A_IN, B_IN)
        kernel_out = kernel_impl(A_IN, B_IN)
        ref_cpu = ref_out
        ker_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "a_shape": tuple(A_IN.shape),
            "b_shape": tuple(B_IN.shape),
            "out_shape": tuple(kernel_out.shape),
            "in_dtype": str(A_IN.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(A_IN.device),
            "causal": CAUSAL,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(A_IN, B_IN))
        kern_stats = _bench(lambda: kernel_impl(A_IN, B_IN))
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
            "Kernel: _bmm_chunk_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
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
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
