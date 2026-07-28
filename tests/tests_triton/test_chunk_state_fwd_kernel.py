"""
Standalone QAIC validation for `_chunk_state_fwd_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py
  - _chunk_state_fwd_kernel  (launched via _chunk_state_fwd)

Computes the per-chunk SSM states by contracting x with B, weighting each time
step by the decay-adjusted dt. For each chunk c, head h:
    scale_t = exp(min(dA_cumsum_last - dA_cumsum_t, 0)) * dt_t
    state[c, h] = sum_t (x_t[:, None] * scale_t) * B_t[None, :]
i.e. an outer-product accumulation of x (headdim) with B (dstate), decay-
weighted so later-in-chunk tokens keep more weight. This produces the
intra-chunk state contribution consumed by the inter-chunk state-passing step.

Shapes: x (seqlen, nheads, headdim); B (seqlen, ngroups, dstate);
dt, dA_cumsum (nheads, nchunks, chunk_size). Output states
(nchunks, nheads, headdim, dstate).

Small: seqlen=32, chunk_size=16 (2 chunks), nheads=2, ngroups=1, headdim=8,
dstate=8.

Reference: pure PyTorch of the decay-weighted outer-product accumulation.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import (
    _chunk_cumsum_fwd,
    _chunk_state_fwd,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_state_fwd.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py"

DEVICE = "qaic"
CHUNK_SIZE = 16
SEQLEN = 32
NCHUNKS = SEQLEN // CHUNK_SIZE
NHEADS = 2
NGROUPS = 1
HEADDIM = 8
DSTATE = 8

torch.manual_seed(42)
X = torch.randn(SEQLEN, NHEADS, HEADDIM, dtype=torch.float32, device=DEVICE)
B = torch.randn(SEQLEN, NGROUPS, DSTATE, dtype=torch.float32, device=DEVICE)
DT_RAW = torch.rand(SEQLEN, NHEADS, dtype=torch.float32, device=DEVICE) + 0.1
A = -torch.rand(NHEADS, dtype=torch.float32, device=DEVICE) - 0.5
CU_CHUNK_SEQLENS = torch.tensor(
    [0, CHUNK_SIZE, SEQLEN], dtype=torch.int32, device=DEVICE
)

# Produce dt (dt_out) and dA_cumsum via the (separately validated) cumsum step.
DA_CUMSUM, DT = _chunk_cumsum_fwd(
    DT_RAW, A, CHUNK_SIZE, CU_CHUNK_SEQLENS, dt_bias=None, dt_softplus=False
)


def pytorch_ref(x, B, dt, dA_cumsum):
    """Pure PyTorch per-chunk decay-weighted state accumulation.

    state[c, h] = sum_t (x[t, h] outer B[t, g]) * (exp(min(dA_last - dA_t, 0)) * dt_t)
    Output: (nchunks, nheads, headdim, dstate).
    """
    x = x.cpu().to(torch.float32)
    B = B.cpu().to(torch.float32)
    dt = dt.cpu().to(torch.float32)          # (nheads, nchunks, chunk_size)
    dA_cumsum = dA_cumsum.cpu().to(torch.float32)
    cu = CU_CHUNK_SEQLENS.cpu().tolist()
    ratio = NHEADS // NGROUPS

    states = torch.zeros(
        NCHUNKS, NHEADS, HEADDIM, DSTATE, dtype=torch.float32
    )
    for c in range(NCHUNKS):
        s, e = cu[c], cu[c + 1]
        n = e - s
        for h in range(NHEADS):
            g = h // ratio
            dA_last = dA_cumsum[h, c, CHUNK_SIZE - 1]
            acc = torch.zeros(HEADDIM, DSTATE, dtype=torch.float32)
            for ti in range(n):
                dA_t = dA_cumsum[h, c, ti]
                scale = torch.exp(torch.clamp(dA_last - dA_t, max=0.0)) * dt[h, c, ti]
                acc += torch.outer(x[s + ti, h, :], B[s + ti, g, :]) * scale
            states[c, h] = acc
    return states


def kernel_impl(x, B, dt, dA_cumsum):
    return _chunk_state_fwd(
        B, x, dt, dA_cumsum, CU_CHUNK_SEQLENS, states_in_fp32=True
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
        ref_out = pytorch_ref(X, B, DT, DA_CUMSUM)
        kernel_out = kernel_impl(X, B, DT, DA_CUMSUM)
        ref_cpu = ref_out
        ker_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "x_shape": tuple(X.shape),
            "B_shape": tuple(B.shape),
            "states_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, B, DT, DA_CUMSUM))
        kern_stats = _bench(lambda: kernel_impl(X, B, DT, DA_CUMSUM))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _chunk_state_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n"
                )
        else:
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
