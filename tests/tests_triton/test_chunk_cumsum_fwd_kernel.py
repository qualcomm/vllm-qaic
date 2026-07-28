"""
Standalone QAIC validation for `_chunk_cumsum_fwd_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py
  - _chunk_cumsum_fwd_kernel  (launched via _chunk_cumsum_fwd)

Computes, per chunk and per head, the discretized time-step dt and the
cumulative sum of the decay dA = dt * A along the time axis within the chunk.
These are the intra-chunk time-decay factors used by the SSD scan.

Exact per source (DT_SOFTPLUS=True, HAS_DT_BIAS=True):
    dt = dt + dt_bias
    dt = where(dt <= 20, softplus(dt), dt)   # softplus w/ linear branch @20
    dt = clamp(dt, dt_min, dt_max)
    dt = 0 outside the valid chunk length
    dA = dt * A                              # A is per-head scalar
    dA_cumsum = cumsum(dA, axis=time)        # within the chunk

Launcher returns (dA_cumsum, dt_out), each shaped (nheads, nchunks, chunk_size).

Reference: pure PyTorch of the above. Small: seqlen=32, chunk_size=16
(2 chunks), nheads=2. dt_limit=(0.0, inf).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_cumsum_fwd.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py"

DEVICE = "qaic"
CHUNK_SIZE = 16
SEQLEN = 32
NCHUNKS = SEQLEN // CHUNK_SIZE
NHEADS = 2
DT_SOFTPLUS = True
DT_LIMIT = (0.0, float("inf"))

torch.manual_seed(42)
DT = torch.rand(SEQLEN, NHEADS, dtype=torch.float32, device=DEVICE) + 0.1
A = -torch.rand(NHEADS, dtype=torch.float32, device=DEVICE) - 0.5  # stable decay
DT_BIAS = torch.rand(NHEADS, dtype=torch.float32, device=DEVICE) * 0.1
CU_CHUNK_SEQLENS = torch.tensor(
    [0, CHUNK_SIZE, SEQLEN], dtype=torch.int32, device=DEVICE
)


def pytorch_ref(dt, A, dt_bias):
    """Pure PyTorch per-chunk dt processing + cumulative decay sum.

    Returns (dA_cumsum, dt_out), each (nheads, nchunks, chunk_size).
    """
    dt = dt.cpu().to(torch.float32)
    A = A.cpu().to(torch.float32)
    dt_bias = dt_bias.cpu().to(torch.float32)
    cu = CU_CHUNK_SEQLENS.cpu().tolist()

    dt_out = torch.zeros(NHEADS, NCHUNKS, CHUNK_SIZE, dtype=torch.float32)
    dA_cumsum = torch.zeros(NHEADS, NCHUNKS, CHUNK_SIZE, dtype=torch.float32)

    for c in range(NCHUNKS):
        s, e = cu[c], cu[c + 1]
        n = e - s
        # dt for this chunk: (n, nheads) -> transpose to (nheads, n)
        dtc = dt[s:e, :].transpose(0, 1).clone()  # (nheads, n)
        dtc = dtc + dt_bias[:, None]
        if DT_SOFTPLUS:
            dtc = torch.nn.functional.softplus(dtc, beta=1.0, threshold=20.0)
        dtc = torch.clamp(dtc, DT_LIMIT[0], DT_LIMIT[1])
        dt_out[:, c, :n] = dtc
        dA = dtc * A[:, None]  # (nheads, n)
        dA_cumsum[:, c, :n] = torch.cumsum(dA, dim=1)
    return dA_cumsum, dt_out


def kernel_impl(dt, A, dt_bias):
    dA_cumsum, dt_out = _chunk_cumsum_fwd(
        dt,
        A,
        CHUNK_SIZE,
        CU_CHUNK_SEQLENS,
        dt_bias=dt_bias,
        dt_softplus=DT_SOFTPLUS,
        dt_limit=DT_LIMIT,
    )
    return dA_cumsum, dt_out


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
        ref_cs, ref_dt = pytorch_ref(DT, A, DT_BIAS)
        ker_cs, ker_dt = kernel_impl(DT, A, DT_BIAS)

        rcs, kcs = ref_cs, ker_cs.cpu().to(torch.float32)
        rdt, kdt = ref_dt, ker_dt.cpu().to(torch.float32)
        torch.testing.assert_close(kcs, rcs, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kdt, rdt, rtol=1e-3, atol=1e-3)

        csdiff = (kcs - rcs).abs()
        dtdiff = (kdt - rdt).abs()
        stats = {
            "dt_shape": tuple(DT.shape),
            "dA_cumsum_shape": tuple(ker_cs.shape),
            "dt_out_shape": tuple(ker_dt.shape),
            "in_dtype": str(DT.dtype),
            "out_dtype": str(ker_cs.dtype),
            "device": str(DT.device),
            "dA_cumsum_max_abs_diff": csdiff.max().item(),
            "dA_cumsum_mean_abs_diff": csdiff.mean().item(),
            "dt_out_max_abs_diff": dtdiff.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(DT, A, DT_BIAS))
        kern_stats = _bench(lambda: kernel_impl(DT, A, DT_BIAS))
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
            "Kernel: _chunk_cumsum_fwd_kernel\n",
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
