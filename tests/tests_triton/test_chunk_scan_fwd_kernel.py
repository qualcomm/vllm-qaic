"""
Standalone QAIC validation for `_chunk_scan_fwd_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py
  - _chunk_scan_fwd_kernel  (launched via _chunk_scan_fwd)

This is the intra-chunk causal scan output kernel of the SSD algorithm. For
every chunk c and head h it combines three terms into the output
out[chunk_start + m, h, :] (m indexes positions within the chunk):

  1. INTER-CHUNK (previous state) term:
       acc_inter[m] = exp(dA_cumsum[h,c,m]) * (C_m @ prev_state[c-1,h]^T)
     where prev_state = states[c-1, h] for a continuing sequence, or zeros at
     the first chunk of a sequence (no initial_states supplied here).

  2. INTRA-CHUNK causal attention-like term:
       L[m,k] = cb[c,g,m,k] * exp(min(dA_cumsum_m - dA_cumsum_k, 0)) * dt[h,c,k]
       L[m,k] = 0 for k > m           (causal mask, IS_CAUSAL=True)
       acc_intra[m] = sum_k L[m,k] * x[chunk_start + k, h, :]
     `cb` is the C@B^T chunk matrix from the bmm step.

  3. D skip connection:  acc[m] += x[chunk_start + m, h, :] * D[h]

Z (SiLU) gating is SKIPPED in this test (z=None) and initial_states are NOT
used (initial_states=None) to keep the reference focused on the core term
structure; both are documented simplifications.

Shapes: cb (nchunks, ngroups, chunk_size, chunk_size); x (seqlen, nheads,
headdim); dt, dA_cumsum (nheads, nchunks, chunk_size); C (seqlen, ngroups,
dstate); states (nchunks, nheads, headdim, dstate); out (seqlen, nheads,
headdim). Small: seqlen=32, chunk_size=16 (2 chunks), nheads=2, ngroups=1,
headdim=8, dstate=8, D per-head scalar.

Reference: pure PyTorch of the three terms above.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_scan_fwd.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py"

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
C = torch.randn(SEQLEN, NGROUPS, DSTATE, dtype=torch.float32, device=DEVICE)
CB = torch.randn(
    NCHUNKS, NGROUPS, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32, device=DEVICE
)
STATES = torch.randn(
    NCHUNKS, NHEADS, HEADDIM, DSTATE, dtype=torch.float32, device=DEVICE
)
D = torch.randn(NHEADS, dtype=torch.float32, device=DEVICE)
DT_RAW = torch.rand(SEQLEN, NHEADS, dtype=torch.float32, device=DEVICE) + 0.1
A = -torch.rand(NHEADS, dtype=torch.float32, device=DEVICE) - 0.5
CU_CHUNK_SEQLENS = torch.tensor(
    [0, CHUNK_SIZE, SEQLEN], dtype=torch.int32, device=DEVICE
)
# Single sequence spanning both chunks.
SEQ_IDX = torch.zeros(NCHUNKS, dtype=torch.int32, device=DEVICE)

# Realistic dt (processed) and dA_cumsum from the (separately validated) cumsum.
DA_CUMSUM, DT = _chunk_cumsum_fwd(
    DT_RAW, A, CHUNK_SIZE, CU_CHUNK_SEQLENS, dt_bias=None, dt_softplus=False
)


def pytorch_ref(cb, x, dt, dA_cumsum, C, states, D, seq_idx):
    """Pure PyTorch intra-chunk scan output (inter + intra + D terms)."""
    cb = cb.cpu().to(torch.float32)
    x = x.cpu().to(torch.float32)
    dt = dt.cpu().to(torch.float32)              # (nheads, nchunks, csize)
    dA = dA_cumsum.cpu().to(torch.float32)       # (nheads, nchunks, csize)
    C = C.cpu().to(torch.float32)
    states = states.cpu().to(torch.float32)
    D = D.cpu().to(torch.float32)
    seq_idx = seq_idx.cpu().tolist()
    cu = CU_CHUNK_SEQLENS.cpu().tolist()
    ratio = NHEADS // NGROUPS

    out = torch.zeros(SEQLEN, NHEADS, HEADDIM, dtype=torch.float32)
    for c in range(NCHUNKS):
        s, e = cu[c], cu[c + 1]
        n = e - s
        seq_prev = seq_idx[c - 1] if c >= 1 else -1
        new_seq = seq_idx[c] != seq_prev
        for h in range(NHEADS):
            g = h // ratio
            dA_c = dA[h, c, :]        # (csize,)
            dt_c = dt[h, c, :]        # (csize,)
            scale_m = torch.exp(dA_c[:n])   # (n,)

            # 1. inter-chunk (previous state) term
            if new_seq:
                # first chunk of a sequence, no initial_states -> zeros
                acc = torch.zeros(n, HEADDIM, dtype=torch.float32)
            else:
                prev_state = states[c - 1, h]          # (headdim, dstate)
                Cm = C[s:e, g, :]                      # (n, dstate)
                inter = Cm @ prev_state.transpose(0, 1)  # (n, headdim)
                acc = inter * scale_m[:, None]

            # 2. intra-chunk causal term
            for m in range(n):
                for k in range(m + 1):  # causal: k <= m
                    decay = torch.exp(torch.clamp(dA_c[m] - dA_c[k], max=0.0))
                    coef = cb[c, g, m, k] * decay * dt_c[k]
                    acc[m] += coef * x[s + k, h, :]

            # 3. D skip connection (D per-head scalar)
            for m in range(n):
                acc[m] += x[s + m, h, :] * D[h]

            out[s:e, h, :] = acc
    return out


def kernel_impl(cb, x, dt, dA_cumsum, C, states, D, seq_idx):
    out = torch.empty(SEQLEN, NHEADS, HEADDIM, dtype=torch.float32, device=x.device)
    _chunk_scan_fwd(
        cb,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        CU_CHUNK_SEQLENS,
        out,
        seq_idx,
        D=D,
        z=None,
        initial_states=None,
    )
    return out


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
        ref_out = pytorch_ref(CB, X, DT, DA_CUMSUM, C, STATES, D, SEQ_IDX)
        kernel_out = kernel_impl(CB, X, DT, DA_CUMSUM, C, STATES, D, SEQ_IDX)
        ref_cpu = ref_out
        ker_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "x_shape": tuple(X.shape),
            "cb_shape": tuple(CB.shape),
            "out_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "notes": "Z-gating skipped, no initial_states",
        }
        pt_stats = _bench(lambda: pytorch_ref(CB, X, DT, DA_CUMSUM, C, STATES, D, SEQ_IDX))
        kern_stats = _bench(lambda: kernel_impl(CB, X, DT, DA_CUMSUM, C, STATES, D, SEQ_IDX))
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
            "Kernel: _chunk_scan_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: Z (SiLU) gating skipped, initial_states not used.\n",
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
