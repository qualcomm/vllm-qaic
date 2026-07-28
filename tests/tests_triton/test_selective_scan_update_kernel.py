"""
Standalone QAIC validation for `_selective_scan_update_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/mamba_ssm.py
  - _selective_scan_update_kernel  (launched via selective_state_update)

Performs one step of the Mamba selective-scan SSM state update: it discretizes
A and B with the (optionally softplus'd, bias'd) time-step dt, advances the
recurrent state, computes the output projection with C, adds the optional D
skip connection, and applies the optional Z (SiLU) gate. The state tensor is
updated in place; the output is written to a preallocated `out` tensor.

Exact single-step recurrence (per source, TIE_HDIM=False, DT_SOFTPLUS=False):
    dt   = dt + dt_bias
    dA   = exp(A * dt[:, None])                 # (dim, dstate)
    dB   = B[None, :] * dt[:, None]             # (dim, dstate)
    state = state * dA + dB * x[:, None]
    out  = sum(state * C[None, :], axis=1)      # (dim,)
    out += x * D                                # D skip
    out *= z * sigmoid(z)                       # Z (SiLU) gate

Reference: pure PyTorch implementation of the above. We validate both the
output and the in-place-updated state. batch=2, nheads=1, dim=16, dstate=16.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_selective_scan_update.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/mamba_ssm.py"

DEVICE = "qaic"
BATCH = 2
NHEADS = 1
DIM = 16
DSTATE = 16
DT_SOFTPLUS = False

torch.manual_seed(42)
# state: (batch, dim, dstate); A negative for a stable decay.
STATE = torch.randn(BATCH, DIM, DSTATE, dtype=torch.float32, device=DEVICE)
X = torch.randn(BATCH, DIM, dtype=torch.float32, device=DEVICE)
DT = torch.rand(BATCH, DIM, dtype=torch.float32, device=DEVICE) + 0.1
A = -torch.rand(DIM, DSTATE, dtype=torch.float32, device=DEVICE) - 0.5
B = torch.randn(BATCH, DSTATE, dtype=torch.float32, device=DEVICE)
C = torch.randn(BATCH, DSTATE, dtype=torch.float32, device=DEVICE)
D = torch.randn(DIM, dtype=torch.float32, device=DEVICE)
DT_BIAS = torch.rand(DIM, dtype=torch.float32, device=DEVICE) * 0.1
Z = torch.randn(BATCH, DIM, dtype=torch.float32, device=DEVICE)


def pytorch_ref(state, x, dt, A, B, C, D, dt_bias, z):
    """Pure PyTorch single-step selective-scan recurrence.

    Returns (out, new_state).
    """
    state = state.clone().to(torch.float32)
    dt = dt.to(torch.float32) + dt_bias.to(torch.float32)[None, :]  # (B, dim)
    if DT_SOFTPLUS:
        dt = torch.nn.functional.softplus(dt, beta=1.0, threshold=20.0)
    # dA: (B, dim, dstate)
    dA = torch.exp(dt[:, :, None] * A[None, :, :])
    # dB: (B, dim, dstate) = B[b, None, dstate] * dt[b, dim, None]
    dB = B[:, None, :] * dt[:, :, None]
    new_state = state * dA + dB * x[:, :, None]
    # out: sum over dstate of state*C -> (B, dim)
    out = (new_state * C[:, None, :]).sum(dim=-1)
    out = out + x * D[None, :]
    out = out * (z * torch.sigmoid(z))
    return out, new_state


def kernel_impl(state, x, dt, A, B, C, D, dt_bias, z):
    state = state.clone()
    out = torch.empty_like(x)
    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=DT_SOFTPLUS,
        out=out,
    )
    return out, state


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


def _timing_lines(stats):
    if "pytorch_latency_ms" not in stats:
        return []
    return [
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


_TIMING_KEYS = (
    "pytorch_latency_ms",
    "kernel_latency_ms",
    "speedup_kernel_over_pytorch",
)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_state = pytorch_ref(STATE, X, DT, A, B, C, D, DT_BIAS, Z)
        ker_out, ker_state = kernel_impl(STATE, X, DT, A, B, C, D, DT_BIAS, Z)

        ro, ko = ref_out.cpu(), ker_out.cpu().to(torch.float32)
        rs, ks = ref_state.cpu(), ker_state.cpu().to(torch.float32)
        torch.testing.assert_close(ko, ro, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ks, rs, rtol=1e-3, atol=1e-3)

        odiff = (ko - ro).abs()
        sdiff = (ks - rs).abs()
        stats = {
            "state_shape": tuple(STATE.shape),
            "x_shape": tuple(X.shape),
            "out_shape": tuple(ker_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(ker_out.dtype),
            "device": str(X.device),
            "out_max_abs_diff": odiff.max().item(),
            "out_mean_abs_diff": odiff.mean().item(),
            "state_max_abs_diff": sdiff.max().item(),
            "state_mean_abs_diff": sdiff.mean().item(),
        }
        pt_stats = _bench(
            lambda: pytorch_ref(STATE, X, DT, A, B, C, D, DT_BIAS, Z)
        )
        kern_stats = _bench(
            lambda: kernel_impl(STATE, X, DT, A, B, C, D, DT_BIAS, Z)
        )
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
            "Kernel: _selective_scan_update_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                if k in _TIMING_KEYS:
                    continue
                lines.append(f"- {k}: {v}\n")
            lines += _timing_lines(stats)
        else:
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
