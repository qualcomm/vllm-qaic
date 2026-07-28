"""
Standalone QAIC validation for `_selective_scan_fwd_kernel`.

Source under test:
vllm/_xpu_ops.py
  - _selective_scan_fwd_kernel  (via launcher `xpu_ops.selective_scan_fwd`)

Mamba selective-scan (S6) forward. Per (batch b, dim d) the kernel runs the
sequential recurrence over the sequence (group = d // dim_ngroups_ratio):
    delta = delta_in + delta_bias[d]                 (if HAS_DELTA_BIAS)
    delta = softplus(delta)                          (if delta_softplus)
    dA    = exp(delta * A[d, :])                     (elementwise over dstate)
    state = dA * state + (delta * u) * B[b, group, :, pos]
    out   = sum(state * C[b, group, :, pos]) + D[d] * u     (D optional)
    out_z = out * z * sigmoid(z)                     (if HAS_Z)
and writes the final `state` back into ssm_states[cache_slot, d, :].

`out` aliases `delta` and `out_z` aliases `z` (in-place). This differs from
the existing test_selective_scan_update_kernel.py, which validates the
single-step *update* kernel; this file validates the full forward scan, hence
the `_fwd` file name.

Config tested: NON-varlen, NO prefix-cache, NO initial state, with D and
delta_bias and delta_softplus, HAS_Z=False. batch=2, dim=4, seqlen=8,
dstate=16, n_groups=2. Both the output (mutated `delta`) and the updated
`ssm_states` are validated with rtol/atol=1e-3.

NOTE: `vllm._xpu_ops` imports the external `vllm_xpu_kernels` package at load
time; it is not needed by this kernel, so a lightweight stub is installed
before import purely to satisfy the top-level import.
"""

import datetime
import os
import sys
import traceback
import types

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_selective_scan_fwd_kernel.txt")
KERNEL_FILE_PATH = "vllm/_xpu_ops.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

# ---- Stub the optional external dependency so the module import succeeds. ----
if "vllm_xpu_kernels" not in sys.modules:
    _m = types.ModuleType("vllm_xpu_kernels")
    _fai = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")
    _fai.flash_attn_varlen_func = lambda *a, **k: None
    _m.flash_attn_interface = _fai
    sys.modules["vllm_xpu_kernels"] = _m
    sys.modules["vllm_xpu_kernels.flash_attn_interface"] = _fai

import torch  # noqa: E402

from vllm._xpu_ops import xpu_ops  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
BATCH = 2
DIM = 4
SEQLEN = 8
DSTATE = 16
N_GROUPS = 2
DIM_NGROUPS_RATIO = DIM // N_GROUPS
NULL_BLOCK_ID = -1
DELTA_SOFTPLUS = True

# Keep values modest so exp(delta*A) stays well-conditioned; A is negative
# (standard Mamba parameterization: A = -exp(...)).
U = torch.randn(BATCH, DIM, SEQLEN, dtype=torch.float32, device=DEVICE) * 0.5
DELTA_IN = torch.rand(BATCH, DIM, SEQLEN, dtype=torch.float32, device=DEVICE) * 0.5
A = -torch.rand(DIM, DSTATE, dtype=torch.float32, device=DEVICE)
B = torch.randn(BATCH, N_GROUPS, DSTATE, SEQLEN, dtype=torch.float32, device=DEVICE) * 0.3
C = torch.randn(BATCH, N_GROUPS, DSTATE, SEQLEN, dtype=torch.float32, device=DEVICE) * 0.3
D = torch.randn(DIM, dtype=torch.float32, device=DEVICE)
DELTA_BIAS = torch.rand(DIM, dtype=torch.float32, device=DEVICE) * 0.1


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


def _softplus_ref(x):
    return torch.where(x <= 20.0, torch.log(torch.exp(x) + 1.0), x)


def pytorch_ref():
    """Pure PyTorch selective scan. Returns (out, ssm_states)."""
    u = U.cpu()
    delta_in = DELTA_IN.cpu()
    a = A.cpu()
    b = B.cpu()
    c = C.cpu()
    d = D.cpu()
    delta_bias = DELTA_BIAS.cpu()

    out = torch.empty(BATCH, DIM, SEQLEN, dtype=torch.float32)
    ssm_states = torch.zeros(BATCH, DIM, DSTATE, dtype=torch.float32)
    for bi in range(BATCH):
        for di in range(DIM):
            group = di // DIM_NGROUPS_RATIO
            state = torch.zeros(DSTATE, dtype=torch.float32)
            A_vals = a[di]
            for pos in range(SEQLEN):
                delta_val = delta_in[bi, di, pos] + delta_bias[di]
                if DELTA_SOFTPLUS:
                    delta_val = _softplus_ref(delta_val)
                u_val = u[bi, di, pos]
                delta_u = delta_val * u_val
                dA = torch.exp(delta_val * A_vals)
                B_vals = b[bi, group, :, pos]
                C_vals = c[bi, group, :, pos]
                state = dA * state + delta_u * B_vals
                out_val = torch.sum(state * C_vals) + d[di] * u_val
                out[bi, di, pos] = out_val
            ssm_states[bi, di] = state
    return out, ssm_states


def kernel_impl():
    """Kernel launch only. Returns (out, ssm_states).

    `out` aliases `delta`, so we pass a fresh clone of delta each call and
    read the result back from it after the launch.
    """
    delta = DELTA_IN.clone()
    ssm_states = torch.zeros(BATCH, DIM, DSTATE, dtype=torch.float32, device=DEVICE)
    xpu_ops.selective_scan_fwd(
        U,
        delta,
        A,
        B,
        C,
        D,
        None,  # z_
        DELTA_BIAS,
        DELTA_SOFTPLUS,
        None,  # query_start_loc (non-varlen)
        None,  # cache_indices
        None,  # has_initial_state
        ssm_states,
        NULL_BLOCK_ID,
    )
    return delta, ssm_states


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_ssm = pytorch_ref()
        kern_out, kern_ssm = kernel_impl()

        ref_out_cpu = ref_out.cpu()
        kern_out_cpu = kern_out.cpu()
        ref_ssm_cpu = ref_ssm.cpu()
        kern_ssm_cpu = kern_ssm.cpu()

        torch.testing.assert_close(
            kern_out_cpu.float(), ref_out_cpu.float(), rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            kern_ssm_cpu.float(), ref_ssm_cpu.float(), rtol=1e-3, atol=1e-3
        )

        diff = (kern_out_cpu.float() - ref_out_cpu.float()).abs()
        ssm_diff = (kern_ssm_cpu.float() - ref_ssm_cpu.float()).abs()
        stats = {
            "input_shape": tuple(U.shape),
            "output_shape": tuple(kern_out.shape),
            "in_dtype": str(U.dtype),
            "out_dtype": str(kern_out.dtype),
            "device": str(U.device),
            "max_abs_diff": max(diff.max().item(), ssm_diff.max().item()),
            "mean_abs_diff": diff.mean().item(),
            "ssm_max_abs_diff": ssm_diff.max().item(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: _selective_scan_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape (u): {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- ssm_max_abs_diff: {stats['ssm_max_abs_diff']}\n",
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
