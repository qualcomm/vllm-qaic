"""
Standalone QAIC validation for `_softplus` (XPU ops variant).

Source under test:
vllm/_xpu_ops.py
  - _softplus(x)  device-side @triton.jit helper:
        tl.where(x <= 20.0, log(exp(x) + 1.0), x)

    i.e. numerically-stable softplus with a linear tail above x == 20.

This collides by name with the existing test_softplus.py (which tests the
mamba_ssm softplus); this file targets the vllm/_xpu_ops.py helper, hence the
`_xpu` suffix.

`_softplus` is a device helper, so we wrap it in a minimal @triton.jit
launcher that loads a tile, calls the helper, and stores the result (same
pattern as test_apply_softcap.py / test_find_seq_idx.py).

NOTE: `vllm._xpu_ops` imports the external `vllm_xpu_kernels` package at
module load. That package is not required by this helper, so we install a
lightweight stub before import purely to satisfy the top-level import.

Reference: torch.nn.functional.softplus with beta=1, threshold=20, which is
exactly `x <= 20 ? log(exp(x)+1) : x`. float32, rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback
import types

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_softplus_xpu.txt")
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

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm._xpu_ops import _softplus  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
N = 256
# Span values around and beyond the x==20 branch point.
X = torch.linspace(-30.0, 40.0, N, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(x):
    """Pure PyTorch softplus with the same threshold behaviour (beta=1, thr=20)."""
    return torch.nn.functional.softplus(x, beta=1.0, threshold=20.0)


@triton.jit
def _softplus_launcher(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    res = _softplus(x)
    tl.store(out_ptr + offs, res)


def kernel_impl(x):
    """Kernel launch only."""
    out = torch.empty_like(x)
    _softplus_launcher[(1,)](x.reshape(-1), out.reshape(-1), N=N)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X)
        kernel_out = kernel_impl(X)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(
            kernel_cpu.float(), ref_cpu.float(), rtol=1e-3, atol=1e-3
        )

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        denom = ref_cpu.float().abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(X))
        kern_stats = _bench(lambda: kernel_impl(X))
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
            "Kernel: _softplus (vllm/_xpu_ops.py device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
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
