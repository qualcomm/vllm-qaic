"""
Standalone QAIC validation for `_layer_norm_fwd_1pass_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/layernorm_gated.py
  - _layer_norm_fwd_1pass_kernel
    Single-pass (gated) LayerNorm / RMSNorm forward with an optional
    SiLU/swish-gated branch (Mamba2 gated norm).

Config tested (documented):
  - is_rms_norm = True            -> RMSNorm (no mean subtraction)
  - norm_before_gate = True       -> gate applied AFTER norm+weight
  - HAS_Z = True (gate present)   -> gate activation is fixed swish/SiLU:
        y = (x * rstd * w) * z * sigmoid(z)
  - HAS_BIAS = False, group_size = N (single group)

Launcher: `rms_norm_gated(x, weight, bias, z=..., eps=1e-6,
                          norm_before_gate=True)`.

Reference: pure-PyTorch RMSNorm followed by swish gating (float compare).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_layer_norm_fwd_1pass_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/layernorm_gated.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.mamba.ops.layernorm_gated import (  # noqa: E402
    rms_norm_gated,
)

torch.manual_seed(42)

# Global shared inputs
M = 16  # rows
N = 128  # feature dim
EPS = 1e-6
DEVICE = "qaic"

INPUT = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
GATE = torch.randn(M, N, dtype=torch.float32, device=DEVICE)
WEIGHT = torch.randn(N, dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
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


def pytorch_ref(x, z, weight, eps):
    """RMSNorm then swish (SiLU) output gate, norm_before_gate=True."""
    x = x.float()
    z = z.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    x_hat = x * rstd
    y = x_hat * weight.float()
    # swish gate applied after norm (norm_before_gate=True): z * sigmoid(z)
    y = y * z * torch.sigmoid(z)
    return y


def kernel_impl(x, z, weight, eps):
    return rms_norm_gated(
        x,
        weight,
        None,  # bias
        z=z,
        eps=eps,
        norm_before_gate=True,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(INPUT, GATE, WEIGHT, EPS)
        kernel_out = kernel_impl(INPUT, GATE, WEIGHT, EPS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu().float()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel = diff / (ref_cpu.abs() + 1e-6)
        stats = {
            "input_shape": tuple(INPUT.shape),
            "gate_shape": tuple(GATE.shape),
            "weight_shape": tuple(WEIGHT.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(INPUT.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(INPUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "max_rel_err": rel.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(INPUT, GATE, WEIGHT, EPS))
        kern_stats = _bench(lambda: kernel_impl(INPUT, GATE, WEIGHT, EPS))
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
            "Kernel: _layer_norm_fwd_1pass_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Config: RMSNorm + swish gate, norm_before_gate=True, no bias\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- gate shape: {stats['gate_shape']}\n")
            lines.append(f"- weight shape: {stats['weight_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- max_rel_err: {stats['max_rel_err']}\n")
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
    main()
