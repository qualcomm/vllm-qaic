"""
Standalone QAIC validation for `layer_norm_gated_fwd_kernel1`.

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - layer_norm_gated_fwd_kernel1  (@triton.jit, one-row-per-program variant)
  - layer_norm_gated_fwd          (launcher; D > 512 dispatches to this kernel)

Same gated (RMS/layer) norm math as layer_norm_gated_fwd_kernel, but this
variant handles a single row per program (grid = (T,)) instead of a block of
rows via block pointers. The launcher selects it when D > 512.

For the config exercised here (is_rms_norm=True, weight given, bias=None,
residual=None, activation="swish"):
    rstd  = 1 / sqrt(mean(x^2, dim=-1) + eps)
    x_hat = x * rstd
    y     = x_hat * weight
    y     = y * g * sigmoid(g)          # swish/silu output gate
We pass out_dtype so the launcher writes a fresh output tensor (the default
path would overwrite x in place).

Reference: pure PyTorch replication of the above.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_layer_norm_gated_fwd_kernel1.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.fla.ops.kda import (  # noqa: E402
    layer_norm_gated_fwd,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
# D > 512 so the launcher selects layer_norm_gated_fwd_kernel1 (grid=(T,)).
T = 32
D = 1024
EPS = 1e-5
ACTIVATION = "swish"
X = torch.randn(T, D, dtype=torch.float32, device=DEVICE)
G = torch.randn(T, D, dtype=torch.float32, device=DEVICE)
WEIGHT = torch.randn(D, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(x, g, weight, eps, activation):
    """Pure PyTorch gated RMSNorm with a swish output gate."""
    xf = x.to(torch.float32)
    gf = g.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = xf * rstd
    y = x_hat * weight.to(torch.float32)
    if activation in ("swish", "silu"):
        y = y * gf * torch.sigmoid(gf)
    elif activation == "sigmoid":
        y = y * torch.sigmoid(gf)
    return y.to(x.dtype)


def kernel_impl(x, g, weight, eps, activation):
    y, _mean, _rstd, _res = layer_norm_gated_fwd(
        x=x,
        g=g,
        weight=weight,
        bias=None,
        activation=activation,
        eps=eps,
        residual=None,
        out_dtype=x.dtype,
        is_rms_norm=True,
    )
    return y


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X, G, WEIGHT, EPS, ACTIVATION)
        kernel_out = kernel_impl(X, G, WEIGHT, EPS, ACTIVATION)

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

        pt_stats = _bench(lambda: pytorch_ref(X, G, WEIGHT, EPS, ACTIVATION))
        kern_stats = _bench(lambda: kernel_impl(X, G, WEIGHT, EPS, ACTIVATION))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: layer_norm_gated_fwd_kernel1 (D>512 variant)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- x shape: {stats['input_shape']}, D={D}, eps={EPS}\n")
            lines.append(f"- g shape: {tuple(G.shape)}, weight shape: {tuple(WEIGHT.shape)}\n")
            lines.append(f"- activation: {ACTIVATION}, is_rms_norm=True\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- max_rel_err: {stats['rel_err']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
