"""
Standalone QAIC validation for `_mtp_shared_head_rmsnorm_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py
  - _mtp_shared_head_rmsnorm_kernel  (RMSNorm over a (T, H) bf16 tensor)
  - launcher: mtp_shared_head_rmsnorm(hidden_states, weight, eps)

This is MTP's SharedHead.norm: a plain per-row RMSNorm, computed in fp32:
    y = x * rsqrt(mean(x^2) + eps) * w
using the same _rmsnorm_row body as fused_mtp_input_rmsnorm.

Config tested: T=6 tokens, HIDDEN=256, bf16 IO.
Reference: pure PyTorch RMSNorm. Compared with assert_close (bf16 tolerance).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_mtp_shared_head_rmsnorm_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (  # noqa: E402
    mtp_shared_head_rmsnorm,
)

torch.manual_seed(42)

T = 6
HIDDEN = 256
EPS = 1e-6

# ---- Global shared inputs (used by BOTH implementations) ----
HIDDEN_STATES = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
WEIGHT = torch.randn(HIDDEN, dtype=torch.bfloat16, device=DEVICE)


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


def pytorch_ref(hidden_states, weight, eps):
    """Pure PyTorch RMSNorm in fp32."""
    x = hidden_states.cpu().float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    y = x * torch.rsqrt(var + eps) * weight.cpu().float()
    return y.to(hidden_states.dtype)


def kernel_impl(hidden_states, weight, eps):
    """Kernel launch only."""
    return mtp_shared_head_rmsnorm(hidden_states, weight, eps)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(HIDDEN_STATES, WEIGHT, EPS)
        kernel_out = kernel_impl(HIDDEN_STATES, WEIGHT, EPS)

        ref_cpu = ref_out.cpu().float()
        kernel_cpu = kernel_out.cpu().float()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-2, atol=1e-2)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(HIDDEN_STATES.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(HIDDEN_STATES.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(HIDDEN_STATES.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(HIDDEN_STATES, WEIGHT, EPS))
        kern_stats = _bench(lambda: kernel_impl(HIDDEN_STATES, WEIGHT, EPS))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _mtp_shared_head_rmsnorm_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
