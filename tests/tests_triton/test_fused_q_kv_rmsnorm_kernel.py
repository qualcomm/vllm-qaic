"""
Standalone QAIC validation for `_fused_q_kv_rmsnorm_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py
  - _fused_q_kv_rmsnorm_kernel  (per-row RMSNorm applied independently to a Q
    tensor and a KV tensor, fused into one launch via grid-y task id). Launched
    via the public wrapper `fused_q_kv_rmsnorm`.

Exact source math (per row, all in fp32, single cast at store):
    variance = mean(x^2)
    rrms     = 1 / sqrt(variance + eps)
    y        = x * rrms * weight
Task 0 normalizes Q rows (width Q_SIZE), task 1 normalizes KV rows (width
KV_SIZE), each with its own weight vector.

Config tested: num_tokens=16, q_size=192, kv_size=128, eps=1e-6, fp32.
Reference: pure PyTorch RMSNorm computed in fp32.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_q_kv_rmsnorm_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm import (  # noqa: E402
    fused_q_kv_rmsnorm,
)

torch.manual_seed(42)

NUM_TOKENS = 16
Q_SIZE = 192
KV_SIZE = 128
EPS = 1e-6

QR = torch.randn(NUM_TOKENS, Q_SIZE, dtype=torch.float32, device=DEVICE)
KV = torch.randn(NUM_TOKENS, KV_SIZE, dtype=torch.float32, device=DEVICE)
Q_WEIGHT = torch.randn(Q_SIZE, dtype=torch.float32, device=DEVICE)
KV_WEIGHT = torch.randn(KV_SIZE, dtype=torch.float32, device=DEVICE)


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


def _rmsnorm(x, w, eps):
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(var + eps)
    return (x * rrms * w.float())


def pytorch_ref(qr, kv, q_weight, kv_weight, eps):
    """Pure PyTorch RMSNorm for both Q and KV rows."""
    return _rmsnorm(qr, q_weight, eps), _rmsnorm(kv, kv_weight, eps)


def kernel_impl(qr, kv, q_weight, kv_weight, eps):
    """Kernel launch only."""
    return fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_q, ref_kv = pytorch_ref(QR, KV, Q_WEIGHT, KV_WEIGHT, EPS)
        ker_q, ker_kv = kernel_impl(QR, KV, Q_WEIGHT, KV_WEIGHT, EPS)

        ref_q_c, ref_kv_c = ref_q.cpu(), ref_kv.cpu()
        ker_q_c, ker_kv_c = ker_q.cpu(), ker_kv.cpu()
        torch.testing.assert_close(ker_q_c, ref_q_c, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_kv_c, ref_kv_c, rtol=1e-3, atol=1e-3)

        diff_q = (ker_q_c - ref_q_c).abs()
        diff_kv = (ker_kv_c - ref_kv_c).abs()
        max_abs = max(diff_q.max().item(), diff_kv.max().item())
        mean_abs = (diff_q.mean().item() + diff_kv.mean().item()) / 2.0
        stats = {
            "input_shape": tuple(QR.shape),
            "output_shape": tuple(ker_q.shape),
            "in_dtype": str(QR.dtype),
            "out_dtype": str(ker_q.dtype),
            "device": str(QR.device),
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
        }

        pt_stats = _bench(lambda: pytorch_ref(QR, KV, Q_WEIGHT, KV_WEIGHT, EPS))
        kern_stats = _bench(lambda: kernel_impl(QR, KV, Q_WEIGHT, KV_WEIGHT, EPS))
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
            "Kernel: _fused_q_kv_rmsnorm_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- qr shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- q out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
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
