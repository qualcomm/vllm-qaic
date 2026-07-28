"""
Standalone QAIC validation for `_temperature_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/gumbel.py
  - _temperature_kernel (per-request temperature scaling of logits)

For each token row, divide the logits by the per-request temperature. Rows
whose temperature is 0.0 or 1.0 are left unmodified (kernel early-returns).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.gumbel import apply_temperature

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_temperature_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/gumbel.py"
KERNEL_NAME = "_temperature_kernel"

NUM_REQS = 8
VOCAB_SIZE = 4096
DEVICE = "qaic"

torch.manual_seed(42)
LOGITS = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
# temperature per request; 0.0 and 1.0 are no-ops in the kernel.
TEMPERATURE = torch.tensor(
    [0.0, 1.0, 0.5, 2.0, 0.7, 1.5, 0.1, 4.0],
    dtype=torch.float32,
    device=DEVICE,
)


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
        f"Kernel: {KERNEL_NAME}\n",
        f"Kernel file: {KERNEL_FILE_PATH}\n",
        f"Device target: QAIC (device='{DEVICE}')\n",
        f"Status: {status}\n\n",
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
                f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")

    else:
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


def pytorch_ref(logits, expanded_idx_mapping, temperature):
    out = logits.cpu().clone()
    mapping = expanded_idx_mapping.cpu()
    temp = temperature.cpu()
    for token_idx in range(out.shape[0]):
        t = float(temp[int(mapping[token_idx].item())].item())
        if t == 0.0 or t == 1.0:
            continue
        out[token_idx] = out[token_idx] / t
    return out


def kernel_impl(logits, expanded_idx_mapping, temperature):
    logits = logits.clone()
    apply_temperature(logits, expanded_idx_mapping, temperature)
    return logits


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(LOGITS, EXPANDED_IDX_MAPPING, TEMPERATURE)
        out = kernel_impl(LOGITS, EXPANDED_IDX_MAPPING, TEMPERATURE)
        ref_cpu = ref.cpu()
        out_cpu = out.cpu()
        torch.testing.assert_close(out_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (out_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "temperature": TEMPERATURE.cpu().tolist(),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(LOGITS, EXPANDED_IDX_MAPPING, TEMPERATURE))
        kern_stats = _bench(lambda: kernel_impl(LOGITS, EXPANDED_IDX_MAPPING, TEMPERATURE))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
