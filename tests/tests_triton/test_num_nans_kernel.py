"""
Standalone QAIC validation for `_num_nans_kernel`.

Source under test:
vllm/v1/worker/gpu/metrics/logits.py
  - _num_nans_kernel  (via launcher `get_num_nans`)

Per-request count of NaN entries in a [num_reqs, vocab_size] logits tensor.
The kernel iterates the vocab dim in BLOCK_SIZE chunks, casts to fp32, tests
libdevice.isnan, and sums the mask into an int32 count per request.

Config tested: [num_reqs=4, vocab_size=6000] float32 logits with a known
number of NaNs scattered per row. Output is an integer count -> EXACT integer
equality against a pure-PyTorch `torch.isnan(...).sum(dim=1)` reference.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_num_nans_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/metrics/logits.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.metrics.logits import get_num_nans  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 4
VOCAB_SIZE = 6000
LOGITS = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
# Inject a known, distinct number of NaNs per row.
_nan_counts = [0, 1, 5, 17]
for _r, _c in enumerate(_nan_counts):
    if _c:
        LOGITS[_r, :_c] = float("nan")


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


def pytorch_ref(logits):
    """Pure PyTorch per-row NaN count."""
    return torch.isnan(logits).sum(dim=1).to(torch.int32)


def kernel_impl(logits):
    """Kernel launch only."""
    return get_num_nans(logits)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(LOGITS)
        kernel_out = kernel_impl(LOGITS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), (
            f"ref={ref_cpu.tolist()} k={kernel_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(LOGITS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(LOGITS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "num_nans": kernel_cpu.tolist(),
        }

        pt_stats = _bench(lambda: pytorch_ref(LOGITS))
        kern_stats = _bench(lambda: kernel_impl(LOGITS))
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
            "Kernel: _num_nans_kernel\n",
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
                f"- num_nans: {stats['num_nans']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact-match comparison)\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
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
