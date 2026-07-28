"""
Standalone QAIC validation for `_topk_log_softmax_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/logprob.py
  - _topk_log_softmax_kernel (log-softmax logprobs only at specified token ids),
    driven by the `compute_token_logprobs` launcher.

For each row: max_val = max(logits); lse = log(sum(exp(logits - max_val)));
logprob(token) = logits[token] - max_val - lse. Only the requested token_ids
per row are materialized (avoids the full [B, vocab] logprob tensor).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_topk_log_softmax_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/logprob.py"
KERNEL_NAME = "_topk_log_softmax_kernel"

DEVICE = "qaic"
BATCH_SIZE = 6
VOCAB_SIZE = 4096
NUM_LOGPROBS = 5

torch.manual_seed(42)
LOGITS = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
TOKEN_IDS = torch.randint(
    0, VOCAB_SIZE, (BATCH_SIZE, NUM_LOGPROBS), dtype=torch.int64, device=DEVICE
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


def pytorch_ref(logits, token_ids):
    x = logits.cpu().to(torch.float32)
    ids = token_ids.cpu().to(torch.int64)
    logprobs = torch.log_softmax(x, dim=-1)
    return torch.gather(logprobs, 1, ids)


def kernel_impl(logits, token_ids):
    return compute_token_logprobs(logits, token_ids)


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
        ref = pytorch_ref(LOGITS, TOKEN_IDS)
        out = kernel_impl(LOGITS, TOKEN_IDS)
        out_cpu = out.cpu().to(torch.float32)
        torch.testing.assert_close(out_cpu, ref, rtol=1e-3, atol=1e-3)
        diff = (out_cpu - ref).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "token_ids_shape": tuple(TOKEN_IDS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff / (ref.abs() + 1e-6)).mean().item(),
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(LOGITS, TOKEN_IDS))
        kern_stats = _bench(lambda: kernel_impl(LOGITS, TOKEN_IDS))
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
