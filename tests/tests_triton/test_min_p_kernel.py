"""
Standalone QAIC validation for `_min_p_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/min_p.py
  - _min_p_kernel (min-p logits filtering), via the `apply_min_p` launcher.

For each token row: max_logit = max(logits); threshold = max_logit +
log(min_p); logits below threshold are set to -inf. Rows with min_p == 0.0 are
left unmodified. Float compare on finite entries + exact -inf-mask equality
(modeled on kernels_v23/test_min_p.py, same source file).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.min_p import apply_min_p

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_min_p_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/min_p.py"
KERNEL_NAME = "_min_p_kernel"

DEVICE = "qaic"
NUM_REQS = 8
VOCAB_SIZE = 4096

torch.manual_seed(42)
LOGITS = torch.randn(NUM_REQS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
MIN_P = torch.tensor(
    [0.0, 0.1, 0.2, 0.5, 0.0, 0.3, 0.05, 0.8],
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
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


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


def pytorch_ref(logits, expanded_idx_mapping, min_p):
    out = logits.cpu().clone()
    mapping = expanded_idx_mapping.cpu()
    mp = min_p.cpu()
    for token_idx in range(out.shape[0]):
        p = float(mp[int(mapping[token_idx].item())].item())
        if p == 0.0:
            continue
        row = out[token_idx]
        threshold = row.max() + torch.log(torch.tensor(p, dtype=torch.float32))
        row[row < threshold] = float("-inf")
    return out


def kernel_impl(logits, expanded_idx_mapping, min_p):
    logits = logits.clone()
    apply_min_p(logits, expanded_idx_mapping, min_p)
    return logits


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(LOGITS, EXPANDED_IDX_MAPPING, MIN_P)
        out = kernel_impl(LOGITS, EXPANDED_IDX_MAPPING, MIN_P)
        ref_cpu = ref.cpu()
        out_cpu = out.cpu()

        same_inf = torch.isinf(ref_cpu) == torch.isinf(out_cpu)
        finite_mask = ~torch.isinf(ref_cpu)
        torch.testing.assert_close(
            out_cpu[finite_mask], ref_cpu[finite_mask], rtol=1e-3, atol=1e-3
        )
        assert bool(same_inf.all()), "-inf mask mismatch"

        diff = (out_cpu[finite_mask] - ref_cpu[finite_mask]).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "min_p": MIN_P.cpu().tolist(),
            "inf_mask_match": bool(same_inf.all()),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(LOGITS, EXPANDED_IDX_MAPPING, MIN_P))
        kern_stats = _bench(lambda: kernel_impl(LOGITS, EXPANDED_IDX_MAPPING, MIN_P))
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
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
