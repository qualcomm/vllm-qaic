"""
Standalone QAIC validation for `_gumbel_sample_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/gumbel.py
  - _gumbel_sample_kernel (per-vocab-block local argmax/max after
    temperature + Gumbel perturbation), driven by the `gumbel_sample` launcher.

The launcher computes per-block (local_max, local_argmax), then reduces across
blocks to a single sampled token id per row.

RNG caveat: Triton's philox Gumbel noise is impractical to reproduce bit-exact
in PyTorch, so the *correctness* comparison uses temperature == 0.0 (greedy).
With temp == 0 the kernel skips temperature scaling and Gumbel noise, so
`gumbel_sample` reduces to plain global argmax, which PyTorch reproduces
exactly. We additionally verify the temp > 0 sampling path is deterministic for
a fixed seed and yields valid in-range token ids.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_gumbel_sample_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/gumbel.py"
KERNEL_NAME = "_gumbel_sample_kernel"

DEVICE = "qaic"
NUM_TOKENS = 6
VOCAB_SIZE = 4096  # multiple blocks (BLOCK_SIZE=1024 in launcher)

torch.manual_seed(42)
LOGITS = torch.randn(NUM_TOKENS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
SEEDS = torch.arange(1, NUM_TOKENS + 1, dtype=torch.int32, device=DEVICE)
POS = torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
TEMP_GREEDY = torch.zeros(NUM_TOKENS, dtype=torch.float32, device=DEVICE)
TEMP_GUMBEL = torch.full((NUM_TOKENS,), 0.8, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(logits):
    """Greedy (temp==0) global argmax per row."""
    return torch.argmax(logits.cpu(), dim=-1).to(torch.int64)


def kernel_impl(temp):
    return gumbel_sample(
        LOGITS,
        EXPANDED_IDX_MAPPING,
        temp,
        SEEDS,
        POS,
        apply_temperature=True,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(LOGITS)
        out = kernel_impl(TEMP_GREEDY)
        out_cpu = out.cpu().to(torch.int64)

        mismatch = int((out_cpu != ref).sum().item())
        assert mismatch == 0, f"greedy sampled-id mismatch count={mismatch}"

        # Gumbel path: determinism + valid range.
        g1 = kernel_impl(TEMP_GUMBEL)
        g2 = kernel_impl(TEMP_GUMBEL)
        deterministic = bool(torch.equal(g1.cpu(), g2.cpu()))
        in_range = bool(((g1.cpu() >= 0) & (g1.cpu() < VOCAB_SIZE)).all())
        assert deterministic, "gumbel sampling not deterministic for fixed seed"
        assert in_range, "sampled token id out of range"

        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "greedy_mismatch_count": mismatch,
            "greedy_max_abs_diff": 0,
            "gumbel_deterministic": deterministic,
            "gumbel_id_in_range": in_range,
            "comparison": "temp==0 exact argmax; temp>0 determinism+range",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(LOGITS))
        kern_stats = _bench(lambda: kernel_impl(TEMP_GREEDY))
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
