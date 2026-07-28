"""
Standalone QAIC validation for `_unpack_seq_triton_kernel`.

Source under test:
vllm/v1/attention/ops/common.py
  - _unpack_seq_triton_kernel  (launched via `unpack_seq_triton`)

Inverse of `_pack_seq_kernel`: unpacks a padded batched tensor `packed`
[B, Lmax, D] back to a ragged tensor `out` [N, D] (N = sum(lengths)). For each
batch b it gathers the first lengths[b] valid time rows packed[b, :lengths[b]]
into out[cumlen[b] : cumlen[b] + lengths[b]].

Launched via the public `unpack_seq_triton(packed, lengths)` wrapper. We also
verify the pack->unpack round trip reproduces the original ragged tensor.
Float compare (rtol/atol=1e-3).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_unpack_seq_triton_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/common.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
LENGTHS_LIST = [3, 5, 2, 4]
B = len(LENGTHS_LIST)
D = 8
N = sum(LENGTHS_LIST)
LMAX = max(LENGTHS_LIST)

# Original ragged tensor; PACKED is produced from it so unpack can round-trip.
X = torch.randn(N, D, dtype=torch.float32, device=DEVICE)
LENGTHS = torch.tensor(LENGTHS_LIST, dtype=torch.int32, device=DEVICE)
PACKED = pack_seq_triton(X, LENGTHS, pad_value=0.0)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(packed, lengths, n, d):
    packed = packed.cpu()
    lengths = lengths.cpu()
    out = torch.empty(n, d, dtype=packed.dtype)
    start = 0
    for b in range(lengths.shape[0]):
        ln = int(lengths[b].item())
        out[start:start + ln] = packed[b, :ln]
        start += ln
    return out


def kernel_impl(packed, lengths):
    return unpack_seq_triton(packed, lengths)


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
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(PACKED, LENGTHS, N, D)
        ker = kernel_impl(PACKED, LENGTHS).cpu()
        torch.testing.assert_close(ker, ref, rtol=1e-3, atol=1e-3)
        # Round-trip: unpack(pack(x)) == x.
        torch.testing.assert_close(ker, X.cpu(), rtol=1e-3, atol=1e-3)
        diff = (ker - ref).abs()
        stats = {
            "packed_shape": tuple(PACKED.shape),
            "out_shape": tuple(ker.shape),
            "dtype": str(X.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "roundtrip": True,
            "grid": f"({B}, cdiv(Lmax,64), cdiv(D,64))",
        }
        pt_stats = _bench(lambda: pytorch_ref(PACKED, LENGTHS, N, D))
        kern_stats = _bench(lambda: kernel_impl(PACKED, LENGTHS))
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
        lines = [
            f"{timestamp}\n",
            "Kernel: _unpack_seq_triton_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"packed shape: {stats['packed_shape']}  out shape: {stats['out_shape']}\n")
            lines.append(f"dtype: {stats['dtype']}  device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"roundtrip(pack->unpack==x): {stats['roundtrip']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"mean_abs_diff: {stats['mean_abs_diff']}\n")
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
    main()
