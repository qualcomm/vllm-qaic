"""
Standalone QAIC validation for `tl_rand64`.

Source under test:
vllm/v1/worker/gpu/sample/gumbel.py
  - tl_rand64 (device helper: 64-bit-precision uniform from 4x32-bit philox RNG)

`tl_rand64` is a @triton.jit *device helper* (called inside a parent kernel).
We wrap it in a tiny standalone @triton.jit kernel that (a) stores the raw
`tl.randint4x(seed, offset)` low/high 32-bit words and (b) stores the helper's
final uniform `u`.

Bit-exact reference: the helper's math is
    r = (uint64(hi) << 32) | uint64(lo)
    u = r * (1 / 2**64)                      # float64
    if not includes_zero: u = max(u, 5.0e-324-ish float64 tiny)
We reproduce this exactly in numpy using the *same* lo/hi words emitted by the
kernel (so the philox draw itself comes from Triton; only the documented
bit-combination is checked against pure Python/numpy).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import tl_rand64

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_tl_rand64.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/gumbel.py"
KERNEL_NAME = "tl_rand64"

DEVICE = "qaic"
N = 256
BLOCK = 256
SEED = 12345
INCLUDES_ZERO = False
_FP64_TINY = 2.2250738585072014e-308

torch.manual_seed(42)


@triton.jit
def _rand64_wrapper(
    out_u_ptr,
    out_lo_ptr,
    out_hi_ptr,
    seed,
    n,
    INCLUDES_ZERO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    lo, hi, _, _ = tl.randint4x(seed, offs)
    tl.store(out_lo_ptr + offs, lo, mask=mask)
    tl.store(out_hi_ptr + offs, hi, mask=mask)
    u = tl_rand64(seed, offs, includes_zero=INCLUDES_ZERO)
    tl.store(out_u_ptr + offs, u, mask=mask)


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


def pytorch_ref(lo_i32, hi_i32, includes_zero):
    """Bit-exact reproduction of tl_rand64's combination formula, using the
    lo/hi 32-bit philox words emitted by the wrapper kernel."""
    lo = lo_i32.cpu().numpy().astype(np.uint32).astype(np.uint64)
    hi = hi_i32.cpu().numpy().astype(np.uint32).astype(np.uint64)
    r = (hi << np.uint64(32)) | lo
    scale = 5.421010862427522170037e-20  # 1 / 2**64
    u = r.astype(np.float64) * scale
    if not includes_zero:
        u = np.maximum(u, _FP64_TINY)
    return torch.from_numpy(u).to(torch.float64)


def kernel_impl():
    out_u = torch.empty(N, dtype=torch.float64, device=DEVICE)
    out_lo = torch.empty(N, dtype=torch.int32, device=DEVICE)
    out_hi = torch.empty(N, dtype=torch.int32, device=DEVICE)
    _rand64_wrapper[(1,)](
        out_u,
        out_lo,
        out_hi,
        SEED,
        N,
        INCLUDES_ZERO=INCLUDES_ZERO,
        BLOCK=BLOCK,
    )
    return out_u, out_lo, out_hi


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
        out_u, out_lo, out_hi = kernel_impl()
        ref = pytorch_ref(out_lo, out_hi, INCLUDES_ZERO)
        out_cpu = out_u.cpu().to(torch.float64)

        # Structural checks: range + determinism.
        in_range = bool(((out_cpu >= 0.0) & (out_cpu < 1.0)).all())
        # Determinism: re-run and confirm identical output for fixed seed.
        out_u2, _, _ = kernel_impl()
        deterministic = bool(torch.equal(out_u.cpu(), out_u2.cpu()))

        # Bit-exact check against the documented combination formula.
        torch.testing.assert_close(out_cpu, ref, rtol=0.0, atol=0.0)
        diff = (out_cpu - ref).abs()
        assert in_range, "uniform out of [0,1) range"
        assert deterministic, "output not deterministic for fixed seed"

        stats = {
            "output_shape": tuple(out_u.shape),
            "output_dtype": str(out_u.dtype),
            "device": str(out_u.device),
            "seed": SEED,
            "includes_zero": INCLUDES_ZERO,
            "in_range_[0,1)": in_range,
            "deterministic": deterministic,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "comparison": "bit-exact combination formula (RNG sourced from kernel) "
            "+ range + determinism",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(out_lo, out_hi, INCLUDES_ZERO))
        kern_stats = _bench(lambda: kernel_impl())
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
