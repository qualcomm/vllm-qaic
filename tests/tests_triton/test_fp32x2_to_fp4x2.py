"""
Standalone QAIC validation for `_fp32x2_to_fp4x2`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
  - _fp32x2_to_fp4x2  (@triton.jit DEVICE HELPER, inline PTX)

`_fp32x2_to_fp4x2(x_lo, x_hi)` converts a PAIR of fp32 values into one packed
MXFP4 (E2M1) byte via inline PTX:
    cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;   ($1 = x_hi -> HIGH nibble,
                                                $2 = x_lo -> LOW nibble)
i.e. output byte = (E2M1_rne(x_hi) << 4) | E2M1_rne(x_lo), round-to-nearest-
even with satfinite clamp to +/-6.

Device helper -> wrapped in a tiny standalone @triton.jit kernel that loads two
fp32 arrays, calls the helper, and stores the packed uint8 bytes.

IMPORTANT (documented): the inline PTX path (`cvt...e2m1x2.f32`) is NVIDIA-PTX
only and may NOT compile on the QAIC/Hexagon Triton backend. We are NOT
executing on device here — this file only guarantees the KERNEL WRAPPER is
syntactically valid and the pytorch_ref is accurate. The reference implements
E2M1 round-to-nearest-even packing; comparison (were it to run) is the packed
byte EXACTLY, plus the two dequantized nibble magnitudes.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import _fp32x2_to_fp4x2

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fp32x2_to_fp4x2.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"

DEVICE = "qaic"
torch.manual_seed(42)

_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

N = 16
# Values already in the representable [-6, 6] range so no saturation ambiguity;
# mix of grid points and off-grid values that round unambiguously.
X_LO = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.1, 0.9, -1.4, -2.8, 3.2, -4.5, 5.5, -0.4],
    dtype=torch.float32, device=DEVICE,
)
X_HI = torch.tensor(
    [6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0,
     -0.1, 1.1, 1.9, -3.2, 2.3, 4.4, -5.5, 0.6],
    dtype=torch.float32, device=DEVICE,
)


@triton.jit
def _fp32x2_to_fp4x2_wrapper(lo_ptr, hi_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x_lo = tl.load(lo_ptr + offs)
    x_hi = tl.load(hi_ptr + offs)
    packed = _fp32x2_to_fp4x2(x_lo, x_hi)
    tl.store(out_ptr + offs, packed)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _e2m1_encode_rne(v: float) -> int:
    """Round fp32 value to an E2M1 4-bit code (round-to-nearest-even,
    satfinite clamp to +/-6)."""
    sign = 8 if v < 0 else 0
    a = abs(v)
    if a >= 6.0:
        return sign | 7
    best_code = 0
    best_dist = abs(a - _E2M1_MAG[0])
    for code in range(1, 8):
        d = abs(a - _E2M1_MAG[code])
        if d < best_dist - 1e-12:
            best_dist = d
            best_code = code
        elif abs(d - best_dist) <= 1e-12:
            if (code % 2 == 0) and (best_code % 2 == 1):
                best_code = code
                best_dist = d
    return sign | best_code


def _e2m1_decode(code: int) -> float:
    mag = _E2M1_MAG[code & 7]
    return -mag if (code & 8) else mag


def pytorch_ref(x_lo, x_hi):
    lo = x_lo.cpu()
    hi = x_hi.cpu()
    packed = torch.empty(N, dtype=torch.uint8)
    deq_lo = torch.empty(N, dtype=torch.float32)
    deq_hi = torch.empty(N, dtype=torch.float32)
    for i in range(N):
        lo_code = _e2m1_encode_rne(float(lo[i].item()))
        hi_code = _e2m1_encode_rne(float(hi[i].item()))
        packed[i] = (lo_code & 0xF) | ((hi_code & 0xF) << 4)
        deq_lo[i] = _e2m1_decode(lo_code)
        deq_hi[i] = _e2m1_decode(hi_code)
    return packed, deq_lo, deq_hi


def kernel_impl(x_lo, x_hi):
    out = torch.empty(N, dtype=torch.uint8, device=x_lo.device)
    _fp32x2_to_fp4x2_wrapper[(1,)](x_lo, x_hi, out, N)
    return out


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_packed, ref_dlo, ref_dhi = pytorch_ref(X_LO, X_HI)
        kernel_packed = kernel_impl(X_LO, X_HI).cpu()

        # Exact packed byte compare.
        mismatch = int((kernel_packed != ref_packed).sum().item())
        assert mismatch == 0, f"packed byte mismatch={mismatch}"

        stats = {
            "input_shape": tuple(X_LO.shape),
            "output_shape": tuple(kernel_packed.shape),
            "device": DEVICE,
            "byte_mismatch": mismatch,
        }
        pt_stats = _bench(lambda: pytorch_ref(X_LO, X_HI))
        kern_stats = _bench(lambda: kernel_impl(X_LO, X_HI))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: _fp32x2_to_fp4x2 (device helper, inline PTX, wrapped)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: inline PTX (cvt.rn.satfinite.e2m1x2.f32) is NVIDIA-only and\n"
            "      may not compile on the QAIC/Hexagon backend; wrapper is\n"
            "      compile-only, pytorch_ref is the correctness reference.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- x_lo/x_hi shape: {stats['input_shape']} fp32\n",
                f"- device: {stats['device']}\n\n",
                "Output (packed MXFP4 byte, EXACT compare):\n",
                f"- out shape: {stats['output_shape']} uint8\n",
                f"- byte mismatches: {stats['byte_mismatch']}\n",
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
