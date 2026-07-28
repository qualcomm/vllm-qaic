"""
Standalone QAIC validation for `_quantize_mxfp4_pair`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py
  - _quantize_mxfp4_pair  (@triton.jit DEVICE HELPER)
  - (calls _fp32x2_to_fp4x2 -> inline PTX)

`_quantize_mxfp4_pair(x_lo, x_hi)` quantizes a MXFP4_BLOCK_SIZE(=32)-element
block, supplied as two interleaved halves (x_lo = even-index values, x_hi =
odd-index values), to packed MXFP4 with a per-block UE8M0 scale:
    amax   = max(|x_lo|, |x_hi|), clamped to >= 6 * 2^-126
    log2r  = clamp(ceil(log2(amax / 6.0)), -127, 127)
    scale  = 2^log2r ;  ue8m0 = (log2r + 127) as uint8
    packed = _fp32x2_to_fp4x2(x_lo/scale, x_hi/scale)  (E2M1 rne, 2 nibbles/byte)
Returns (packed[HALF_BLOCK] uint8, ue8m0 scalar uint8).

Device helper -> wrapped in a tiny standalone @triton.jit kernel that loads the
two halves, calls the helper, and stores packed bytes + the ue8m0 scale.

FLOAT/quant kernel. Comparison: UE8M0 scale byte EXACT + DEQUANTIZED values
(E2M1 magnitude * 2^(ue8m0-127)) at rtol/atol=1e-3. Reuses the E2M1 rne
pattern from test_fused_indexer_q_rope_mxfp4_kernel.py. NOTE the helper calls
inline PTX (`_fp32x2_to_fp4x2`) which may not compile on QAIC/Hexagon; this
file is compile-only for the wrapper and accurate for pytorch_ref.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
    _quantize_mxfp4_pair,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_quantize_mxfp4_pair.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"

DEVICE = "qaic"
torch.manual_seed(42)

_E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

BLOCK = MXFP4_BLOCK_SIZE       # 32
HALF = BLOCK // 2              # 16

X_LO = torch.randn(HALF, dtype=torch.float32, device=DEVICE)
X_HI = torch.randn(HALF, dtype=torch.float32, device=DEVICE)


@triton.jit
def _quantize_mxfp4_pair_wrapper(
    lo_ptr, hi_ptr, packed_ptr, scale_ptr, HALF: tl.constexpr
):
    offs = tl.arange(0, HALF)
    x_lo = tl.load(lo_ptr + offs)
    x_hi = tl.load(hi_ptr + offs)
    packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
    tl.store(packed_ptr + offs, packed)
    tl.store(scale_ptr, ue8m0)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


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


def _e2m1_encode_rne(v: float) -> int:
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
    amax = torch.maximum(lo.abs().max(), hi.abs().max())
    amax = torch.maximum(amax, torch.tensor(6.0 * (2.0 ** -126)))
    log2r = torch.ceil(torch.log2(amax * (1.0 / 6.0)))
    log2r = torch.clamp(log2r, -127.0, 127.0)
    scale = float(torch.exp2(log2r).item())
    ue8m0 = int((log2r + 127.0).item()) & 0xFF
    inv = 1.0 / scale

    packed = torch.empty(HALF, dtype=torch.uint8)
    deq_lo = torch.empty(HALF, dtype=torch.float32)
    deq_hi = torch.empty(HALF, dtype=torch.float32)
    for i in range(HALF):
        lo_code = _e2m1_encode_rne(float(lo[i].item()) * inv)
        hi_code = _e2m1_encode_rne(float(hi[i].item()) * inv)
        packed[i] = (lo_code & 0xF) | ((hi_code & 0xF) << 4)
        deq_lo[i] = _e2m1_decode(lo_code) * scale
        deq_hi[i] = _e2m1_decode(hi_code) * scale
    return packed, ue8m0, deq_lo, deq_hi


def kernel_impl(x_lo, x_hi):
    packed = torch.empty(HALF, dtype=torch.uint8, device=x_lo.device)
    scale = torch.empty(1, dtype=torch.uint8, device=x_lo.device)
    _quantize_mxfp4_pair_wrapper[(1,)](x_lo, x_hi, packed, scale, HALF)
    return packed, scale


def _dequant(packed, ue8m0):
    scl = 2.0 ** (int(ue8m0) - 127)
    deq_lo = torch.empty(HALF, dtype=torch.float32)
    deq_hi = torch.empty(HALF, dtype=torch.float32)
    for i in range(HALF):
        byte = int(packed[i].item())
        deq_lo[i] = _e2m1_decode(byte & 0xF) * scl
        deq_hi[i] = _e2m1_decode((byte >> 4) & 0xF) * scl
    return deq_lo, deq_hi


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_packed, ref_ue8, ref_dlo, ref_dhi = pytorch_ref(X_LO, X_HI)
        k_packed, k_scale = kernel_impl(X_LO, X_HI)
        k_packed = k_packed.cpu()
        k_ue8 = int(k_scale.cpu()[0].item())

        # (a) UE8M0 scale byte EXACT.
        assert k_ue8 == ref_ue8, f"ue8m0 mismatch: {k_ue8} vs {ref_ue8}"

        # (b) dequantized values from kernel packing.
        k_dlo, k_dhi = _dequant(k_packed, k_ue8)
        torch.testing.assert_close(k_dlo, ref_dlo, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_dhi, ref_dhi, rtol=1e-3, atol=1e-3)

        diff = torch.cat([(k_dlo - ref_dlo).abs(), (k_dhi - ref_dhi).abs()])
        stats = {
            "input_shape": tuple(X_LO.shape),
            "packed_shape": tuple(k_packed.shape),
            "device": DEVICE,
            "ue8m0": k_ue8,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
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
            "Kernel: _quantize_mxfp4_pair (device helper, wrapped)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: calls inline-PTX _fp32x2_to_fp4x2 (NVIDIA-only); wrapper is\n"
            "      compile-only, pytorch_ref is the correctness reference.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- x_lo/x_hi shape: {stats['input_shape']} fp32 (block={BLOCK})\n",
                f"- device: {stats['device']}\n\n",
                "Output (UE8M0 scale EXACT + dequant rtol/atol=1e-3):\n",
                f"- packed shape: {stats['packed_shape']} uint8\n",
                f"- ue8m0 scale byte: {stats['ue8m0']}\n",
                f"- dequant max_abs_diff: {stats['max_abs_diff']}\n",
                f"- dequant mean_abs_diff: {stats['mean_abs_diff']}\n",
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
