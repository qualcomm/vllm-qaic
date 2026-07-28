"""
Standalone QAIC validation for `_tq_fused_store_fp8`.

Source under test:
vllm/v1/attention/ops/triton_turboquant_store.py
  - _tq_fused_store_fp8  (@triton.jit)  TurboQuant FP8 store:
      casts raw keys to FP8 in-kernel (tl.float8e4nv / tl.float8e4b15) and
      scatters them, then uniform-4-bit-quantizes + packs the values with a
      trailing fp16 scale/zero (shared `_store_quantized_value` helper).
  - triton_turboquant_store  (public launcher, key_fp8=True path)

Packed byte layout (per source):
  bytes [0, D)                         : FP8 key bytes (one fp8 per element)
  bytes [KPS, KPS+VAL_DATA_BYTES)      : 4-bit value quant pairs
  bytes [KPS+VAL_DATA_BYTES, +4)       : fp16 value scale, fp16 value zero
where KPS = key_packed_size (= D for the fp8 key layout).

DOCUMENTED LIMITATION (spec edge case + test_tq_decode_stage1.py):
  This kernel performs an IN-KERNEL FP8 CAST (`k_vals.to(tl.float8e4nv)` /
  `tl.float8e4b15`). This QAIC/Hexagon Triton backend does NOT support fp8
  cast types ("type fp8e4nv not supported in this architecture"), so the
  kernel cannot be JIT-compiled/executed here, AND the source store kernels'
  dynamic-shift value bit-packing is separately documented to corrupt bytes on
  this backend. A meaningful numeric comparison of the FP8 key bytes is
  therefore not possible on this device.

  Per the task's edge-case guidance we STILL create and exercise the file:
  we build the closest faithful packed-byte reference we can (FP8 key bytes
  via a host-side float8_e4m3 cast when available, plus the exact value-quant
  layout) and attempt the real kernel. If the kernel executes we compare
  packed bytes; if it fails to compile/run (expected here due to the fp8
  cast) we FALL BACK to kernel-latency-only timing where possible and log the
  reason. No bogus reference is asserted as passing.
Reference: pure PyTorch/NumPy replication of the packed byte layout.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_tq_fused_store_fp8.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_turboquant_store.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm.v1.attention.ops.triton_turboquant_store import (  # noqa: E402
    triton_turboquant_store,
)

torch.manual_seed(42)

# ---- Global shared inputs / layout constants ----
N = 4  # tokens
H = 1  # kv heads
D = 16  # head dim
VALUE_QUANT_BITS = 4
KEY_PACKED_SIZE = D  # fp8 key: one byte per element
VAL_DATA_BYTES = math.ceil(D * VALUE_QUANT_BITS / 8)
PADDED_SLOT = KEY_PACKED_SIZE + VAL_DATA_BYTES + 4  # + fp16 scale/zero
BLOCK_SIZE = N
NUM_BLOCKS = 1

KEY = torch.randn(N, H, D, dtype=torch.float16, device=DEVICE)
VALUE = torch.randn(N, H, D, dtype=torch.float16, device=DEVICE)
SLOT_MAPPING = torch.arange(N, dtype=torch.int32, device=DEVICE)
# Unused in the fp8 path (no rotation / midpoints), but launcher signature
# requires them.
PIT = torch.eye(D, dtype=torch.float32, device=DEVICE)
MIDPOINTS = torch.zeros(1, dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns latency stats (ms)."""
    import time

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


def pytorch_ref():
    """Faithful packed-byte reference: fp8 key bytes + 4-bit value quant."""
    NH = N * H
    k_np = KEY.cpu().float().reshape(NH, D).numpy()
    v_np = VALUE.cpu().float().reshape(NH, D).numpy()

    # FP8 key bytes (e4m3 host cast as the closest available equivalent to the
    # e4nv path). May not match this backend's chosen fp8 flavour exactly.
    try:
        k_fp8 = torch.from_numpy(k_np).to(torch.float8_e4m3fn)
        key_bytes = k_fp8.view(torch.uint8).numpy().reshape(NH, D)
    except Exception:
        # No host fp8 support: leave key bytes zero (documented; comparison of
        # key section then not meaningful).
        key_bytes = np.zeros((NH, D), dtype=np.uint8)

    # value uniform 4-bit quant (bit-exact with _store_quantized_value)
    val_min = v_np.min(axis=1, keepdims=True)
    val_max = v_np.max(axis=1, keepdims=True)
    v_scale = np.clip((val_max - val_min) / 15.0, 1e-8, None)
    q_all = np.clip(np.floor((v_np - val_min) / v_scale + 0.5), 0, 15).astype(np.uint8)
    q_pairs = q_all.reshape(NH, VAL_DATA_BYTES, 2)
    val_packed = ((q_pairs[..., 0] & 0xF) | ((q_pairs[..., 1] & 0xF) << 4)).astype(
        np.uint8
    )
    scale_u16 = v_scale.astype(np.float16).reshape(NH).view(np.uint16)
    zero_u16 = val_min.astype(np.float16).reshape(NH).view(np.uint16)

    kv = np.zeros((NUM_BLOCKS, BLOCK_SIZE, H, PADDED_SLOT), dtype=np.uint8)
    sm = SLOT_MAPPING.cpu().numpy()
    for tok in range(N):
        slot = int(sm[tok])
        blk, off = slot // BLOCK_SIZE, slot % BLOCK_SIZE
        for h in range(H):
            row = tok * H + h
            kv[blk, off, h, 0:D] = key_bytes[row]
            kv[blk, off, h, KEY_PACKED_SIZE:KEY_PACKED_SIZE + VAL_DATA_BYTES] = (
                val_packed[row]
            )
            sc = KEY_PACKED_SIZE + VAL_DATA_BYTES
            kv[blk, off, h, sc] = scale_u16[row] & 0xFF
            kv[blk, off, h, sc + 1] = (scale_u16[row] >> 8) & 0xFF
            kv[blk, off, h, sc + 2] = zero_u16[row] & 0xFF
            kv[blk, off, h, sc + 3] = (zero_u16[row] >> 8) & 0xFF
    return torch.from_numpy(kv)


def kernel_impl():
    """Launch only: run the real FP8 store kernel into a fresh cache."""
    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, H, PADDED_SLOT, dtype=torch.uint8, device=DEVICE
    )
    triton_turboquant_store(
        KEY,
        VALUE,
        kv_cache,
        SLOT_MAPPING,
        PIT,
        MIDPOINTS,
        0,  # mse_bits (unused in fp8 path)
        KEY_PACKED_SIZE,
        VALUE_QUANT_BITS,
        key_fp8=True,
    )
    return kv_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kernel_ran = False
    kern_stats = None
    try:
        ref_out = pytorch_ref()

        # Attempt to exercise the real kernel (expected to fail on this
        # backend due to the unsupported in-kernel fp8 cast).
        kernel_out = kernel_impl()
        kernel_ran = True

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        diff = (kernel_cpu.int() - ref_cpu.int()).abs()
        stats = {
            "input_shape": tuple(KEY.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(KEY.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(KEY.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.float().mean().item()),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup

        assert torch.equal(kernel_cpu, ref_cpu), (
            "packed bytes differ from faithful reference "
            f"(max byte diff={stats['max_abs_diff']})."
        )
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE (expected: in-kernel fp8 cast unsupported on this backend)")
        print(error_text)
        # Kernel-latency-only fallback attempt, if the kernel is runnable.
        if kern_stats is None:
            try:
                kern_stats = _bench(kernel_impl)
            except Exception:
                kern_stats = None
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _tq_fused_store_fp8 (FP8 key + VQB=4 value quant)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Kernel executed: {kernel_ran}\n",
            "Note: in-kernel fp8 cast (tl.float8e4nv/e4b15) is unsupported on\n"
            "      this QAIC/Hexagon backend; see module docstring. Numeric\n"
            "      key-byte validation is not feasible; kernel-latency logged\n"
            "      as fallback when the kernel is runnable.\n",
            f"Status: {status}\n\n",
        ]
        if stats:
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats.get('input_shape')}\n")
            lines.append(f"- in dtype: {stats.get('in_dtype')}\n")
            lines.append(f"- device: {stats.get('device')}\n\n")
            lines.append("Output (packed uint8 KV-cache):\n")
            lines.append(f"- output shape: {stats.get('output_shape')}\n")
            lines.append(f"- out dtype: {stats.get('out_dtype')}\n")
            lines.append(f"- max_abs_diff (byte): {stats.get('max_abs_diff')}\n")
            lines.append(f"- mean_abs_diff (byte): {stats.get('mean_abs_diff')}\n")
        if kern_stats is not None:
            lines.append("Timing:\n")
            if "pytorch_latency_ms" in stats:
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
            lines.append(
                f"- Kernel latency (ms): avg={kern_stats['avg_ms']:.4f} "
                f"min={kern_stats['min_ms']:.4f} "
                f"max={kern_stats['max_ms']:.4f} "
                f"median={kern_stats['median_ms']:.4f}\n")
            if "speedup_kernel_over_pytorch" in stats:
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        if status != "SUCCESS":
            lines.append("Error / documented-limitation note:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
