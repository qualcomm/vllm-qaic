"""
Standalone QAIC validation for `_store_quantized_value` (device helper).

Source under test:
vllm/v1/attention/ops/triton_turboquant_store.py
  - _store_quantized_value  (@triton.jit device helper)

This @triton.jit helper is not launched directly; it is invoked from inside
`_tq_fused_store_fp8` / `_tq_fused_store_mse`.  Per task instructions we wrap
it in a tiny standalone launcher kernel (`_store_value_wrapper`) that computes
the per-(token,head) slot_base into the packed TurboQuant uint8 KV cache and
calls the helper.

What it does: uniformly quantizes a value vector to VQB (3 or 4) bits via
absmax over the head dim (v_scale = (max-min)/(2^VQB-1), floored at 1e-8),
q = clip(round_half_up((v-min)/scale), 0, 2^VQB-1), packs two 4-bit codes
per byte (VQB==4) into the cache at offset KPS, and stores the fp16 scale and
zero (=val_min) as 2 bytes each right after the packed data.  We validate the
VQB==4 path.

Layout reuse: byte offsets mirror test_tq_full_dequant_kv.py's TurboQuant
value section — packed nibbles at [KPS : KPS+VAL_DATA_BYTES], fp16 scale at
[KPS+VAL_DATA_BYTES : +2], fp16 zero at [+2 : +4].

QUANT kernel: packed value bytes + scale/zero bytes compared EXACTLY against
a NumPy replica of the same quantization+packing; also cross-checked by
dequantizing (q*scale+zero) to rtol/atol=1e-2.

Prior-round note: on real QAIC hardware the sibling store kernels
(_tq_fused_store_*) were found to miscompile the dynamic bit-shift packing
(scale=0 / zero=inf); this file is COMPILE-ONLY per directive.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_store_quantized_value.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_turboquant_store.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.ops.triton_turboquant_store import (  # noqa: E402
    _store_quantized_value,
)

DEVICE = "qaic"
torch.manual_seed(42)
np.random.seed(42)

D = 16
NH = 4  # num (token*head) vectors
VQB = 4
VAL_DATA_BYTES = math.ceil(D * VQB / 8)  # 8
KPS = math.ceil(D * VQB / 8) + 2  # mirror KEY_PACKED_SIZE layout (=10)
PADDED_SLOT = KPS + VAL_DATA_BYTES + 4  # value data + fp16 scale/zero
BLOCK_D = triton.next_power_of_2(D)
BLOCK_VAL = triton.next_power_of_2(VAL_DATA_BYTES)
BLOCK_GRP = triton.next_power_of_2(D // 8) if D >= 8 else 1

VALUE = torch.randn(NH, D, dtype=torch.float32, device=DEVICE)


@triton.jit
def _store_value_wrapper(
    Value_ptr,
    KV_cache_ptr,
    D: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    PADDED_SLOT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_VAL: tl.constexpr,
    BLOCK_GRP: tl.constexpr,
):
    """Minimal launcher: one program per vector, slot_base = pid*PADDED_SLOT."""
    pid = tl.program_id(0)
    base = pid * D
    slot_base = pid * PADDED_SLOT
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    _store_quantized_value(
        Value_ptr,
        KV_cache_ptr,
        base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
    )


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


def pytorch_ref(value):
    """NumPy replica of the 4-bit uniform quant + pack + scale/zero store."""
    v = value.cpu().numpy().astype(np.float32)  # [NH, D]
    val_min = v.min(axis=1, keepdims=True)
    val_max = v.max(axis=1, keepdims=True)
    v_scale = (val_max - val_min) / (2**VQB - 1)
    v_scale = np.where(v_scale > 1e-8, v_scale, 1e-8)
    # round-half-up == floor(x + 0.5) for non-negative x
    q = np.clip(np.floor((v - val_min) / v_scale + 0.5), 0, 2**VQB - 1).astype(
        np.uint8
    )
    q_pairs = q.reshape(NH, D // 2, 2)
    packed = ((q_pairs[..., 0] & 0xF) | ((q_pairs[..., 1] & 0xF) << 4)).astype(
        np.uint8
    )  # [NH, VAL_DATA_BYTES]

    scale_f16 = v_scale.astype(np.float16).reshape(NH)
    zero_f16 = val_min.astype(np.float16).reshape(NH)
    scale_u16 = scale_f16.view(np.uint16)
    zero_u16 = zero_f16.view(np.uint16)

    cache = np.zeros((NH, PADDED_SLOT), dtype=np.uint8)
    cache[:, KPS:KPS + VAL_DATA_BYTES] = packed
    sc = KPS + VAL_DATA_BYTES
    cache[:, sc] = (scale_u16 & 0xFF).astype(np.uint8)
    cache[:, sc + 1] = ((scale_u16 >> 8) & 0xFF).astype(np.uint8)
    cache[:, sc + 2] = (zero_u16 & 0xFF).astype(np.uint8)
    cache[:, sc + 3] = ((zero_u16 >> 8) & 0xFF).astype(np.uint8)
    return torch.from_numpy(cache), q.astype(np.float32), v_scale, val_min


def kernel_impl(value):
    """Launch only: writes into a freshly-allocated packed uint8 cache."""
    cache = torch.zeros(NH, PADDED_SLOT, dtype=torch.uint8, device=DEVICE)
    _store_value_wrapper[(NH,)](
        value,
        cache.view(-1),
        D=D,
        KPS=KPS,
        VQB=VQB,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        PADDED_SLOT=PADDED_SLOT,
        BLOCK_D=BLOCK_D,
        BLOCK_VAL=BLOCK_VAL,
        BLOCK_GRP=BLOCK_GRP,
        num_warps=1,
        num_stages=1,
    )
    return cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_cache, q_ref, scale_ref, zero_ref = pytorch_ref(VALUE)
        ker_cache = kernel_impl(VALUE).cpu()

        # Exact packed-byte comparison (packed nibbles + fp16 scale/zero)
        torch.testing.assert_close(ker_cache, ref_cache, rtol=0, atol=0)

        # Cross-check dequantized value: q * scale + zero
        deq_ref = q_ref * scale_ref + zero_ref  # [NH, D]
        # Decode kernel-written cache the same way for a dequant sanity check
        ker_np = ker_cache.numpy()
        packed = ker_np[:, KPS:KPS + VAL_DATA_BYTES]
        lo = (packed & 0xF).astype(np.float32)
        hi = ((packed >> 4) & 0xF).astype(np.float32)
        q_dec = np.empty((NH, D), dtype=np.float32)
        q_dec[:, 0::2] = lo
        q_dec[:, 1::2] = hi
        sc = KPS + VAL_DATA_BYTES
        scale_dec = (
            ker_np[:, sc].astype(np.uint16) | (ker_np[:, sc + 1].astype(np.uint16) << 8)
        ).view(np.float16).astype(np.float32)
        zero_dec = (
            ker_np[:, sc + 2].astype(np.uint16)
            | (ker_np[:, sc + 3].astype(np.uint16) << 8)
        ).view(np.float16).astype(np.float32)
        deq_ker = q_dec * scale_dec[:, None] + zero_dec[:, None]
        deq_diff = np.abs(deq_ker - deq_ref).max()
        assert deq_diff <= 1e-2, f"dequant diff {deq_diff}"

        stats = {
            "value_shape": tuple(VALUE.shape),
            "cache_shape": tuple(ker_cache.shape),
            "device": DEVICE,
            "packed_exact": bool(torch.equal(ker_cache, ref_cache)),
            "deq_max_diff": float(deq_diff),
        }
        pt_stats = _bench(lambda: pytorch_ref(VALUE))
        kern_stats = _bench(lambda: kernel_impl(VALUE))
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
            "Kernel: _store_quantized_value (device helper, wrapped)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: VQB=4 uniform quant; packed bytes EXACT, dequant "
            "cross-check rtol/atol=1e-2; TQ value layout reused.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- value shape: {stats['value_shape']} (D={D}, VQB={VQB})\n",
                f"- device: {stats['device']}\n\n",
                "Output (packed uint8 cache):\n",
                f"- cache shape: {stats['cache_shape']}\n",
                f"- packed bytes exact: {stats['packed_exact']}\n",
                f"- dequant max_abs_diff: {stats['deq_max_diff']}\n",
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
