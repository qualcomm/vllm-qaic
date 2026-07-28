"""
Standalone QAIC validation for `_tq_fused_store_mse`.

Source under test:
vllm/v1/attention/ops/triton_turboquant_store.py
  - _tq_fused_store_mse  (@triton.jit)  fused TurboQuant MSE store:
      binary-search bucketize of rotated normalized keys against sorted
      midpoints -> MSE centroid indices, 4-bit index packing, fp16 key-norm
      store, and uniform 4-bit value quantization + packing + fp16 scale/zero.
  - triton_turboquant_store  (public launcher, key_fp8=False path)
  - _store_quantized_value   (shared value-quant device helper)

We build a bit-exact pure-PyTorch/NumPy reference of the packed KV-cache byte
layout (documented in the source):
  bytes [0, MSE_BYTES)                     : 4-bit MSE centroid index pairs
  bytes [MSE_BYTES, MSE_BYTES+2)           : fp16 key L2-norm (lo, hi)
  bytes [KPS, KPS+VAL_DATA_BYTES)          : 4-bit value quant pairs
  bytes [KPS+VAL_DATA_BYTES, +4)           : fp16 value scale, fp16 value zero
where KPS = key_packed_size = MSE_BYTES + 2, and the external pre-processing
is norms=||k||, x_hat=k/(norm+1e-8), y=x_hat@PiT, idx=searchsorted(midpoints,y).

IMPORTANT / documented limitation (see test_tq_decode_stage1.py docstring):
  On this QAIC/Hexagon Triton backend the *store* kernels' dynamic-shift
  bit-packing path is documented to emit CORRUPTED packed bytes (value
  scale=0, zero=inf) even though the unpacking side reads correct bytes back.
  We therefore treat the exact packed-byte comparison as the primary check
  but ALSO guarantee kernel-latency timing: the kernel is exercised and timed
  regardless of whether the byte comparison passes, per the task's edge-case
  guidance. If the byte comparison fails (corruption reproduced) the run is
  reported FAILURE with the diff + timing logged; if it passes it is SUCCESS.
Reference: pure PyTorch/NumPy replication of the packed byte layout.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_tq_fused_store_mse.txt")
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
MSE_BITS = 4
VALUE_QUANT_BITS = 4
N_CENTROIDS = 2**MSE_BITS
MSE_BYTES = math.ceil(D * MSE_BITS / 8)
KEY_PACKED_SIZE = MSE_BYTES + 2  # mse indices + fp16 norm
VAL_DATA_BYTES = math.ceil(D * VALUE_QUANT_BITS / 8)
PADDED_SLOT = KEY_PACKED_SIZE + VAL_DATA_BYTES + 4  # + fp16 scale/zero
BLOCK_SIZE = N
NUM_BLOCKS = 1

KEY = torch.randn(N, H, D, dtype=torch.float32, device=DEVICE)
VALUE = torch.randn(N, H, D, dtype=torch.float32, device=DEVICE)
SLOT_MAPPING = torch.arange(N, dtype=torch.int32, device=DEVICE)
# Random orthonormal-ish rotation (any [D,D] fp32 works; must match in ref).
_Q, _ = torch.linalg.qr(torch.randn(D, D, dtype=torch.float32))
PIT = _Q.contiguous().to(DEVICE)
# Sorted midpoints (N_CENTROIDS-1 values) spanning the rotated-key range.
MIDPOINTS = torch.linspace(-1.5, 1.5, N_CENTROIDS - 1, dtype=torch.float32).to(DEVICE)


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
    """Bit-exact NumPy replication of the packed KV-cache byte layout."""
    NH = N * H
    k_np = KEY.cpu().float().reshape(NH, D).numpy()
    v_np = VALUE.cpu().float().reshape(NH, D).numpy()
    pit_np = PIT.cpu().float().numpy()
    mids_np = MIDPOINTS.cpu().float().numpy()

    # --- key path: normalize + rotate + bucketize ---
    norms = np.linalg.norm(k_np, axis=1, keepdims=True)  # [NH,1]
    x_hat = k_np / (norms + 1e-8)
    y = x_hat @ pit_np  # [NH, D]
    # binary-search bucketize == searchsorted(side='right'), clipped
    idx = np.searchsorted(mids_np, y, side="right").astype(np.int64)
    idx = np.clip(idx, 0, N_CENTROIDS - 1)

    # pack 4-bit index pairs
    idx_pairs = idx.reshape(NH, D // 2, 2)
    mse_packed = ((idx_pairs[..., 0] & 0xF) | ((idx_pairs[..., 1] & 0xF) << 4)).astype(
        np.uint8
    )  # [NH, MSE_BYTES]

    norm_f16 = norms.astype(np.float16).reshape(NH)
    norm_u16 = norm_f16.view(np.uint16)

    # --- value path: uniform 4-bit quant ---
    val_min = v_np.min(axis=1, keepdims=True)
    val_max = v_np.max(axis=1, keepdims=True)
    v_scale = np.clip((val_max - val_min) / 15.0, 1e-8, None)
    # kernel uses ((v-min)/scale + 0.5).to(int32) which truncates toward zero;
    # inputs here are >= 0 so this equals floor(x + 0.5).
    q_all = np.clip(np.floor((v_np - val_min) / v_scale + 0.5), 0, 15).astype(np.uint8)
    q_pairs = q_all.reshape(NH, VAL_DATA_BYTES, 2)
    val_packed = ((q_pairs[..., 0] & 0xF) | ((q_pairs[..., 1] & 0xF) << 4)).astype(
        np.uint8
    )

    scale_f16 = v_scale.astype(np.float16).reshape(NH)
    zero_f16 = val_min.astype(np.float16).reshape(NH)
    scale_u16 = scale_f16.view(np.uint16)
    zero_u16 = zero_f16.view(np.uint16)

    # --- assemble packed cache [num_blocks, block_size, H, padded_slot] ---
    kv = np.zeros((NUM_BLOCKS, BLOCK_SIZE, H, PADDED_SLOT), dtype=np.uint8)
    sm = SLOT_MAPPING.cpu().numpy()
    for tok in range(N):
        slot = int(sm[tok])
        blk = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        for h in range(H):
            row = tok * H + h
            kv[blk, off, h, 0:MSE_BYTES] = mse_packed[row]
            kv[blk, off, h, MSE_BYTES] = norm_u16[row] & 0xFF
            kv[blk, off, h, MSE_BYTES + 1] = (norm_u16[row] >> 8) & 0xFF
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
    """Launch only: run the real MSE store kernel into a fresh cache."""
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
        MSE_BITS,
        KEY_PACKED_SIZE,
        VALUE_QUANT_BITS,
        key_fp8=False,
    )
    return kv_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    kernel_ran = False
    try:
        ref_out = pytorch_ref()

        # Exercise (and later time) the real kernel regardless of correctness.
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

        # Always record timing (kernel-latency fallback per edge-case rules).
        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup

        # Primary exact packed-byte comparison.
        assert torch.equal(kernel_cpu, ref_cpu), (
            "packed bytes differ from faithful reference "
            f"(max byte diff={stats['max_abs_diff']}); this reproduces the "
            "documented store-kernel byte-corruption on this backend."
        )
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _tq_fused_store_mse (MSE path, VQB=4)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Kernel executed: {kernel_ran}\n",
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
        if status != "SUCCESS":
            lines.append("Error / documented-limitation note:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
