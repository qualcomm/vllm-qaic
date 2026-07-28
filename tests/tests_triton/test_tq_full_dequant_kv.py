"""
Standalone QAIC validation for `_tq_full_dequant_kv`.

Source under test:
vllm/v1/attention/ops/triton_turboquant_decode.py
  - _tq_full_dequant_kv  (@triton.jit)

Fully dequantizes TurboQuant-packed K and V cache entries back to FP16 (used
for reference / bulk-GEMM decode). There is no dedicated public launcher for
this kernel in the source, so we replicate its launch site directly (grid =
(max_seq, B * NUM_KV_HEADS); constants from `_get_layout`).

We use the MSE-centroid KEY path (KEY_FP8=0) to match the sibling test
test_tq_decode_stage1.py — in-kernel FP8 casts (tl.float8e4nv/e4b15) are not
supported on this QAIC/Hexagon backend. K dequant (MSE path):
    mse_idx  = unpack 4-bit centroid code (2 codes/byte, low then high)
    k_recon  = centroids[mse_idx] * vec_norm         (fp16 norm at MSE_BYTES)
V dequant (uniform 4-bit):
    v_idx    = unpack 4-bit code (2/byte)
    v_recon  = v_idx * v_scale + v_zero              (fp16 scale/zero after
                                                      VAL_DATA_BYTES)

The packed kv_cache buffer is built in pure NumPy/PyTorch, BIT-FOR-BIT
identical to the TurboQuant layout used by test_tq_decode_stage1.py's
`_pack_turboquant_kv_cache_numpy` (this is data prep, not a kernel call).

FLOAT/4-bit kernel: dequantized K and V compared against a pure-PyTorch
reference at rtol/atol=1e-2 (fp16). Documented layout reuse.
"""

import datetime
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import numpy as np
import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.triton_turboquant_decode import _tq_full_dequant_kv

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_tq_full_dequant_kv.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_turboquant_decode.py"

DEVICE = "qaic"
torch.manual_seed(42)
np.random.seed(42)

D = 16
SEQ_LEN = 16
HK = 1
BATCH = 1
MSE_BITS = 4
VALUE_QUANT_BITS = 4
N_CENTROIDS = 2 ** MSE_BITS
MSE_BYTES = math.ceil(D * MSE_BITS / 8)
KEY_PACKED_SIZE = MSE_BYTES + 2                 # mse codes + fp16 norm
VAL_DATA_BYTES = math.ceil(D * VALUE_QUANT_BITS / 8)
PADDED_SLOT = KEY_PACKED_SIZE + VAL_DATA_BYTES + 4  # + fp16 scale/zero
BLOCK_SIZE = SEQ_LEN
NUM_BLOCKS = 1
BLOCK_D = triton.next_power_of_2(D)

K_RAW_CPU = torch.randn(SEQ_LEN, HK, D, dtype=torch.float32)
V_RAW_CPU = torch.randn(SEQ_LEN, HK, D, dtype=torch.float32)
CENTROIDS_CPU = torch.linspace(-1.5, 1.5, N_CENTROIDS, dtype=torch.float32)


def _pack_turboquant_kv_cache_numpy(k_raw, v_raw, centroids):
    """Pure NumPy bit-packing into the TurboQuant MSE kv_cache uint8 layout.
    Byte-for-byte identical to test_tq_decode_stage1.py's packing helper
    (mirrors triton_turboquant_store.py). Data prep only — no kernel call."""
    centroids_np = centroids.numpy()
    k_np = k_raw.numpy()
    norms = np.linalg.norm(k_np, axis=-1, keepdims=True)
    x_hat = k_np / (norms + 1e-8)
    mse_idx = np.abs(
        x_hat[..., None] - centroids_np[None, None, None, :]
    ).argmin(axis=-1).astype(np.uint8)

    idx_pairs = mse_idx.reshape(SEQ_LEN, HK, D // 2, 2)
    mse_packed = (
        (idx_pairs[..., 0] & 0xF) | ((idx_pairs[..., 1] & 0xF) << 4)
    ).astype(np.uint8)
    norm_f16 = norms.astype(np.float16).reshape(SEQ_LEN, HK)
    norm_u16 = norm_f16.view(np.uint16)

    v_np = v_raw.numpy()
    val_min = v_np.min(axis=-1, keepdims=True)
    val_max = v_np.max(axis=-1, keepdims=True)
    v_scale = np.clip((val_max - val_min) / 15.0, 1e-8, None)
    q_all = np.clip(np.round((v_np - val_min) / v_scale), 0, 15).astype(np.uint8)
    q_pairs = q_all.reshape(SEQ_LEN, HK, D // 2, 2)
    val_packed = (
        (q_pairs[..., 0] & 0xF) | ((q_pairs[..., 1] & 0xF) << 4)
    ).astype(np.uint8)
    scale_f16 = v_scale.astype(np.float16).reshape(SEQ_LEN, HK)
    zero_f16 = val_min.astype(np.float16).reshape(SEQ_LEN, HK)
    scale_u16 = scale_f16.view(np.uint16)
    zero_u16 = zero_f16.view(np.uint16)

    kv_np = np.zeros((NUM_BLOCKS, BLOCK_SIZE, HK, PADDED_SLOT), dtype=np.uint8)
    kv_np[0, :, :, 0:MSE_BYTES] = mse_packed
    kv_np[0, :, :, MSE_BYTES] = (norm_u16 & 0xFF).astype(np.uint8)
    kv_np[0, :, :, MSE_BYTES + 1] = ((norm_u16 >> 8) & 0xFF).astype(np.uint8)
    kv_np[0, :, :, KEY_PACKED_SIZE:KEY_PACKED_SIZE + VAL_DATA_BYTES] = val_packed
    sc_off = KEY_PACKED_SIZE + VAL_DATA_BYTES
    kv_np[0, :, :, sc_off] = (scale_u16 & 0xFF).astype(np.uint8)
    kv_np[0, :, :, sc_off + 1] = ((scale_u16 >> 8) & 0xFF).astype(np.uint8)
    kv_np[0, :, :, sc_off + 2] = (zero_u16 & 0xFF).astype(np.uint8)
    kv_np[0, :, :, sc_off + 3] = ((zero_u16 >> 8) & 0xFF).astype(np.uint8)

    return kv_np, mse_idx, norms, q_all, v_scale, val_min


(
    _KV_CACHE_NP,
    _MSE_IDX_NP,
    _NORMS_NP,
    _Q_ALL_NP,
    _V_SCALE_NP,
    _VAL_MIN_NP,
) = _pack_turboquant_kv_cache_numpy(K_RAW_CPU, V_RAW_CPU, CENTROIDS_CPU)

KV_CACHE = torch.from_numpy(_KV_CACHE_NP).to(DEVICE)
CENTROIDS = CENTROIDS_CPU.to(DEVICE)
BLOCK_TABLE = torch.zeros(BATCH, 1, dtype=torch.int32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(mse_idx, norms, q_all, v_scale, val_min, centroids):
    """K = centroids[mse_idx] * norm; V = q * scale + zero. fp16."""
    centroids_np = centroids.cpu().numpy()
    # K: [seq, Hk, D]
    k_deq = norms * centroids_np[mse_idx]           # broadcast norm over D
    v_deq = q_all.astype(np.float32) * v_scale + val_min
    # -> [B, Hk, seq, D]
    k_out = torch.from_numpy(k_deq).to(torch.float16).permute(1, 0, 2).unsqueeze(0)
    v_out = torch.from_numpy(v_deq).to(torch.float16).permute(1, 0, 2).unsqueeze(0)
    return k_out.contiguous(), v_out.contiguous()


def kernel_impl(kv_cache, block_table, centroids):
    """Replicated launch site for _tq_full_dequant_kv (no public launcher)."""
    k_out = torch.zeros(BATCH, HK, SEQ_LEN, D, dtype=torch.float16, device=DEVICE)
    v_out = torch.zeros(BATCH, HK, SEQ_LEN, D, dtype=torch.float16, device=DEVICE)
    grid = (SEQ_LEN, BATCH * HK)
    _tq_full_dequant_kv[grid](
        kv_cache,
        block_table,
        centroids,
        k_out,
        v_out,
        k_out.stride(0),
        k_out.stride(1),
        k_out.stride(2),
        v_out.stride(0),
        v_out.stride(1),
        v_out.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=D,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_KV_HEADS=HK,
        MSE_BYTES=MSE_BYTES,
        KPS=KEY_PACKED_SIZE,
        VQB=VALUE_QUANT_BITS,
        VAL_DATA_BYTES=VAL_DATA_BYTES,
        MSE_BITS=MSE_BITS,
        KEY_FP8=0,
        BLOCK_D=BLOCK_D,
        NORM_CORRECTION=0,
        FP8_E4B15=0,
        num_warps=1,
        num_stages=1,
    )
    return k_out, v_out


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
        ref_k, ref_v = pytorch_ref(
            _MSE_IDX_NP, _NORMS_NP, _Q_ALL_NP, _V_SCALE_NP, _VAL_MIN_NP, CENTROIDS
        )
        k_out, v_out = kernel_impl(KV_CACHE, BLOCK_TABLE, CENTROIDS)

        k_cpu = k_out.cpu().to(torch.float32)
        v_cpu = v_out.cpu().to(torch.float32)
        ref_k_f = ref_k.to(torch.float32)
        ref_v_f = ref_v.to(torch.float32)

        torch.testing.assert_close(k_cpu, ref_k_f, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(v_cpu, ref_v_f, rtol=1e-2, atol=1e-2)

        k_diff = (k_cpu - ref_k_f).abs()
        v_diff = (v_cpu - ref_v_f).abs()
        stats = {
            "kv_cache_shape": tuple(KV_CACHE.shape),
            "k_out_shape": tuple(k_out.shape),
            "v_out_shape": tuple(v_out.shape),
            "device": DEVICE,
            "head_dim": D,
            "seq_len": SEQ_LEN,
            "k_max_abs_diff": k_diff.max().item(),
            "k_mean_abs_diff": k_diff.mean().item(),
            "v_max_abs_diff": v_diff.max().item(),
            "v_mean_abs_diff": v_diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(
            _MSE_IDX_NP, _NORMS_NP, _Q_ALL_NP, _V_SCALE_NP, _VAL_MIN_NP, CENTROIDS
        ))
        kern_stats = _bench(lambda: kernel_impl(KV_CACHE, BLOCK_TABLE, CENTROIDS))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
        status = "SUCCESS"
        print("SUCCESS", stats)
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _tq_full_dequant_kv\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: MSE-centroid key path (KEY_FP8=0), 4-bit values; kv_cache\n"
            "      packed via pure NumPy (byte-identical to "
            "test_tq_decode_stage1.py's layout); launch site replicated (no\n"
            "      public launcher for this kernel).\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- kv_cache shape: {stats['kv_cache_shape']} uint8\n",
                f"- head_dim={stats['head_dim']}, seq_len={stats['seq_len']}, "
                f"mse_bits={MSE_BITS}, value_quant_bits={VALUE_QUANT_BITS}\n",
                f"- device: {stats['device']}\n\n",
                "Output (dequant K + V fp16, rtol/atol=1e-2):\n",
                f"- k_out shape: {stats['k_out_shape']}\n",
                f"- v_out shape: {stats['v_out_shape']}\n",
                f"- K max_abs_diff: {stats['k_max_abs_diff']}\n",
                f"- K mean_abs_diff: {stats['k_mean_abs_diff']}\n",
                f"- V max_abs_diff: {stats['v_max_abs_diff']}\n",
                f"- V mean_abs_diff: {stats['v_mean_abs_diff']}\n",
            ]
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
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
