"""
Standalone QAIC validation for `_reshape_cache_per_token_head`.

Source under test:
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
  - _reshape_cache_per_token_head  (@triton.jit)
  - triton_reshape_and_cache_flash_per_token_head_quant  (public launcher)

Grid is (num_tokens, num_kv_heads); each program handles one (token, head):
loads that head's K (or V) vector, computes scale = absmax / QUANT_MAX
(floored at 1e-6), quantizes q = clamp(x / scale, QUANT_MIN, QUANT_MAX), and
scatter-writes the quantized data into the paged cache plus the per-head
float32 scale into a separate scale cache.  We exercise the int8 path
(QUANT_MAX=127, QUANT_MIN=-128), avoiding in-kernel FP8 casts unsupported on
this QAIC/Hexagon backend.

QUANT kernel: dequantized cache (q_int8 * scale) compared at rtol/atol=1e-2;
per-head scales compared at rtol/atol=1e-3.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_reshape_cache_per_token_head.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash_per_token_head_quant,
)

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 6
NUM_KV_HEADS = 2
HEAD_SIZE = 16
HEAD_SIZE_V = 16
BLOCK_SIZE = 8
NUM_BLOCKS = 3
QUANT_MAX = 127.0
QUANT_MIN = -128.0

KEY = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
VALUE = torch.randn(
    NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE_V, dtype=torch.float32, device=DEVICE
)
SLOT_MAPPING = torch.tensor(
    [0, 5, 8, 17, -1, 23], dtype=torch.int32, device=DEVICE
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


def _quant_ref(x_head):
    """Per-head absmax quantization matching the kernel exactly."""
    scale = max(float(x_head.abs().max().item()) / QUANT_MAX, 1e-6)
    q = torch.clamp(x_head * (1.0 / scale), QUANT_MIN, QUANT_MAX)
    q_int8 = q.to(torch.int8)  # truncation toward zero, matching tl.store cast
    return q_int8, scale


def pytorch_ref(key, value, slot_mapping):
    """Per-(token,head) quantize + scatter into paged int8 caches."""
    key_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.int8
    )
    value_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE_V, dtype=torch.int8
    )
    k_scale_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, dtype=torch.float32
    )
    v_scale_cache = torch.zeros_like(k_scale_cache)
    key_c = key.cpu()
    value_c = value.cpu()
    slot_c = slot_mapping.cpu()
    for tok in range(NUM_TOKENS):
        slot = int(slot_c[tok].item())
        if slot < 0:
            continue
        blk = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        for head in range(NUM_KV_HEADS):
            kq, ksc = _quant_ref(key_c[tok, head])
            key_cache[blk, off, head] = kq
            k_scale_cache[blk, off, head] = ksc
            vq, vsc = _quant_ref(value_c[tok, head])
            value_cache[blk, off, head] = vq
            v_scale_cache[blk, off, head] = vsc
    return key_cache, value_cache, k_scale_cache, v_scale_cache


def kernel_impl(key, value, slot_mapping):
    """Launch only."""
    key_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE,
        dtype=torch.int8, device=DEVICE,
    )
    value_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE_V,
        dtype=torch.int8, device=DEVICE,
    )
    k_scale_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, dtype=torch.float32, device=DEVICE
    )
    v_scale_cache = torch.zeros_like(k_scale_cache)
    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )
    return key_cache, value_cache, k_scale_cache, v_scale_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(KEY, VALUE, SLOT_MAPPING)
        ker = kernel_impl(KEY, VALUE, SLOT_MAPPING)
        ref_kc, ref_vc, ref_ks, ref_vs = ref
        ker_kc, ker_vc, ker_ks, ker_vs = (t.cpu() for t in ker)

        # Compare per-head scales (tight)
        torch.testing.assert_close(ker_ks, ref_ks, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_vs, ref_vs, rtol=1e-3, atol=1e-3)

        # Dequantize and compare (only where a scale was written)
        ker_k_deq = ker_kc.to(torch.float32) * ker_ks.unsqueeze(-1)
        ref_k_deq = ref_kc.to(torch.float32) * ref_ks.unsqueeze(-1)
        ker_v_deq = ker_vc.to(torch.float32) * ker_vs.unsqueeze(-1)
        ref_v_deq = ref_vc.to(torch.float32) * ref_vs.unsqueeze(-1)
        torch.testing.assert_close(ker_k_deq, ref_k_deq, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(ker_v_deq, ref_v_deq, rtol=1e-2, atol=1e-2)

        k_diff = (ker_k_deq - ref_k_deq).abs()
        v_diff = (ker_v_deq - ref_v_deq).abs()
        stats = {
            "key_shape": tuple(KEY.shape),
            "cache_shape": tuple(ker_kc.shape),
            "device": DEVICE,
            "k_deq_max_diff": k_diff.max().item(),
            "v_deq_max_diff": v_diff.max().item(),
            "scale_max_diff": (ker_ks - ref_ks).abs().max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(KEY, VALUE, SLOT_MAPPING))
        kern_stats = _bench(lambda: kernel_impl(KEY, VALUE, SLOT_MAPPING))
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
            "Kernel: _reshape_cache_per_token_head\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: int8 per-(token,head) absmax quant; dequant compare "
            "rtol/atol=1e-2, scales 1e-3.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- key/value shape: {stats['key_shape']}\n",
                f"- slot_mapping: {SLOT_MAPPING.cpu().tolist()}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- int8 cache shape: {stats['cache_shape']}\n",
                f"- K dequant max_abs_diff: {stats['k_deq_max_diff']}\n",
                f"- V dequant max_abs_diff: {stats['v_deq_max_diff']}\n",
                f"- scale max_abs_diff: {stats['scale_max_diff']}\n",
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
