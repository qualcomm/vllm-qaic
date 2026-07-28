"""
Standalone QAIC validation for `reshape_and_cache_kernel_flash_diffkv`.

Source under test:
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
  - reshape_and_cache_kernel_flash_diffkv  (@triton.jit)
  - triton_reshape_and_cache_flash_diffkv  (public launcher)

Writes K and V (which may have DIFFERENT head sizes, as in MLA) into a single
combined paged cache of shape
[num_blocks, block_size, num_heads, head_size_k + head_size_v]: per (token,
head) it lays out the K vector followed immediately by the V vector within
the slot.  Optional in-kernel FP8 quantization is supported; we validate the
NON-FP8 "auto" path (plain byte-copy scatter) since in-kernel FP8 casts are
unsupported on this QAIC/Hexagon Triton backend.

FLOAT byte-copy kernel: combined cache contents compared for EXACT equality.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_reshape_and_cache_kernel_flash_diffkv.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash_diffkv,
)

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 6
NUM_HEADS = 2
HEAD_SIZE_K = 32  # different K / V head sizes (MLA-style)
HEAD_SIZE_V = 16
BLOCK_SIZE = 8
NUM_BLOCKS = 3
DTYPE = torch.float32
COMBINED = HEAD_SIZE_K + HEAD_SIZE_V

KEY = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE_K, dtype=DTYPE, device=DEVICE)
VALUE = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE_V, dtype=DTYPE, device=DEVICE)
SLOT_MAPPING = torch.tensor(
    [0, 5, 8, 17, -1, 23], dtype=torch.int32, device=DEVICE
)
K_SCALE = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
V_SCALE = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(key, value, slot_mapping):
    """Scatter K then V into the combined slot (non-FP8 path)."""
    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, COMBINED, dtype=DTYPE
    )
    key_c = key.cpu()
    value_c = value.cpu()
    slot_c = slot_mapping.cpu()
    for tok in range(NUM_TOKENS):
        slot = int(slot_c[tok].item())
        if slot < 0:
            continue
        blk = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        for head in range(NUM_HEADS):
            kv_cache[blk, off, head, :HEAD_SIZE_K] = key_c[tok, head]
            kv_cache[blk, off, head, HEAD_SIZE_K:] = value_c[tok, head]
    return kv_cache


def kernel_impl(key, value, slot_mapping):
    """Launch only."""
    kv_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, COMBINED, dtype=DTYPE, device=DEVICE
    )
    triton_reshape_and_cache_flash_diffkv(
        key,
        value,
        kv_cache,
        slot_mapping,
        "auto",
        K_SCALE,
        V_SCALE,
    )
    return kv_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(KEY, VALUE, SLOT_MAPPING)
        ker = kernel_impl(KEY, VALUE, SLOT_MAPPING).cpu()

        torch.testing.assert_close(ker, ref, rtol=0, atol=0)

        stats = {
            "key_shape": tuple(KEY.shape),
            "value_shape": tuple(VALUE.shape),
            "cache_shape": tuple(ker.shape),
            "device": DEVICE,
            "exact": bool(torch.equal(ker, ref)),
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
            "Kernel: reshape_and_cache_kernel_flash_diffkv\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: non-FP8 'auto' path, different K/V head sizes into one\n"
            "      combined cache; EXACT equality byte-copy check.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- key shape: {stats['key_shape']} (head_size_k={HEAD_SIZE_K})\n",
                f"- value shape: {stats['value_shape']} (head_size_v={HEAD_SIZE_V})\n",
                f"- slot_mapping: {SLOT_MAPPING.cpu().tolist()}\n",
                f"- device: {stats['device']}\n\n",
                "Output (combined paged cache, exact equality):\n",
                f"- cache shape: {stats['cache_shape']}\n",
                f"- exact match: {stats['exact']}\n",
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
