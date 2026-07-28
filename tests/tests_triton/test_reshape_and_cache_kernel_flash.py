"""
Standalone QAIC validation for `reshape_and_cache_kernel_flash`.

Source under test:
vllm/v1/attention/ops/triton_reshape_and_cache_flash.py
  - reshape_and_cache_kernel_flash  (@triton.jit)
  - triton_reshape_and_cache_flash  (public launcher)

The kernel scatter-writes new per-token K/V vectors into a flash-style paged
KV cache at the positions given by `slot_mapping` (slot -> block/offset).  It
supports an optional in-kernel FP8 quantization path and two cache layouts:
the standard 4D layout [num_blocks, block_size, num_heads, head_size] and a
5D "head-major" layout.  We validate the NON-FP8 standard-layout path (the
common bf16/fp16/fp32 auto path) here, which is a pure gather-free scatter of
the source K/V into the paged cache.  FP8 in-kernel casts (tl.float8*) are not
supported on this QAIC/Hexagon Triton backend, so we use kv_cache_dtype="auto".

FLOAT byte-copy kernel: cache contents compared for EXACT equality (the
non-FP8 path performs a plain copy, no arithmetic).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_reshape_and_cache_kernel_flash.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_reshape_and_cache_flash.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash,
)

DEVICE = "qaic"
torch.manual_seed(42)

# Small shapes
NUM_TOKENS = 6
NUM_HEADS = 2
HEAD_SIZE = 16
BLOCK_SIZE = 8
NUM_BLOCKS = 3
DTYPE = torch.float32

KEY = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
VALUE = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
# Distinct slots across blocks; include a padding token (-1) that must be
# ignored by the kernel.
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
    """Pure PyTorch scatter into the paged cache (non-FP8 path).

    key_cache/value_cache: [num_blocks, block_size, num_heads, head_size].
    For each non-padding token, write its K/V vector at (block, offset)
    derived from the flat slot index.
    """
    key_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, dtype=DTYPE
    )
    value_cache = torch.zeros_like(key_cache)
    key_c = key.cpu()
    value_c = value.cpu()
    slot_c = slot_mapping.cpu()
    for tok in range(key_c.shape[0]):
        slot = int(slot_c[tok].item())
        if slot < 0:
            continue
        blk = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        key_cache[blk, off] = key_c[tok]
        value_cache[blk, off] = value_c[tok]
    return key_cache, value_cache


def kernel_impl(key, value, slot_mapping):
    """Launch only: writes into freshly-allocated paged caches."""
    key_cache = torch.zeros(
        NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE
    )
    value_cache = torch.zeros_like(key_cache)
    triton_reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        "auto",
        K_SCALE,
        V_SCALE,
    )
    return key_cache, value_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_k, ref_v = pytorch_ref(KEY, VALUE, SLOT_MAPPING)
        ker_k, ker_v = kernel_impl(KEY, VALUE, SLOT_MAPPING)

        ker_k_c = ker_k.cpu()
        ker_v_c = ker_v.cpu()

        torch.testing.assert_close(ker_k_c, ref_k, rtol=0, atol=0)
        torch.testing.assert_close(ker_v_c, ref_v, rtol=0, atol=0)

        stats = {
            "key_shape": tuple(KEY.shape),
            "cache_shape": tuple(ker_k.shape),
            "dtype": str(DTYPE),
            "device": DEVICE,
            "k_exact": bool(torch.equal(ker_k_c, ref_k)),
            "v_exact": bool(torch.equal(ker_v_c, ref_v)),
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
            "Kernel: reshape_and_cache_kernel_flash\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            "Note: non-FP8 standard 4D layout (kv_cache_dtype='auto'); pure\n"
            "      byte-copy scatter, EXACT equality check.\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- key/value shape: {stats['key_shape']} {stats['dtype']}\n",
                f"- slot_mapping: {SLOT_MAPPING.cpu().tolist()}\n",
                f"- device: {stats['device']}\n\n",
                "Output (paged cache, exact equality):\n",
                f"- cache shape: {stats['cache_shape']}\n",
                f"- key cache exact match: {stats['k_exact']}\n",
                f"- value cache exact match: {stats['v_exact']}\n",
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
