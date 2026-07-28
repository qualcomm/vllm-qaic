"""
Standalone QAIC validation for `reshape_and_cache_shuffle_kernel`.

Source under test:
vllm/v1/attention/backends/rocm_aiter_fa.py
  - reshape_and_cache_shuffle_kernel  (scatter-writes per-token K/V vectors into
    a *shuffled* paged KV cache layout used by the AITER flash-attention path)
  - reshape_and_cache_shuffle_triton  (launcher)

The shuffled cache layouts are:
  K cache: [num_blocks, num_kv_heads, head_size // x, block_size, x]
  V cache: [num_blocks, num_kv_heads, block_size // x, head_size, x]
where x = 16 // element_size (=4 for fp32). For a token at slot_id ->
(block_id, block_offset) and head element d in [0, head_size):
  K flat write idx (within block/head) = (d // x)*block_size*x
                                        + block_offset*x + (d % x)
  V flat write idx (within block/head) = (block_offset // x)*head_size*x
                                        + d*x + (block_offset % x)

IMPORTANT (why the kernel is re-declared here):
  This kernel and its launcher live inside an `if current_platform.is_rocm():`
  block in the source, so they are not bound on this QAIC/non-ROCm platform and
  cannot be imported. Per the device-helper pattern, we re-declare the
  `@triton.jit` kernel VERBATIM from source and drive it via a launch-only
  wrapper. The kernel body is unchanged from vllm.

Config tested: QUANT=False (kv_cache_dtype='auto'); a plain byte-copy scatter,
so cache contents are compared for EXACT float equality.
Reference: pure PyTorch replication of the shuffled-layout index math.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_reshape_and_cache_shuffle_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/rocm_aiter_fa.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_TOKENS = 6
NUM_KV_HEADS = 2
HEAD_SIZE = 16
BLOCK_SIZE = 8
NUM_BLOCKS = 3
DTYPE = torch.float32
X = 16 // torch.tensor([], dtype=DTYPE).element_size()  # =4 for fp32

KEY = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
VALUE = torch.randn(NUM_TOKENS, NUM_KV_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE)
# Distinct slots; include a padding token (-1) that must be skipped.
SLOT_MAPPING = torch.tensor(
    [0, 5, 8, 17, -1, 23], dtype=torch.int32, device=DEVICE
)
K_SCALES = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
V_SCALES = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)

K_CACHE_SHAPE = (NUM_BLOCKS, NUM_KV_HEADS, HEAD_SIZE // X, BLOCK_SIZE, X)
V_CACHE_SHAPE = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE // X, HEAD_SIZE, X)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


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


# ---- @triton.jit kernel re-declared VERBATIM from rocm_aiter_fa.py ----
@triton.jit
def reshape_and_cache_shuffle_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    slot_mapping_ptr,  # [num_tokens]
    k_scale_ptr,  # [num_blocks, num_kv_heads, block_size]
    v_scale_ptr,  # [num_blocks, num_kv_heads, block_size]
    x,
    k_stride0,
    v_stride0,
    k_cache_block_stride,
    v_cache_block_stride,
    block_size,
    head_size,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    QUANT: tl.constexpr,
):
    tid = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    src_offset_k = tid * k_stride0 + head_id * head_size
    src_offset_v = tid * v_stride0 + head_id * head_size
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    k_dst_offset = block_id * k_cache_block_stride + head_id * head_size * block_size
    dst_k_shuffle_offset = (
        k_dst_offset + offset // x * block_size * x + block_offset * x + offset % x
    )
    v_dst_offset = block_id * v_cache_block_stride + head_id * head_size * block_size
    dst_v_shuffle_offset = (
        v_dst_offset
        + block_offset // x * head_size * x
        + offset * x
        + block_offset % x
    )
    k_val = tl.load(key_ptr + src_offset_k + offset)
    v_val = tl.load(value_ptr + src_offset_v + offset)
    if QUANT:
        k_scale = 1.0
        v_scale = 1.0
        k_dtype = key_cache_ptr.type.element_ty
        v_dtype = value_cache_ptr.type.element_ty
        k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
        v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)
    tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
    tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)


def pytorch_ref():
    """Pure PyTorch scatter into the shuffled paged cache (QUANT=False)."""
    key_c = KEY.cpu()
    value_c = VALUE.cpu()
    slot_c = SLOT_MAPPING.cpu()
    key_cache = torch.zeros(K_CACHE_SHAPE, dtype=DTYPE)
    value_cache = torch.zeros(V_CACHE_SHAPE, dtype=DTYPE)
    x = X
    for tid in range(NUM_TOKENS):
        slot = int(slot_c[tid].item())
        if slot < 0:
            continue
        blk = slot // BLOCK_SIZE
        boff = slot % BLOCK_SIZE
        for h in range(NUM_KV_HEADS):
            for d in range(HEAD_SIZE):
                key_cache[blk, h, d // x, boff, d % x] = key_c[tid, h, d]
                value_cache[blk, h, boff // x, d, boff % x] = value_c[tid, h, d]
    return key_cache, value_cache


def kernel_impl():
    """Launch only: scatter into freshly-allocated shuffled caches."""
    key_cache = torch.zeros(K_CACHE_SHAPE, dtype=DTYPE, device=DEVICE)
    value_cache = torch.zeros(V_CACHE_SHAPE, dtype=DTYPE, device=DEVICE)
    grid = (NUM_TOKENS, NUM_KV_HEADS)
    reshape_and_cache_shuffle_kernel[grid](
        KEY,
        VALUE,
        key_cache,
        value_cache,
        SLOT_MAPPING,
        K_SCALES,
        V_SCALES,
        X,
        KEY.stride(0),
        VALUE.stride(0),
        key_cache.stride(0),
        value_cache.stride(0),
        BLOCK_SIZE,
        HEAD_SIZE,
        NUM_KV_HEADS,
        BLOCK_SIZE=HEAD_SIZE,
        QUANT=False,
    )
    return key_cache, value_cache


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_k, ref_v = pytorch_ref()
        ker_k, ker_v = kernel_impl()

        ker_k_c = ker_k.cpu()
        ker_v_c = ker_v.cpu()
        torch.testing.assert_close(ker_k_c, ref_k, rtol=0, atol=0)
        torch.testing.assert_close(ker_v_c, ref_v, rtol=0, atol=0)

        diff = (ker_k_c - ref_k).abs()
        stats = {
            "input_shape": tuple(KEY.shape),
            "output_shape": tuple(ker_k.shape),
            "in_dtype": str(KEY.dtype),
            "out_dtype": str(ker_k.dtype),
            "device": str(KEY.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup

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
            "Kernel: reshape_and_cache_shuffle_kernel (QUANT=False)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
