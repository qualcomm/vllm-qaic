"""
Standalone QAIC validation for `cp_mha_gather_cache_kernel`.

Source under test:
vllm/v1/attention/backends/rocm_aiter_fa.py
  - cp_mha_gather_cache_kernel  (gathers per-token K/V vectors out of a paged
    KV cache into a contiguous [num_tokens, num_heads, head_size] layout, using
    a block table + cu_seqlens/seq_start/token_to_batch index mapping; supports
    NHD and SHUFFLE cache layouts and an optional dequant scale multiply)
  - cp_mha_gather_cache  (launcher)

IMPORTANT (why the kernel is re-declared here):
  In the vLLM source this kernel and its launcher are defined *inside* an
  `if current_platform.is_rocm():` block, so on this QAIC/non-ROCm platform
  they are never bound at module import time and cannot be imported. Per the
  device-helper pattern used by the other tests, we therefore re-declare the
  `@triton.jit` kernel VERBATIM (byte-for-byte from the source) inside this
  test and drive it through a minimal launch-only wrapper. The kernel body
  under test is unchanged from vllm.

Config tested: NHD cache layout, DEQUANT=False (plain gather, no scale). This
is a pure paged-cache gather, so the reference is an exact PyTorch gather and
the comparison is EXACT float equality.
Reference: pure PyTorch replication of the block/slot index math + gather.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_cp_mha_gather_cache_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/rocm_aiter_fa.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_HEADS = 2
HEAD_SIZE = 16
PAGE_SIZE = 4
TOTAL_TOKENS = 6
NUM_BLOCKS = 4
MAX_BLOCK_NUM = 2  # blocks per batch in the block table
DTYPE = torch.float32

KEY_CACHE = torch.randn(
    NUM_BLOCKS, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE
)
VALUE_CACHE = torch.randn(
    NUM_BLOCKS, PAGE_SIZE, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE
)
# Single batch covering all tokens.
BLOCK_TABLE = torch.tensor([[0, 1]], dtype=torch.int32, device=DEVICE)
CU_SEQLENS_KV = torch.tensor([0, TOTAL_TOKENS], dtype=torch.int32, device=DEVICE)
TOKEN_TO_BATCH = torch.zeros(TOTAL_TOKENS, dtype=torch.int32, device=DEVICE)
SEQ_STARTS = torch.zeros(1, dtype=torch.int32, device=DEVICE)
# Dummy scales (only read when DEQUANT=True).
K_SCALES = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
V_SCALES = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)


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
def cp_mha_gather_cache_kernel(
    key_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    value_cache_ptr,  # [num_blocks, page_size, num_head, head_size]
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    block_table_ptr,  # [num_batches, max_block_num]
    cu_seqlens_kv_ptr,  # [num_batches + 1]
    token_to_batch_ptr,  # [max_cum_tokens]
    seq_start_ptr,  # [num_batches]
    k_scale_ptr,  # [1] / [num_blocks, num_kv_heads, page_size]
    v_scale_ptr,
    num_heads,
    head_size,
    x,
    max_block_num,
    k_cache_stride0,
    k_cache_stride1,
    k_cache_stride2,
    v_cache_stride0,
    v_cache_stride1,
    v_cache_stride2,
    DEQUANT: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    CACHE_FORMAT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    key_ptr_offset = key_ptr + token_id * head_size * num_heads + head_id * head_size
    value_ptr_offset = (
        value_ptr + token_id * head_size * num_heads + head_id * head_size
    )
    batch_idx = tl.load(token_to_batch_ptr + token_id)
    batch_start = tl.load(seq_start_ptr + batch_idx)
    token_start = tl.load(cu_seqlens_kv_ptr + batch_idx)
    batch_offset = token_id - token_start + batch_start
    block_offset = batch_offset // PAGE_SIZE
    block_id = tl.load(
        block_table_ptr + max_block_num * batch_idx + block_offset
    ).to(tl.int64)
    slot_id = batch_offset % PAGE_SIZE

    if CACHE_FORMAT == "NHD":
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * k_cache_stride0
            + slot_id * k_cache_stride1
            + head_id * k_cache_stride2
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * v_cache_stride0
            + slot_id * v_cache_stride1
            + head_id * v_cache_stride2
        )
        k_reg = tl.load(key_cache_ptr_offset + col_offsets)
        v_reg = tl.load(value_cache_ptr_offset + col_offsets)
        if DEQUANT:
            k_scale = tl.load(k_scale_ptr)
            v_scale = tl.load(v_scale_ptr)
            k_reg = (k_reg.to(tl.float32) * k_scale).to(
                key_ptr_offset.dtype.element_ty
            )
            v_reg = (v_reg.to(tl.float32) * v_scale).to(
                value_ptr_offset.dtype.element_ty
            )
        tl.store(key_ptr_offset + col_offsets, k_reg)
        tl.store(value_ptr_offset + col_offsets, v_reg)

    elif CACHE_FORMAT == "SHUFFLE":
        key_cache_ptr_offset = (
            key_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + head_id * head_size * PAGE_SIZE
            + slot_id * x
        )
        value_cache_ptr_offset = (
            value_cache_ptr
            + block_id * num_heads * head_size * PAGE_SIZE
            + head_id * head_size * PAGE_SIZE
            + (slot_id // x) * head_size * x
            + slot_id % x
        )
        k_reg_offset = col_offsets // x * PAGE_SIZE * x + col_offsets % x
        v_reg_offset = col_offsets * x
        k_reg = tl.load(key_cache_ptr_offset + k_reg_offset)
        v_reg = tl.load(value_cache_ptr_offset + v_reg_offset)
        if DEQUANT:
            k_scale = 1.0
            v_scale = 1.0
            k_reg = k_reg.to(tl.float32) * k_scale
            v_reg = v_reg.to(tl.float32) * v_scale
        tl.store(key_ptr_offset + col_offsets, k_reg)
        tl.store(value_ptr_offset + col_offsets, v_reg)


def pytorch_ref():
    """Pure PyTorch gather from the paged NHD cache (no dequant)."""
    kc = KEY_CACHE.cpu()
    vc = VALUE_CACHE.cpu()
    bt = BLOCK_TABLE.cpu()
    t2b = TOKEN_TO_BATCH.cpu()
    seq_start = SEQ_STARTS.cpu()
    cu = CU_SEQLENS_KV.cpu()
    out_k = torch.zeros(TOTAL_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    out_v = torch.zeros_like(out_k)
    for tid in range(TOTAL_TOKENS):
        b = int(t2b[tid].item())
        batch_offset = tid - int(cu[b].item()) + int(seq_start[b].item())
        block_offset = batch_offset // PAGE_SIZE
        blk = int(bt[b, block_offset].item())
        slot = batch_offset % PAGE_SIZE
        for h in range(NUM_HEADS):
            out_k[tid, h] = kc[blk, slot, h]
            out_v[tid, h] = vc[blk, slot, h]
    return out_k, out_v


def kernel_impl():
    """Launch only: gather cache into freshly-allocated contiguous buffers."""
    out_k = torch.zeros(
        TOTAL_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=DTYPE, device=DEVICE
    )
    out_v = torch.zeros_like(out_k)
    x = 16 // KEY_CACHE.element_size()
    ks = KEY_CACHE.stride()
    vs = VALUE_CACHE.stride()
    grid = (TOTAL_TOKENS, NUM_HEADS)
    cp_mha_gather_cache_kernel[grid](
        KEY_CACHE,
        VALUE_CACHE,
        out_k,
        out_v,
        BLOCK_TABLE,
        CU_SEQLENS_KV,
        TOKEN_TO_BATCH,
        SEQ_STARTS,
        K_SCALES,
        V_SCALES,
        NUM_HEADS,
        HEAD_SIZE,
        x,
        MAX_BLOCK_NUM,
        ks[0],
        ks[1],
        ks[2],
        vs[0],
        vs[1],
        vs[2],
        DEQUANT=False,
        PAGE_SIZE=PAGE_SIZE,
        CACHE_FORMAT="NHD",
        BLOCK_SIZE=HEAD_SIZE,
    )
    return out_k, out_v


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
            "input_shape": tuple(KEY_CACHE.shape),
            "output_shape": tuple(ker_k.shape),
            "in_dtype": str(KEY_CACHE.dtype),
            "out_dtype": str(ker_k.dtype),
            "device": str(KEY_CACHE.device),
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
            "Kernel: cp_mha_gather_cache_kernel (NHD, DEQUANT=False)\n",
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
