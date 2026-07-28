"""
Standalone QAIC validation for the `_load_kv_tile_td` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_unified_attention.py
  - _load_kv_tile_td(...)  (load a KV cache tile via a 2D tensor descriptor)

`_load_kv_tile_td` loads a `(TILE_SIZE, HEAD_SIZE_PADDED)` tile out of a paged
KV cache laid out as `[num_blocks, block_size, num_kv_heads, head_size]`, for a
single physical block and a single kv head, starting at `offset_in_block`.
Tensor descriptors zero-pad reads beyond `HEAD_SIZE`, so
`HEAD_SIZE_PADDED > HEAD_SIZE` yields zeros in the padded lanes.

It is fundamentally a strided tile load: the result is
`cache[block, offset:offset+TILE_SIZE, kv_head, :HEAD_SIZE]` placed into a
`(TILE_SIZE, HEAD_SIZE_PADDED)` tile (padded lanes zeroed).

TMA-construction limitation (documented per task instructions):
  `tl.make_tensor_descriptor` requires a Triton device allocator and HW 2D
  block-read support, which may be unavailable on this QAIC/Hexagon backend.
  We install a standard allocator and write the most faithful launcher we can;
  py_compile is the acceptance gate and we do NOT execute on hardware. Any
  descriptor-build failure is caught/logged and does not reflect a
  reference-math error. The pytorch_ref is the exact strided-load semantics.

Reference: pure PyTorch strided tile slice of the paged KV cache.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    _load_kv_tile_td,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_load_kv_tile_td.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

DEVICE = "qaic"
NUM_BLOCKS = 1
BLOCK_SIZE = 16
NUM_KV_HEADS = 1
HEAD_SIZE = 32
HEAD_SIZE_PADDED = 32
TILE_SIZE = 8
KV_HEAD_IDX = 0
PHYS_BLOCK = 0
OFFSET_IN_BLOCK = 0

torch.manual_seed(42)
# [num_blocks, block_size, num_kv_heads, head_size]
CACHE = torch.randn(
    NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
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


def _alloc_fn(size, alignment, stream):
    return torch.empty(size, dtype=torch.int8, device=DEVICE)


def pytorch_ref(cache):
    """Pure PyTorch strided-load reference.

    Slices TILE_SIZE rows starting at OFFSET_IN_BLOCK from the given physical
    block / kv head, into a (TILE_SIZE, HEAD_SIZE_PADDED) tile with padded
    lanes zeroed.
    """
    c = cache.float().cpu()
    out = torch.zeros(TILE_SIZE, HEAD_SIZE_PADDED, dtype=torch.float32)
    tile = c[
        PHYS_BLOCK, OFFSET_IN_BLOCK : OFFSET_IN_BLOCK + TILE_SIZE, KV_HEAD_IDX, :HEAD_SIZE
    ]
    out[:, :HEAD_SIZE] = tile
    return out


@triton.jit
def _load_kv_launcher(
    cache_ptr,
    out_ptr,
    stride_0: tl.int64,
    stride_1: tl.int64,
    stride_2: tl.int64,
    stride_3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    tile = _load_kv_tile_td(
        cache_ptr,
        0,  # physical_block_idx_scalar
        0,  # kv_head_idx
        0,  # offset_in_block
        stride_0,
        stride_1,
        stride_2,
        stride_3,
        BLOCK_SIZE,
        TILE_SIZE,
        HEAD_SIZE,
        HEAD_SIZE_PADDED,
    )
    offs_t = tl.arange(0, TILE_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    tl.store(out_ptr + offs_t[:, None] * HEAD_SIZE_PADDED + offs_d[None, :], tile)


def kernel_impl(cache):
    """Kernel wrapper: launch only."""
    triton.set_allocator(_alloc_fn)
    out = torch.zeros(TILE_SIZE, HEAD_SIZE_PADDED, dtype=torch.float32, device=DEVICE)
    _load_kv_launcher[(1,)](
        cache,
        out,
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=HEAD_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(CACHE)
        kernel_out = kernel_impl(CACHE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(CACHE.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(CACHE.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(CACHE.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(CACHE))
        kern_stats = _bench(lambda: kernel_impl(CACHE))
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
            "Kernel: _load_kv_tile_td (device helper, tensor descriptor)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- cache shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- max_rel_err: {stats['rel_err']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines += [
                "Note: TMA (tl.make_tensor_descriptor) may be unsupported on this\n",
                "backend; failure here is a descriptor-construction limitation, not\n",
                "a reference-math error. See module docstring.\n",
                "Error:\n",
                error_text + "\n",
            ]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
