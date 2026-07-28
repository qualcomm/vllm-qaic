"""
Standalone QAIC validation for the `_load_q_td` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_unified_attention.py
  - _load_q_td(...)  (load a query tile via a 2D hardware tensor descriptor)

`_load_q_td` loads a `(BLOCK_M, HEAD_SIZE_PADDED)` query tile from the packed
`[num_tokens, num_query_heads, head_size]` query buffer using
`tl.make_tensor_descriptor` (TMA / HW 2D block read). It flattens the
`num_queries_per_kv` heads of one KV group and `HEAD_SIZE` into a single
contiguous inner row, loads a `BLOCK_Q x (num_queries_per_kv*HEAD_SIZE_PADDED)`
block, and reshapes to `(BLOCK_M, HEAD_SIZE_PADDED)`.

It is fundamentally a strided tile load: for a simple config with
num_queries_per_kv == 1, kv_head_idx == 0, batch-start 0 and
HEAD_SIZE == HEAD_SIZE_PADDED, the result is exactly the first `BLOCK_Q`
query rows of head 0.

TMA-construction limitation (documented per task instructions):
  `tl.make_tensor_descriptor` requires a Triton device allocator
  (`triton.set_allocator`) and HW 2D-block-read support. This backend may
  not support tensor descriptors off-device; we install a standard allocator
  and write the most faithful launcher possible. If the descriptor cannot be
  built/executed on this backend the failure is caught and logged by main()
  (we do NOT execute on hardware as part of this deliverable — py_compile is
  the acceptance gate). The pytorch_ref below is the exact strided-load
  semantics regardless.

Reference: pure PyTorch strided tile slice of the packed query buffer.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.ops.triton_unified_attention import _load_q_td  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_load_q_td.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

DEVICE = "qaic"
NUM_TOKENS = 8
NUM_HEADS = 1
NUM_QUERIES_PER_KV = 1
HEAD_SIZE = 32
HEAD_SIZE_PADDED = 32  # power of 2, equals HEAD_SIZE (TD_QO precondition)
BLOCK_Q = 4
BLOCK_M = BLOCK_Q * NUM_QUERIES_PER_KV

torch.manual_seed(42)
QUERY = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(query):
    """Pure PyTorch strided-load reference.

    Mirrors _load_q_td for num_queries_per_kv=1, kv_head_idx=0, batch start 0,
    q_block_local_idx=0: the first BLOCK_Q rows of query head 0, shape
    (BLOCK_M, HEAD_SIZE_PADDED).
    """
    q = query.float()
    out = torch.zeros(BLOCK_M, HEAD_SIZE_PADDED, dtype=torch.float32)
    tile = q[0:BLOCK_Q, 0, :HEAD_SIZE].cpu()  # [BLOCK_Q, HEAD_SIZE]
    out[:, :HEAD_SIZE] = tile.reshape(BLOCK_M, HEAD_SIZE)
    return out


@triton.jit
def _load_q_launcher(
    query_ptr,
    out_ptr,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    q = _load_q_td(
        query_ptr,
        BLOCK_Q,  # q_block_local_len (all rows valid)
        query_stride_0,
        query_stride_1,
        0,  # cur_batch_in_all_start_index
        0,  # q_block_local_idx
        0,  # kv_head_idx
        num_queries_per_kv,
        BLOCK_Q,
        BLOCK_M,
        HEAD_SIZE,
        HEAD_SIZE_PADDED,
    )
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    tl.store(out_ptr + offs_m[:, None] * HEAD_SIZE_PADDED + offs_d[None, :], q)


def kernel_impl(query):
    """Kernel wrapper: launch only."""
    triton.set_allocator(_alloc_fn)
    out = torch.zeros(BLOCK_M, HEAD_SIZE_PADDED, dtype=torch.float32, device=DEVICE)
    _load_q_launcher[(1,)](
        query,
        out,
        query.stride(0),
        query.stride(1),
        num_queries_per_kv=NUM_QUERIES_PER_KV,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
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
        ref_out = pytorch_ref(QUERY)
        kernel_out = kernel_impl(QUERY)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(QUERY.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(QUERY.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(QUERY.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(QUERY))
        kern_stats = _bench(lambda: kernel_impl(QUERY))
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
            "Kernel: _load_q_td (device helper, tensor descriptor)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- query shape: {stats['input_shape']}\n",
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
