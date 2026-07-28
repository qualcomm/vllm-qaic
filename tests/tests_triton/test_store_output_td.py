"""
Standalone QAIC validation for the `_store_output_td` @triton.jit device helper.

Source under test:
vllm/v1/attention/ops/triton_unified_attention.py
  - _store_output_td(...)  (store an output tile via a 2D/3D tensor descriptor)

`_store_output_td` writes a `(BLOCK_M, HEAD_SIZE_PADDED)` accumulator into the
packed output buffer `[num_tokens, num_query_heads, head_size]`. It reshapes
`acc` to `(BLOCK_Q, num_queries_per_kv, HEAD_SIZE_PADDED)` and stores it via a
tensor descriptor whose shape is `(q_block_local_len, num_queries_per_kv,
HEAD_SIZE)` with strides `(stride_token, stride_head, 1)`.

It is fundamentally a strided tile store: the accumulator rows land in
`output[token_start : token_start+BLOCK_Q, head_start : head_start+nqpkv,
:HEAD_SIZE]`.

TMA-construction limitation (documented per task instructions):
  `tl.make_tensor_descriptor` requires a Triton device allocator and HW 2D
  block-write support, which may be unavailable on this QAIC/Hexagon backend.
  We install a standard allocator and write the most faithful launcher we can;
  py_compile is the acceptance gate and we do NOT execute on hardware. A
  descriptor-build failure is caught/logged and is not a reference-math error.
  The pytorch_ref is the exact strided-store semantics.

Reference: pure PyTorch reshape + strided placement of the accumulator.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import tl, triton  # noqa: E402
from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    _store_output_td,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_store_output_td.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_unified_attention.py"

DEVICE = "qaic"
NUM_TOKENS = 8
NUM_HEADS = 1
NUM_QUERIES_PER_KV = 1
HEAD_SIZE = 32
HEAD_SIZE_PADDED = 32
BLOCK_Q = 4
BLOCK_M = BLOCK_Q * NUM_QUERIES_PER_KV

torch.manual_seed(42)
# Accumulator produced by the attention loop, shape (BLOCK_M, HEAD_SIZE_PADDED).
ACC = torch.randn(BLOCK_M, HEAD_SIZE_PADDED, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(acc):
    """Pure PyTorch strided-store reference.

    Writes the accumulator into the first BLOCK_Q tokens of head 0. Returns
    the full output buffer [NUM_TOKENS, NUM_HEADS, HEAD_SIZE].
    """
    a = acc.float().cpu()
    out = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32)
    tile = a.reshape(BLOCK_Q, NUM_QUERIES_PER_KV, HEAD_SIZE_PADDED)[..., :HEAD_SIZE]
    out[0:BLOCK_Q, 0:NUM_QUERIES_PER_KV, :] = tile
    return out


@triton.jit
def _store_launcher(
    acc_ptr,
    out_ptr,
    stride_token: tl.int64,
    stride_head: tl.int64,
    num_queries_per_kv: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    acc = tl.load(acc_ptr + offs_m[:, None] * HEAD_SIZE_PADDED + offs_d[None, :])
    _store_output_td(
        out_ptr,
        acc,
        BLOCK_Q,  # q_block_local_len
        stride_token,
        stride_head,
        num_queries_per_kv,
        BLOCK_Q,
        HEAD_SIZE,
        HEAD_SIZE_PADDED,
    )


def kernel_impl(acc):
    """Kernel wrapper: launch only."""
    triton.set_allocator(_alloc_fn)
    out = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE)
    _store_launcher[(1,)](
        acc,
        out,
        out.stride(0),
        out.stride(1),
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
        ref_out = pytorch_ref(ACC)
        kernel_out = kernel_impl(ACC)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().clamp_min(1e-6)
        stats = {
            "input_shape": tuple(ACC.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(ACC.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(ACC.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff / denom).max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(ACC))
        kern_stats = _bench(lambda: kernel_impl(ACC))
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
            "Kernel: _store_output_td (device helper, tensor descriptor)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- acc shape: {stats['input_shape']}\n",
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
