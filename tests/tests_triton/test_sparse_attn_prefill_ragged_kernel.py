"""
Standalone QAIC validation for `_sparse_attn_prefill_ragged_kernel`.

Source under test:
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
  - _sparse_attn_prefill_ragged_kernel
  - launch site replicated from _rocm_sparse_attn_prefill_ragged_triton(...)

Ragged (variable-length, indexed) sparse attention over the selected top-k KV
positions for DeepSeek-V4 sparse-MLA prefill. For each query token, the ragged
`kv_indices[kv_indptr[q] : kv_indptr[q+1]]` list gives the KV rows to attend to;
the kernel runs an online-softmax flash-attention over just those rows (only
slots in [0, num_kv) are valid), with an OPTIONAL per-head attention sink:

    scores = (q @ k_selected^T) * scale                (invalid slots masked out)
    softmax over the selected keys (plus sink logit if present)
    out = softmax_weights @ k_selected

LAUNCH-SITE RECONSTRUCTION: the public wrapper `_rocm_sparse_attn_prefill_ragged_triton`
hard-asserts the DeepSeek dims (head_dim=512, nope=448, rope=64) and requires
non-CPU tensors, so we instead replicate its exact kernel launch here with tiny
dims (head_dim=16). This exercises identical kernel code paths with genuinely
small shapes.

SIMPLIFICATIONS:
  - HAS_ATTN_SINK = False (no attention sink) per the task's "no sink first".
    The kernel's sink branch is not exercised.
  - q/kv/out are float32 (the wrapper casts out to bf16; launching directly we
    keep fp32 so the flash-attention math is compared at full precision).

FLOAT attention kernel -> rtol/atol = 1e-3 against a pure-PyTorch gather+softmax.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import triton
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _sparse_attn_prefill_ragged_kernel,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_sparse_attn_prefill_ragged_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_QUERIES = 3
NUM_HEADS = 4
HEAD_DIM = 16
NUM_KV = 8
SCALE = 1.0 / (HEAD_DIM ** 0.5)

Q = torch.randn(NUM_QUERIES, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
KV = torch.randn(NUM_KV, HEAD_DIM, dtype=torch.float32, device=DEVICE)

# Ragged selection: q0 -> [0,3,5], q1 -> [] (empty), q2 -> [1,2,4,6,7].
_SELECT = [[0, 3, 5], [], [1, 2, 4, 6, 7]]
_flat = [s for row in _SELECT for s in row]
KV_INDICES = torch.tensor(_flat, dtype=torch.int32, device=DEVICE)
_lens = [len(row) for row in _SELECT]
KV_INDPTR = torch.zeros(NUM_QUERIES + 1, dtype=torch.int32, device=DEVICE)
KV_INDPTR[1:] = torch.tensor(_lens, dtype=torch.int32).cumsum(0)


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


def pytorch_ref(q, kv, kv_indices, kv_indptr):
    q = q.cpu().to(torch.float32)
    kv = kv.cpu().to(torch.float32)
    kv_indices = kv_indices.cpu()
    kv_indptr = kv_indptr.cpu()

    out = torch.zeros(NUM_QUERIES, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    for qi in range(NUM_QUERIES):
        start = int(kv_indptr[qi].item())
        end = int(kv_indptr[qi + 1].item())
        slots = [
            int(kv_indices[j].item())
            for j in range(start, end)
            if 0 <= int(kv_indices[j].item()) < NUM_KV
        ]
        if len(slots) == 0:
            continue  # l_i == 0 -> zero output
        k_sel = kv[slots]  # [n_sel, head_dim]
        # scores: [num_heads, n_sel]
        scores = (q[qi] @ k_sel.T) * SCALE
        m = scores.max(dim=-1, keepdim=True).values
        p = torch.exp(scores - m)
        denom = p.sum(dim=-1, keepdim=True)
        out[qi] = (p @ k_sel) / denom
    return out


def kernel_impl(q, kv, kv_indices, kv_indptr):
    attn_sink = torch.empty(1, device=DEVICE, dtype=torch.float32)
    out = torch.empty(
        NUM_QUERIES, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE
    )
    block_h = 16
    block_d = triton.next_power_of_2(HEAD_DIM)
    block_k = 16 if HEAD_DIM >= 256 else 32
    _sparse_attn_prefill_ragged_kernel[
        (NUM_QUERIES, triton.cdiv(NUM_HEADS, block_h))
    ](
        q,
        kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_HEADS,
        HEAD_DIM,
        kv.shape[0],
        float(SCALE),
        HAS_ATTN_SINK=False,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(Q, KV, KV_INDICES, KV_INDPTR)
        out = kernel_impl(Q, KV, KV_INDICES, KV_INDPTR).cpu().to(torch.float32)

        torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
        diff = (out - ref).abs()
        stats = {
            "q_shape": tuple(Q.shape),
            "kv_shape": tuple(KV.shape),
            "out_shape": tuple(out.shape),
            "in_dtype": str(Q.dtype),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "kv_indptr": KV_INDPTR.cpu().tolist(),
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, KV, KV_INDICES, KV_INDPTR))
        kern_stats = _bench(lambda: kernel_impl(Q, KV, KV_INDICES, KV_INDPTR))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _sparse_attn_prefill_ragged_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- q shape: {stats['q_shape']} dtype {stats['in_dtype']}\n",
                f"- kv shape: {stats['kv_shape']}, num_heads={NUM_HEADS}, head_dim={HEAD_DIM}\n",
                f"- kv_indptr: {stats['kv_indptr']} (q1 empty), scale={SCALE}\n",
                f"- HAS_ATTN_SINK=False, device: {stats['device']}\n\n",
                "Output (rtol/atol=1e-3):\n",
                f"- out shape: {stats['out_shape']} dtype {stats['out_dtype']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n"
                )
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
