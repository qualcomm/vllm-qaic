"""
Kernel test for `_linear_attn_decode_kernel` (single-token decode-step
kernel updating a per-sequence KV cache slot with exponential decay), from
lightning (linear) attention. Uses the clean standalone Python launcher
`linear_decode_forward_triton`.

Source under test:
vllm/model_executor/layers/lightning_attn.py
  - _linear_attn_decode_kernel (via linear_decode_forward_triton)

Configuration: B=2, H=2, D=32, BLOCK_SIZE=16 (D divisible by BLOCK_SIZE, so
the D // BLOCK_SIZE grid dimension is exactly 2). q, k, v have shape
[B, H, 1, D]; kv_caches has shape [B, H, D, D] (per-sequence D x D outer
product cache); slope_rate has shape [H]; slot_idx = [0, 1] (both valid,
non-PAD_SLOT_ID slots).

COMPILER COMPROMISE: BLOCK_SIZE=32 (the task's suggested value, D //
BLOCK_SIZE == 1) was tried first and failed to compile on this
environment's QAIC/Hexagon Triton backend with `'scf.for' op init_arg and
yielded value bufferize to inconsistent memory spaces` inside the shared
`translate_linalg_to_obj` lowering stage -- reproduced with the unmodified
upstream `_linear_attn_decode_kernel` itself, and bisected down to a
minimal repro: a standalone kernel doing only `tl.load` + elementwise
multiply + `tl.sum(axis=0)` + `tl.store` fails specifically when both the
reduced (D) and kept (BLOCK_SIZE) tile dimensions equal 32 -- e.g. (D=32,
N=32) and (D=16, N=32) both fail, while (D=32, N=16), (D=32, N=64), (D=16,
N=64), (D=64, N=64) all succeed. This is a backend/compiler tiling
limitation on this box, not a bug in our test harness or in the kernel's
semantics. BLOCK_SIZE=16 avoids the problematic tile shape and was
verified numerically correct (max abs diff ~4e-6) against a hand reference
before being adopted for this file.

Semantics (per source): kv_outer = outer(k, v); kv_new = kv_outer +
exp(-slope_rate[h]) * kv_cache_old[slot]; output = sum_d(q * kv_new) (a
contraction over the query/key feature dimension against kv_new); the KV
cache slot is updated in place to kv_new.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_linear_attn_decode_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/lightning_attn.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.lightning_attn import (  # noqa: E402
    linear_decode_forward_triton,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B, H, D = 2, 2, 32
BLOCK_SIZE = 16  # only tile size that compiles on this QAIC/Hexagon backend

Q = torch.randn(B, H, 1, D, dtype=DTYPE, device=DEVICE)
K = torch.randn(B, H, 1, D, dtype=DTYPE, device=DEVICE)
V = torch.randn(B, H, 1, D, dtype=DTYPE, device=DEVICE)
KV_CACHES = torch.randn(B, H, D, D, dtype=DTYPE, device=DEVICE)
SLOPE_RATE = torch.tensor([0.1, 0.2], dtype=DTYPE, device=DEVICE)
assert PAD_SLOT_ID != 0 and PAD_SLOT_ID != 1, "slot_idx values must not be PAD_SLOT_ID"
SLOT_IDX = torch.tensor([0, 1], dtype=torch.long, device=DEVICE)


def pytorch_ref(q, k, v, kv_caches, slope_rate, slot_idx):
    """Pure PyTorch reference for `_linear_attn_decode_kernel`.

    For each batch b with a valid (non-PAD_SLOT_ID) slot:
      kv_outer = outer(k[b,h,0], v[b,h,0])
      kv_new = kv_outer + exp(-slope_rate[h]) * kv_cache_old[slot]
      output[b, h*D:(h+1)*D] = sum_d(q[b,h,0] * kv_new)   (contract over d)
      kv_caches[slot, h] updated in place to kv_new.

    Returns (output, kv_caches_updated) -- output has shape [B, H*D].
    """
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    kv_caches_cpu = kv_caches.cpu().clone()
    slope_rate_cpu = slope_rate.cpu()
    slot_idx_cpu = slot_idx.cpu()

    b, h, _, d = q_cpu.shape
    output = torch.zeros(b, h * d, dtype=q_cpu.dtype)

    for bi in range(b):
        slot_id = int(slot_idx_cpu[bi].item())
        if slot_id == PAD_SLOT_ID:
            continue
        for hi in range(h):
            q_bh = q_cpu[bi, hi, 0]  # [d]
            k_bh = k_cpu[bi, hi, 0]  # [d]
            v_bh = v_cpu[bi, hi, 0]  # [d]
            ratio = torch.exp(-slope_rate_cpu[hi])
            kv_outer = torch.outer(k_bh, v_bh)  # [d, d]
            kv_cache_old = kv_caches_cpu[slot_id, hi]  # [d, d]
            kv_new = kv_outer + ratio * kv_cache_old
            out_h = (q_bh.unsqueeze(-1) * kv_new).sum(dim=0)  # [d]
            output[bi, hi * d : (hi + 1) * d] = out_h
            kv_caches_cpu[slot_id, hi] = kv_new

    return output, kv_caches_cpu


def kernel_impl(q, k, v, kv_caches, slope_rate, slot_idx):
    """Kernel wrapper: launches `_linear_attn_decode_kernel` via its
    standalone Python launcher `linear_decode_forward_triton`.

    Kernel launch only -- no reference logic, no validation logic.
    `kv_caches` is mutated in place by the kernel, so we clone the shared
    global before launching.
    """
    kv_caches_local = kv_caches.clone()
    output = linear_decode_forward_triton(
        q, k, v, kv_caches_local, slope_rate, slot_idx, BLOCK_SIZE=BLOCK_SIZE
    )
    return output, kv_caches_local


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


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_kv_caches = pytorch_ref(Q, K, V, KV_CACHES, SLOPE_RATE, SLOT_IDX)
        kernel_out, kernel_kv_caches = kernel_impl(
            Q, K, V, KV_CACHES, SLOPE_RATE, SLOT_IDX
        )

        ref_out_cpu = ref_out.cpu()
        kernel_out_cpu = kernel_out.cpu()
        ref_kv_cpu = ref_kv_caches.cpu()
        kernel_kv_cpu = kernel_kv_caches.cpu()

        torch.testing.assert_close(kernel_out_cpu, ref_out_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kernel_kv_cpu, ref_kv_cpu, rtol=1e-3, atol=1e-3)

        diff_out = (kernel_out_cpu - ref_out_cpu).abs()
        diff_kv = (kernel_kv_cpu - ref_kv_cpu).abs()
        rel_err_out = (diff_out / (ref_out_cpu.abs() + 1e-8)).mean().item()
        rel_err_kv = (diff_kv / (ref_kv_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "q_shape": tuple(Q.shape),
            "k_shape": tuple(K.shape),
            "v_shape": tuple(V.shape),
            "kv_caches_shape": tuple(KV_CACHES.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(Q.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(Q.device),
            "max_abs_diff_output": diff_out.max().item(),
            "mean_abs_diff_output": diff_out.mean().item(),
            "relative_error_output": rel_err_out,
            "max_abs_diff_kv_caches": diff_kv.max().item(),
            "mean_abs_diff_kv_caches": diff_kv.mean().item(),
            "relative_error_kv_caches": rel_err_kv,
            "grid": f"({B}, {H}, {D // BLOCK_SIZE})",
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, K, V, KV_CACHES, SLOPE_RATE, SLOT_IDX))
        kern_stats = _bench(lambda: kernel_impl(Q, K, V, KV_CACHES, SLOPE_RATE, SLOT_IDX))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: _linear_attn_decode_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- k shape: {stats['k_shape']}\n")
            lines.append(f"- v shape: {stats['v_shape']}\n")
            lines.append(f"- kv_caches shape: {stats['kv_caches_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- slope_rate: {SLOPE_RATE.cpu().tolist()}\n")
            lines.append(f"- slot_idx: {SLOT_IDX.cpu().tolist()}\n")
            lines.append(f"- BLOCK_SIZE={BLOCK_SIZE}\n\n")
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff (output): {stats['max_abs_diff_output']}\n")
            lines.append(
                f"- mean_abs_diff (output): {stats['mean_abs_diff_output']}\n"
            )
            lines.append(
                f"- relative_error (output): {stats['relative_error_output']}\n"
            )
            lines.append(
                f"- max_abs_diff (kv_caches): {stats['max_abs_diff_kv_caches']}\n"
            )
            lines.append(
                f"- mean_abs_diff (kv_caches): {stats['mean_abs_diff_kv_caches']}\n"
            )
            lines.append(
                f"- relative_error (kv_caches): "
                f"{stats['relative_error_kv_caches']}\n"
            )
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
