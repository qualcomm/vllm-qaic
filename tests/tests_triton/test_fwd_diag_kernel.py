"""
Kernel test for `_fwd_diag_kernel` (diagonal-block causal attention with
exponential decay), from lightning (linear) attention.

Source under test:
vllm/model_executor/layers/lightning_attn.py
  - _fwd_diag_kernel

This test launches the Triton kernel directly (not through the fused
`lightning_attention` pipeline) with a single-block configuration
(b=1, h=1, n=64, d=16, e=16, BLOCK=64 => NUM_BLOCK=1, CBLOCK=32 => NUM_CBLOCK=2)
so the launch grid reduces to (1, 2) and only this one kernel executes.

Semantics (per source): for a block of BLOCK tokens split into CBLOCK
sub-blocks, causal attention within the block with per-head exponential
decay: score[i, j] = (q_i . k_j) * exp(-s_h * (i - j)) for j <= i (relative
to block start), 0 otherwise; out_i = sum_j score[i, j] * v_j. Since
NUM_BLOCK == 1 here, block-local positions equal global positions, so this
is exactly full causal decayed attention over the whole sequence -- the
sub-block (CBLOCK) tiling used internally by the kernel does not change the
result.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fwd_diag_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/lightning_attn.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.lightning_attn import _fwd_diag_kernel  # noqa: E402

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Global inputs (shared by pytorch_ref and kernel_impl)
# ---------------------------------------------------------------------------
DEVICE = "qaic"
DTYPE = torch.float32

B, H, N, D, E = 1, 1, 64, 16, 16
BLOCK = 64
NUM_BLOCK = 1
CBLOCK = 32
NUM_CBLOCK = BLOCK // CBLOCK  # 2

Q = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
K = torch.randn(B, H, N, D, dtype=DTYPE, device=DEVICE)
V = torch.randn(B, H, N, E, dtype=DTYPE, device=DEVICE)
S = torch.full((H,), 0.05, dtype=DTYPE, device=DEVICE)


def pytorch_ref(q, k, v, s):
    """Pure PyTorch reference for `_fwd_diag_kernel`.

    Full causal attention over the (single) block with per-head
    exponential decay: score[i,j] = (q_i . k_j) * exp(-s*(i-j)) for j<=i,
    else 0; out = score @ v.
    """
    q_cpu = q.cpu()
    k_cpu = k.cpu()
    v_cpu = v.cpu()
    s_cpu = s.cpu()

    b, h, n, d = q_cpu.shape
    e = v_cpu.shape[-1]
    out = torch.zeros(b, h, n, e, dtype=q_cpu.dtype)

    idx = torch.arange(n, dtype=torch.float32)
    diff = idx[:, None] - idx[None, :]  # [n, n]
    causal_mask = diff >= 0

    for bi in range(b):
        for hi in range(h):
            s_val = float(s_cpu[hi].item())
            decay = torch.where(
                causal_mask, torch.exp(-s_val * diff), torch.zeros_like(diff)
            )
            qk = q_cpu[bi, hi] @ k_cpu[bi, hi].t()  # [n, n]
            score = qk * decay
            out[bi, hi] = score @ v_cpu[bi, hi]

    return out


def kernel_impl(q, k, v, s):
    """Kernel wrapper: launches `_fwd_diag_kernel` directly.

    Kernel launch only -- no reference logic, no validation logic.
    """
    b, h, n, d = q.shape
    e = v.shape[-1]
    o = torch.zeros(b, h, n, e, dtype=q.dtype, device=q.device)

    grid = (b * h * NUM_BLOCK, NUM_CBLOCK)
    _fwd_diag_kernel[grid](
        q,
        k,
        v,
        o,
        s,
        b,
        h,
        n,
        d,
        e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        CBLOCK=CBLOCK,
    )
    return o


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
        ref_out = pytorch_ref(Q, K, V, S)
        kernel_out = kernel_impl(Q, K, V, S)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "q_shape": tuple(Q.shape),
            "k_shape": tuple(K.shape),
            "v_shape": tuple(V.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(Q.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(Q.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "relative_error": rel_err,
            "grid": f"({B * H * NUM_BLOCK}, {NUM_CBLOCK})",
        }

        pt_stats = _bench(lambda: pytorch_ref(Q, K, V, S))
        kern_stats = _bench(lambda: kernel_impl(Q, K, V, S))
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
            "Kernel: _fwd_diag_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- k shape: {stats['k_shape']}\n")
            lines.append(f"- v shape: {stats['v_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- s (decay): {S.cpu().tolist()}\n")
            lines.append(f"- BLOCK={BLOCK}, NUM_BLOCK={NUM_BLOCK}, CBLOCK={CBLOCK}\n\n")
            lines.append("Grid Configuration:\n")
            lines.append(f"- grid: {stats['grid']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- relative_error: {stats['relative_error']}\n")
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
