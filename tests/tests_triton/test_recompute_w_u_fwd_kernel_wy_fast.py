"""
Standalone QAIC validation for `recompute_w_u_fwd_kernel` (wy_fast.py variant).

Source under test:
vllm/model_executor/layers/fla/ops/wy_fast.py
  - recompute_w_u_fwd_kernel  (Recomputes the WY-representation W
    (decayed/beta-scaled keys) and U (beta-scaled values) tensors.)

Launcher `recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens,
chunk_indices=None)`.

Semantics (per chunk):
  u = A @ (v * beta)                         (beta broadcast over V)
  w = A @ (k * beta * exp(g_cumsum))         (beta/g broadcast over K)
GQA broadcast: k has Hg heads, output w has H heads via head-group mapping
i_h // (H // Hg). With Hg == H this is a no-op (used here).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_recompute_w_u_fwd_kernel_wy_fast.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/wy_fast.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# recompute_w_u_fwd_kernel is wrapped in @triton.autotune. Triton's generic
# autotuner benchmarking path (triton.testing.do_bench) calls
# Event.elapsed_time() across two independently-created torch.qaic Events,
# which raises "expected other to be a torch.Event object" on this QAIC
# backend/torch_qaic version. This is a benchmarking-harness incompatibility,
# unrelated to kernel correctness, so we shim do_bench to just execute the
# candidate config once and report a constant timing. The kernel itself
# still executes for real on qaic hardware.
def _qaic_safe_do_bench(fn, *args, **kwargs):
    fn()
    quantiles = kwargs.get("quantiles")
    if quantiles is not None:
        return [1.0] * len(quantiles) if len(quantiles) > 1 else 1.0
    return 1.0


_triton_testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd  # noqa: E402

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
B, T, Hg, H, K, V = 1, 64, 1, 1, 32, 32
BT = 64  # single chunk, T == BT

torch.manual_seed(42)
K_TENSOR = torch.randn(B, T, Hg, K, dtype=torch.float32, device=DEVICE)
V_TENSOR = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE)
BETA = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5 + 0.5
G_CUMSUM = (torch.rand(B, T, H, dtype=torch.float32, device=DEVICE) - 0.5) * 0.1
# A is a generic attention-weight matrix here (not a triangular-solve input);
# lower-triangular including diagonal, per task instructions.
A_RAW = torch.randn(B, H, T, BT, dtype=torch.float32, device=DEVICE)
A_TENSOR = torch.tril(A_RAW, diagonal=0).permute(0, 2, 1, 3).contiguous()

INPUT = (K_TENSOR, V_TENSOR, BETA, G_CUMSUM, A_TENSOR)
WEIGHT = None
BIAS = None


def pytorch_ref(k, v, beta, g_cumsum, A):
    """Pure PyTorch reference implementation for `recompute_w_u_fwd_kernel`.

    u = A @ (v * beta), w = A @ (k * beta * exp(g_cumsum)), with GQA head
    mapping i_h // (H // Hg) from output head h to input k head.
    """
    k_cpu = k.detach().cpu()
    v_cpu = v.detach().cpu()
    beta_cpu = beta.detach().cpu()
    g_cpu = g_cumsum.detach().cpu()
    A_cpu = A.detach().cpu()

    b, t, hg, kk = k_cpu.shape
    _, _, h, vv = v_cpu.shape
    bt = A_cpu.shape[-1]
    assert t == bt, "reference assumes a single chunk (T == BT)"

    u = torch.zeros_like(v_cpu)
    w = torch.zeros(b, t, h, kk, dtype=k_cpu.dtype)

    group_size = h // hg
    for ib in range(b):
        for ih in range(h):
            ikg = ih // group_size
            A_blk = A_cpu[ib, :, ih, :]  # [T, BT]
            beta_h = beta_cpu[ib, :, ih]  # [T]
            g_h = torch.exp(g_cpu[ib, :, ih])  # [T]

            v_blk = v_cpu[ib, :, ih, :]  # [T, V]
            vb = v_blk * beta_h[:, None]
            u[ib, :, ih, :] = A_blk @ vb

            k_blk = k_cpu[ib, :, ikg, :]  # [T, K]
            kb = k_blk * beta_h[:, None] * g_h[:, None]
            w[ib, :, ih, :] = A_blk @ kb

    return w, u


def kernel_impl(k, v, beta, g_cumsum, A):
    return recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        A=A,
        cu_seqlens=None,
        chunk_indices=None,
    )


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
        ref_w, ref_u = pytorch_ref(*INPUT)
        kernel_w, kernel_u = kernel_impl(*INPUT)

        ref_w_cpu = ref_w.cpu()
        ref_u_cpu = ref_u.cpu()
        kernel_w_cpu = kernel_w.cpu()
        kernel_u_cpu = kernel_u.cpu()

        torch.testing.assert_close(kernel_w_cpu, ref_w_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kernel_u_cpu, ref_u_cpu, rtol=1e-3, atol=1e-3)

        diff_w = (kernel_w_cpu - ref_w_cpu).abs()
        diff_u = (kernel_u_cpu - ref_u_cpu).abs()
        rel_err_w = (diff_w / (ref_w_cpu.abs() + 1e-8)).mean().item()
        rel_err_u = (diff_u / (ref_u_cpu.abs() + 1e-8)).mean().item()

        stats = {
            "k_shape": tuple(K_TENSOR.shape),
            "v_shape": tuple(V_TENSOR.shape),
            "beta_shape": tuple(BETA.shape),
            "g_cumsum_shape": tuple(G_CUMSUM.shape),
            "A_shape": tuple(A_TENSOR.shape),
            "w_shape": tuple(kernel_w.shape),
            "u_shape": tuple(kernel_u.shape),
            "input_dtype": str(K_TENSOR.dtype),
            "output_dtype": str(kernel_w.dtype),
            "device": str(K_TENSOR.device),
            "max_abs_diff_w": diff_w.max().item(),
            "mean_abs_diff_w": diff_w.mean().item(),
            "relative_error_w": rel_err_w,
            "max_abs_diff_u": diff_u.max().item(),
            "mean_abs_diff_u": diff_u.mean().item(),
            "relative_error_u": rel_err_u,
        }

        pt_stats = _bench(lambda: pytorch_ref(*INPUT))
        kern_stats = _bench(lambda: kernel_impl(*INPUT))
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
            "Kernel: recompute_w_u_fwd_kernel (wy_fast.py)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(
                f"- k shape: {stats['k_shape']} (B={B}, T={T}, Hg={Hg}, K={K})\n"
            )
            lines.append(f"- v shape: {stats['v_shape']} (H={H}, V={V})\n")
            lines.append(f"- beta shape: {stats['beta_shape']}\n")
            lines.append(f"- g_cumsum shape: {stats['g_cumsum_shape']}\n")
            lines.append(f"- A shape: {stats['A_shape']} (BT={BT})\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- w shape: {stats['w_shape']}\n")
            lines.append(f"- u shape: {stats['u_shape']}\n")
            lines.append(f"- output dtype: {stats['output_dtype']}\n")
            lines.append(f"- max_abs_diff (w): {stats['max_abs_diff_w']}\n")
            lines.append(f"- mean_abs_diff (w): {stats['mean_abs_diff_w']}\n")
            lines.append(f"- relative_error (w): {stats['relative_error_w']}\n")
            lines.append(f"- max_abs_diff (u): {stats['max_abs_diff_u']}\n")
            lines.append(f"- mean_abs_diff (u): {stats['mean_abs_diff_u']}\n")
            lines.append(f"- relative_error (u): {stats['relative_error_u']}\n")
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
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
