"""
Standalone QAIC validation for `recompute_w_u_fwd_kernel` (kda.py version).

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - recompute_w_u_fwd_kernel  (Recomputes the WY-representation W and U
    pseudo-value tensors for chunked KDA delta-rule attention, with
    per-K-dim gate decay applied to the key/query terms.)

Launcher under test:
  `recompute_w_u_fwd(k, v, beta, A, q=None, gk=None, cu_seqlens=None,
   chunk_indices=None)`.

Semantics (per source, single chunk BT == T):
  u = A @ (v * beta)
  w = A @ (k * beta * exp2(gk))          [gk provided => decay applied]
  STORE_QG = (q is not None)              -> not exercised here (q=None)
  STORE_KG = (kg local var is not None)   -> kg = torch.empty_like(k) if
             gk is not None else None; since gk is provided, kg IS
             allocated, so STORE_KG=True and the kernel additionally
             writes kg = k * beta * exp2(gk_last_in_chunk - gk), where
             gk_last_in_chunk is gk evaluated at the final token index of
             the chunk (broadcast over the BK block). We validate this
             kg output too since it is a real returned value.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_recompute_w_u_fwd_kernel_kda.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"

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

from vllm.model_executor.layers.fla.ops.kda import recompute_w_u_fwd  # noqa: E402

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
torch.manual_seed(42)

B, T, H, K, V = 1, 64, 1, 32, 32
BT = 64  # single chunk == T

K_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
V_INPUT = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE)
BETA_INPUT = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5 + 0.5
# A: well-formed random matrix (not necessarily the true tril-solve inverse;
# fine for pure numerical validation of this kernel's einsum per the task).
A_INPUT = torch.randn(B, T, H, BT, dtype=torch.float32, device=DEVICE) * 0.1
GK_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1


def pytorch_ref(k, v, beta, A, gk):
    """Pure PyTorch reference for `recompute_w_u_fwd_kernel` (kda.py)."""
    Bn, Tn, Hn, Kn = k.shape
    Vn = v.shape[-1]

    # u = A @ (v * beta)   per (b, h)
    vb = v * beta.unsqueeze(-1)  # [B, T, H, V]
    # A: [B, T, H, BT] -> per (b,h) it's a [T, BT] matrix (T==BT here)
    u = torch.zeros_like(v)
    w = torch.zeros_like(k)
    kg = torch.zeros_like(k)

    exp2 = lambda x: torch.exp2(x)

    for b in range(Bn):
        for h in range(Hn):
            A_bh = A[b, :, h, :]  # [T, BT]
            vb_bh = vb[b, :, h, :]  # [T, V]
            u[b, :, h, :] = A_bh @ vb_bh

            k_bh = k[b, :, h, :]  # [T, K]
            beta_bh = beta[b, :, h]  # [T]
            gk_bh = gk[b, :, h, :]  # [T, K]

            kb = k_bh * beta_bh.unsqueeze(-1)
            kb_decayed = kb * exp2(gk_bh)
            w[b, :, h, :] = A_bh @ kb_decayed

            # STORE_KG branch: last_idx = min(BT, T) - 1 = T - 1 (single chunk)
            last_idx = Tn - 1
            gk_last = gk_bh[last_idx, :]  # [K]
            kg[b, :, h, :] = k_bh * exp2(gk_last.unsqueeze(0) - gk_bh)

    return w, u, kg


def kernel_impl(k, v, beta, A, gk):
    w, u, _, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, q=None, gk=gk)
    return w, u, kg


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns dict of latency stats (ms)."""
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
        ref_w, ref_u, ref_kg = pytorch_ref(K_INPUT, V_INPUT, BETA_INPUT, A_INPUT, GK_INPUT)
        ker_w, ker_u, ker_kg = kernel_impl(K_INPUT, V_INPUT, BETA_INPUT, A_INPUT, GK_INPUT)

        ref_w_cpu, ref_u_cpu, ref_kg_cpu = ref_w.cpu(), ref_u.cpu(), ref_kg.cpu()
        ker_w_cpu, ker_u_cpu, ker_kg_cpu = ker_w.cpu(), ker_u.cpu(), ker_kg.cpu()

        torch.testing.assert_close(ker_w_cpu, ref_w_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_u_cpu, ref_u_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_kg_cpu, ref_kg_cpu, rtol=1e-3, atol=1e-3)

        diff_w = (ker_w_cpu - ref_w_cpu).abs()
        diff_u = (ker_u_cpu - ref_u_cpu).abs()
        diff_kg = (ker_kg_cpu - ref_kg_cpu).abs()
        max_abs_diff = max(diff_w.max().item(), diff_u.max().item(), diff_kg.max().item())
        mean_abs_diff = (diff_w.mean().item() + diff_u.mean().item() + diff_kg.mean().item()) / 3.0

        stats = {
            "k_shape": tuple(K_INPUT.shape),
            "v_shape": tuple(V_INPUT.shape),
            "A_shape": tuple(A_INPUT.shape),
            "w_shape": tuple(ker_w.shape),
            "u_shape": tuple(ker_u.shape),
            "kg_shape": tuple(ker_kg.shape),
            "dtype": str(K_INPUT.dtype),
            "device": str(K_INPUT.device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff_w": diff_w.max().item(),
            "max_abs_diff_u": diff_u.max().item(),
            "max_abs_diff_kg": diff_kg.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(K_INPUT, V_INPUT, BETA_INPUT, A_INPUT, GK_INPUT))
        kern_stats = _bench(lambda: kernel_impl(K_INPUT, V_INPUT, BETA_INPUT, A_INPUT, GK_INPUT))
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
            "Kernel: recompute_w_u_fwd_kernel (kda.py)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- k shape: {stats['k_shape']}, v shape: {stats['v_shape']}, A shape: {stats['A_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}, device: {stats['device']}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- w shape: {stats['w_shape']}, u shape: {stats['u_shape']}, kg shape: {stats['kg_shape']}\n")
            lines.append(f"- max_abs_diff (overall): {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff (overall): {stats['mean_abs_diff']}\n")
            lines.append(f"- max_abs_diff_w: {stats['max_abs_diff_w']}\n")
            lines.append(f"- max_abs_diff_u: {stats['max_abs_diff_u']}\n")
            lines.append(f"- max_abs_diff_kg: {stats['max_abs_diff_kg']}\n")
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
