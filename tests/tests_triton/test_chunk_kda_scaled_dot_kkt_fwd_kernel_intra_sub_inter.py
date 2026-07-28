"""
Standalone QAIC validation for `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`.

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter
    (Computes the beta-scaled K@K^T cross terms between different
    sub-chunks (i_i > i_j pairs) with KDA's exp2-based per-K-dim gate
    decay, plus the matching scaled Q@K^T term Aqk used later for the
    query-side attention.)

Dispatch note: this kernel and `..._intra_sub_intra` are ALWAYS launched
together by the higher-level `chunk_kda_scaled_dot_kkt_fwd(...)`; there is
no launcher parameter that isolates just this kernel. To keep this test
targeted at a single kernel launch (per Claude.md's "kernel launch only"
philosophy), we invoke the raw triton-jitted kernel object directly,
replicating exactly the grid/kwargs that `chunk_kda_scaled_dot_kkt_fwd`
uses for this specific kernel (grid=(NT, NC*NC, B*H), no BK/IS_VARLEN
kwargs passed -- BK is chosen by the kernel's own @triton.autotune
configs, and IS_VARLEN is computed by @triton.heuristics from
cu_seqlens=None).

Semantics (per source, single chunk NT=1, i_t=0):
  For each sub-chunk pair (i_i, i_j) with i_i > i_j (i_i, i_j in [0, NC)):
    b_gn = g[i_i*BC, :]                      (gate at first row of i_i block)
    b_k_decayed[p, :] = k[i_i*BC+p, :] * exp2(g[i_i*BC+p, :] - b_gn)
    b_ktg[:, q]        = k[i_j*BC+q, :] * exp2(b_gn - g[i_j*BC+q, :])
    b_A[p, q]  = beta[i_i*BC+p] * sum_k b_k_decayed[p, k] * b_ktg[k, q]
    b_q_decayed[p, :] = q[i_i*BC+p, :] * exp2(g[i_i*BC+p, :] - b_gn) * scale
    b_Aqk[p, q] = sum_k b_q_decayed[p, k] * b_ktg[k, q]
  Stored into A[i_i*BC:+BC, i_j*BC:+BC] / Aqk[...] blocks of the [T, BT]
  matrix (per batch/head); all other (i_i <= i_j) entries stay 0 (buffers
  are zero-initialized, matching `chunk_kda_scaled_dot_kkt_fwd`).

  Note: the b_gn reference point algebraically cancels
  (exp2(g_p - gn) * exp2(gn - g_q) == exp2(g_p - g_q)), so the closed form
  is b_A[p,q] = beta[p] * sum_k k[p,k]*k[q,k]*exp2(g[p,k]-g[q,k]); we
  replicate the literal two-factor decomposition below for fidelity to the
  source's exact floating-point path.
"""

import datetime
import os
import sys
import time
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter.txt"
)
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter is wrapped in
# @triton.autotune. Triton's generic autotuner benchmarking path
# (triton.testing.do_bench) calls Event.elapsed_time() across two
# independently-created torch.qaic Events, which raises "expected other to
# be a torch.Event object" on this QAIC backend/torch_qaic version. This is
# a benchmarking-harness incompatibility, unrelated to kernel correctness,
# so we shim do_bench to just execute the candidate config once and report
# a constant timing. The kernel itself still executes for real on qaic
# hardware.
def _qaic_safe_do_bench(fn, *args, **kwargs):
    fn()
    quantiles = kwargs.get("quantiles")
    if quantiles is not None:
        return [1.0] * len(quantiles) if len(quantiles) > 1 else 1.0
    return 1.0


_triton_testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.kda import (  # noqa: E402
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
torch.manual_seed(42)

B, T, H, K = 1, 64, 1, 32  # K <= 256 required
BT = T  # single chunk, NT = 1
BC = min(16, BT)  # 16
NC = (BT + BC - 1) // BC  # 4
BK = max(1 << (K - 1).bit_length(), 16)  # next_power_of_2(K) padded to >=16 -> 32
SCALE = 1.0 / (K ** 0.5)

Q_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
K_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
G_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
BETA_INPUT = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5 + 0.5


def pytorch_ref(q, k, g, beta, scale, BT_=BT, BC_=BC, NC_=NC):
    """Pure PyTorch reference for the inter-sub-chunk (i_i > i_j) blocks."""
    Bn, Tn, Hn, Kn = q.shape
    A = torch.zeros(Bn, Tn, Hn, BT_, dtype=torch.float32, device=q.device)
    Aqk = torch.zeros(Bn, Tn, Hn, BT_, dtype=torch.float32, device=q.device)

    for b in range(Bn):
        for h in range(Hn):
            q_bh = q[b, :, h, :]
            k_bh = k[b, :, h, :]
            g_bh = g[b, :, h, :]
            beta_bh = beta[b, :, h]

            for i_i in range(NC_):
                for i_j in range(NC_):
                    if i_i <= i_j:
                        continue
                    row0 = i_i * BC_
                    col0 = i_j * BC_
                    b_gn = g_bh[row0, :]  # [K]

                    b_q = q_bh[row0 : row0 + BC_, :]
                    b_k = k_bh[row0 : row0 + BC_, :]
                    b_g = g_bh[row0 : row0 + BC_, :]
                    b_k_decayed = b_k * torch.exp2(b_g - b_gn[None, :])

                    b_kt = k_bh[col0 : col0 + BC_, :]  # [BC, K]
                    b_gk = g_bh[col0 : col0 + BC_, :]  # [BC, K]
                    # b_ktg[k, q] = k[col0+q, k] * exp2(gn[k] - gk[q, k])
                    b_ktg = b_kt.transpose(0, 1) * torch.exp2(
                        b_gn[:, None] - b_gk.transpose(0, 1)
                    )  # [K, BC]

                    b_A = b_k_decayed @ b_ktg  # [BC, BC]
                    b_A = b_A * beta_bh[row0 : row0 + BC_][:, None]

                    b_q_decayed = b_q * torch.exp2(b_g - b_gn[None, :]) * scale
                    b_Aqk = b_q_decayed @ b_ktg  # [BC, BC]

                    A[b, row0 : row0 + BC_, h, col0 : col0 + BC_] = b_A
                    Aqk[b, row0 : row0 + BC_, h, col0 : col0 + BC_] = b_Aqk

    return A, Aqk


def kernel_impl(q, k, g, beta, scale, BT_=BT, BC_=BC, NC_=NC):
    Bn, Tn, Hn, Kn = q.shape
    NT = 1
    grid = (NT, NC_ * NC_, Bn * Hn)

    # This shared QAIC box intermittently returns transient device-level
    # errors ("Failed to synchronize stream", "Failed to create custom
    # library device executable") under contention from other concurrent
    # processes on the same NSPs -- unrelated to kernel correctness. Retry
    # the launch a bounded number of times; this is launch-robustness
    # infra, not reference/validation logic.
    last_err = None
    for _attempt in range(30):
        try:
            A = torch.zeros(Bn, Tn, Hn, BT_, dtype=torch.float32, device=q.device)
            Aqk = torch.zeros(Bn, Tn, Hn, BT_, dtype=torch.float32, device=q.device)
            chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter[grid](
                q=q,
                k=k,
                g=g,
                beta=beta,
                A=A,
                Aqk=Aqk,
                scale=scale,
                cu_seqlens=None,
                chunk_indices=None,
                T=Tn,
                H=Hn,
                K=Kn,
                BT=BT_,
                BC=BC_,
                NC=NC_,
            )
            _ = A.abs().sum().item()  # force sync / surface device errors now
            return A, Aqk
        except RuntimeError as e:
            last_err = e
            time.sleep(5)
    raise last_err


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
        ref_A, ref_Aqk = pytorch_ref(Q_INPUT, K_INPUT, G_INPUT, BETA_INPUT, SCALE)
        ker_A, ker_Aqk = kernel_impl(Q_INPUT, K_INPUT, G_INPUT, BETA_INPUT, SCALE)

        ref_A_cpu, ref_Aqk_cpu = ref_A.cpu(), ref_Aqk.cpu()
        ker_A_cpu, ker_Aqk_cpu = ker_A.cpu(), ker_Aqk.cpu()

        torch.testing.assert_close(ker_A_cpu, ref_A_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ker_Aqk_cpu, ref_Aqk_cpu, rtol=1e-3, atol=1e-3)

        diff_A = (ker_A_cpu - ref_A_cpu).abs()
        diff_Aqk = (ker_Aqk_cpu - ref_Aqk_cpu).abs()
        max_abs_diff = max(diff_A.max().item(), diff_Aqk.max().item())
        mean_abs_diff = (diff_A.mean().item() + diff_Aqk.mean().item()) / 2.0

        stats = {
            "q_shape": tuple(Q_INPUT.shape),
            "k_shape": tuple(K_INPUT.shape),
            "A_shape": tuple(ker_A.shape),
            "Aqk_shape": tuple(ker_Aqk.shape),
            "dtype": str(Q_INPUT.dtype),
            "device": str(Q_INPUT.device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff_A": diff_A.max().item(),
            "max_abs_diff_Aqk": diff_Aqk.max().item(),
            "grid": (1, NC * NC, 1),
        }
        pt_stats = _bench(lambda: pytorch_ref(Q_INPUT, K_INPUT, G_INPUT, BETA_INPUT, SCALE))
        kern_stats = _bench(lambda: kernel_impl(Q_INPUT, K_INPUT, G_INPUT, BETA_INPUT, SCALE))
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
            "Kernel: chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}, k shape: {stats['k_shape']}\n")
            lines.append(f"- BT={BT}, BC={BC}, NC={NC}, BK={BK}, scale={SCALE}\n")
            lines.append(f"- dtype: {stats['dtype']}, device: {stats['device']}\n")
            lines.append(f"- grid: {stats['grid']}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- A shape: {stats['A_shape']}, Aqk shape: {stats['Aqk_shape']}\n")
            lines.append(f"- max_abs_diff (overall): {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff (overall): {stats['mean_abs_diff']}\n")
            lines.append(f"- max_abs_diff_A: {stats['max_abs_diff_A']}\n")
            lines.append(f"- max_abs_diff_Aqk: {stats['max_abs_diff_Aqk']}\n")
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
