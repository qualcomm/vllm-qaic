"""
Standalone QAIC validation for `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra`.

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra
    (Computes the WITHIN-sub-chunk (i_i == i_j, strictly causal j < i_local)
    terms of the same A/Aqk matrices as the inter-sub-chunk kernel, via a
    sequential inner loop over local sub-chunk positions.)

Dispatch note: this kernel and `..._intra_sub_inter` are ALWAYS launched
together by the higher-level `chunk_kda_scaled_dot_kkt_fwd(...)`; there is
no launcher parameter that isolates just this kernel. To keep this test
targeted at a single kernel launch, we invoke the raw triton-jitted kernel
object directly, replicating exactly the grid/kwargs that
`chunk_kda_scaled_dot_kkt_fwd` uses for this specific kernel
(grid=(NT, NC, B*H), passing BK=next_power_of_2(K) explicitly since this
kernel's @triton.autotune configs vary only num_warps, not BK/BC/BT).

Semantics (per source, single chunk NT=1, i_t=0):
  For each sub-chunk i_i in [0, NC):
    row0 = i_i * BC  (start of sub-chunk within the T-length sequence)
    b_q, b_k, b_g = q/k/g rows [row0 : row0+BC]   (local rows indexed o_i)
    b_k *= beta[row0:row0+BC]                      (scale keys by beta)
    For local position j in [0, BC) (sequential loop, causal):
      b_kt = k[row0+j, :]   (the "j-th" key row within the sub-chunk)
      b_gk = g[row0+j, :]
      b_ktg[o_i, :] = b_kt[None,:] * exp2(b_g[o_i,:] - b_gk[None,:])
      b_A[o_i]   = sum_k(b_k[o_i,:] * b_ktg[o_i,:])   masked to o_i > j else 0
      b_Aqk[o_i] = sum_k(b_q[o_i,:] * b_ktg[o_i,:]) * scale, masked to o_i>=j else 0
      Store b_A into A[row0+o_i, row0+j], b_Aqk into Aqk[row0+o_i, row0+j]
      (i.e. writes column j of the (row0:row0+BC, row0:row0+BC) diagonal
      block of the [T, BT] matrices, for each local row o_i)

  Note the asymmetric causal masks: A uses strict `o_i > j` (b_A[j,j]==0,
  matching a strictly-lower-triangular structure combined with the
  inter-kernel's off-diagonal blocks), while Aqk uses `o_i >= j` (keeps the
  diagonal, since Aqk represents query-side attention which is causal
  inclusive of "self").
"""

import datetime
import os
import sys
import time
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra.txt"
)
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra is wrapped in
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
    chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra,
)
import triton  # noqa: E402

# Restrict the autotuner search space to a single config: this kernel's
# @triton.autotune only sweeps num_warps (key=["BK", "BT"]), so any
# individual config is numerically equivalent; picking one config avoids
# extra device launches that increase contention on this shared QAIC box.
chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra.fn.configs = [
    triton.runtime.autotuner.Config({}, num_warps=1)
]
chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra.fn.cache = {}

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
torch.manual_seed(43)

B, T, H, K = 1, 64, 1, 32  # K <= 256 required
BT = T  # single chunk, NT = 1
BC = min(16, BT)  # 16
NC = (BT + BC - 1) // BC  # 4
BK = 1 << (K - 1).bit_length()  # next_power_of_2(K) -> 32
SCALE = 1.0 / (K ** 0.5)

Q_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
K_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
G_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
BETA_INPUT = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE) * 0.5 + 0.5


def pytorch_ref(q, k, g, beta, scale, BT_=BT, BC_=BC, NC_=NC):
    """Pure PyTorch reference for the intra-sub-chunk (i_i == i_j) blocks."""
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
                row0 = i_i * BC_
                b_q = q_bh[row0 : row0 + BC_, :]  # [BC, K]
                b_k = k_bh[row0 : row0 + BC_, :] * beta_bh[row0 : row0 + BC_][:, None]
                b_g = g_bh[row0 : row0 + BC_, :]  # [BC, K]

                for j in range(BC_):
                    b_kt = k_bh[row0 + j, :]  # [K]
                    b_gk = g_bh[row0 + j, :]  # [K]
                    b_ktg = b_kt[None, :] * torch.exp2(b_g - b_gk[None, :])  # [BC, K]

                    b_A_col = (b_k * b_ktg).sum(dim=1)  # [BC]
                    o_i = torch.arange(BC_, device=q.device)
                    b_A_col = torch.where(o_i > j, b_A_col, torch.zeros_like(b_A_col))

                    b_Aqk_col = (b_q * b_ktg).sum(dim=1) * scale  # [BC]
                    b_Aqk_col = torch.where(
                        o_i >= j, b_Aqk_col, torch.zeros_like(b_Aqk_col)
                    )

                    A[b, row0 : row0 + BC_, h, row0 + j] = b_A_col
                    Aqk[b, row0 : row0 + BC_, h, row0 + j] = b_Aqk_col

    return A, Aqk


def kernel_impl(q, k, g, beta, scale, BT_=BT, BC_=BC, BK_=BK):
    Bn, Tn, Hn, Kn = q.shape
    NT = 1
    grid = (NT, BT_ // BC_, Bn * Hn)

    # This shared QAIC box intermittently returns transient device-level
    # errors ("Failed to synchronize stream", "Failed to throttle stream
    # submission", etc.) under contention from other concurrent processes
    # on the same NSPs -- unrelated to kernel correctness. Retry the launch
    # a bounded number of times; this is launch-robustness infra, not
    # reference/validation logic.
    last_err = None
    for _attempt in range(400):
        try:
            A = torch.zeros(Bn, Tn, Hn, BT_, dtype=torch.float32, device=q.device)
            Aqk = torch.zeros(Bn, Tn, Hn, BT_, dtype=torch.float32, device=q.device)
            chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra[grid](
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
                BK=BK_,
            )
            _ = A.abs().sum().item()  # force sync / surface device errors now
            return A, Aqk
        except RuntimeError as e:
            last_err = e
            time.sleep(10)
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
            "grid": (1, NC, 1),
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
            "Kernel: chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra\n",
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
