"""
Standalone QAIC validation for `chunk_gla_fwd_kernel_o`.

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - chunk_gla_fwd_kernel_o  (Computes the chunked gated-linear-attention
    output combining an inter-chunk contribution from the recurrent state h
    and an intra-chunk contribution from a precomputed causally-masked
    attention matrix A.)

Launcher under test:
  `chunk_gla_fwd_o_gk(q, v, g, A, h, o, scale, cu_seqlens=None,
   chunk_indices=None, chunk_size=64)`.

Semantics (per source, single chunk BT == T, NT == 1):
  b_qg = (q * scale) * exp2(g)                         # [T, K]
  o_inter[t, v] = sum_k b_qg[t, k] * h[v, k]            # h stored as [V, K]
  o_intra = tril_mask(A) @ v                            # A already has
                                                         # scale baked in
                                                         # from the caller
  o = o_inter + o_intra

Calling convention (matches the real caller `_chunk_kda_fwd_with_cumulative_g`
in kda.py, which invokes `chunk_gla_fwd_o_gk(..., o=v, ...)` -- i.e. it
passes the *original* v tensor as the `o` output buffer to be overwritten,
since the kernel writes via tl.store (no read-before-write on `o`), so the
buffer's prior contents don't matter. We replicate this exactly: `o` is a
separate pre-allocated buffer (values immaterial) passed positionally as
the output target, distinct from `v` used as input.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_gla_fwd_kernel_o.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# chunk_gla_fwd_kernel_o is wrapped in @triton.autotune. Triton's generic
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

from vllm.model_executor.layers.fla.ops.kda import chunk_gla_fwd_o_gk  # noqa: E402

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
torch.manual_seed(42)

B, T, H, K, V = 1, 64, 1, 32, 32
CHUNK_SIZE = 64  # NT = 1
SCALE = 1.0 / (K ** 0.5)

Q_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE)
V_INPUT = torch.randn(B, T, H, V, dtype=torch.float32, device=DEVICE)
G_INPUT = torch.randn(B, T, H, K, dtype=torch.float32, device=DEVICE) * 0.1
A_INPUT = torch.randn(B, T, H, CHUNK_SIZE, dtype=torch.float32, device=DEVICE)
# Recurrent state h, stored per source layout [B, NT, H, V, K] (NT=1).
H_STATE = torch.randn(B, 1, H, V, K, dtype=torch.float32, device=DEVICE)
# Separate pre-allocated output buffer (distinct from V_INPUT), matching the
# real caller's `o=v` aliasing pattern where `o` is a buffer to overwrite.
O_BUFFER = torch.empty(B, T, H, V, dtype=torch.float32, device=DEVICE)


def pytorch_ref(q, v, g, A, h, o_buffer, scale, chunk_size=64):
    """Pure PyTorch reference for `chunk_gla_fwd_kernel_o`."""
    Bn, Tn, Hn, Kn = q.shape
    Vn = v.shape[-1]

    m_s = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device))

    out = torch.zeros(Bn, Tn, Hn, Vn, dtype=torch.float32, device=q.device)
    for b in range(Bn):
        for hh in range(Hn):
            q_bh = q[b, :, hh, :]  # [T, K]
            g_bh = g[b, :, hh, :]  # [T, K]
            v_bh = v[b, :, hh, :]  # [T, V]
            A_bh = A[b, :, hh, :]  # [T, BT]
            h_bh = h[b, 0, hh, :, :]  # [V, K]

            b_qg = (q_bh * scale) * torch.exp2(g_bh)  # [T, K]
            o_inter = b_qg @ h_bh.T  # [T, K] @ [K, V] -> [T, V]

            A_masked = torch.where(m_s, A_bh, torch.zeros_like(A_bh))
            o_intra = A_masked @ v_bh  # [T, BT] @ [BT, V] -> [T, V]

            out[b, :, hh, :] = o_inter + o_intra
    return out


def kernel_impl(q, v, g, A, h, o_buffer, scale, chunk_size=64):
    return chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g,
        A=A,
        h=h,
        o=o_buffer,
        scale=scale,
        chunk_size=chunk_size,
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
        ref_out = pytorch_ref(
            Q_INPUT, V_INPUT, G_INPUT, A_INPUT, H_STATE, O_BUFFER, SCALE, chunk_size=CHUNK_SIZE
        )
        kernel_out = kernel_impl(
            Q_INPUT, V_INPUT, G_INPUT, A_INPUT, H_STATE, O_BUFFER, SCALE, chunk_size=CHUNK_SIZE
        )

        ref_cpu = ref_out.detach().cpu()
        kernel_cpu = kernel_out.detach().cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-6)).mean().item()
        stats = {
            "q_shape": tuple(Q_INPUT.shape),
            "v_shape": tuple(V_INPUT.shape),
            "h_shape": tuple(H_STATE.shape),
            "A_shape": tuple(A_INPUT.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(Q_INPUT.dtype),
            "device": str(Q_INPUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": rel_err,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                Q_INPUT, V_INPUT, G_INPUT, A_INPUT, H_STATE, O_BUFFER, SCALE, chunk_size=CHUNK_SIZE
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                Q_INPUT, V_INPUT, G_INPUT, A_INPUT, H_STATE, O_BUFFER, SCALE, chunk_size=CHUNK_SIZE
            )
        )
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
            "Kernel: chunk_gla_fwd_kernel_o\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}, v shape: {stats['v_shape']}\n")
            lines.append(f"- h shape: {stats['h_shape']}, A shape: {stats['A_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}, device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err (mean): {stats['rel_err']}\n")
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
