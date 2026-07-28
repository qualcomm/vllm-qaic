"""
Standalone QAIC validation for `chunk_fwd_kernel_o`.

Source under test:
vllm/model_executor/layers/fla/ops/chunk_o.py
  - chunk_fwd_kernel_o  (chunk-parallel forward output for gated linear /
    delta-rule attention: intra-chunk causal attention + inter-chunk term
    using the incoming per-chunk recurrent state `h`, with optional
    per-token log-decay gating `g`.)

For a single chunk (B=1, T=64=chunk_size, Hg=1, H=2, K=32, V=32):
    A = Q @ K^T                              (intra-chunk scores)
    o_inter = Q @ h^T                         (inter-chunk term via state h)
    if g is not None:
        o_inter *= exp(g_i)
        A *= exp(g_i - g_j)
    A = causal_mask(A)  (strictly j <= i)
    o = scale * (A @ V) + scale * o_inter

GQA: Q/K use Hg head-groups, V/O use H heads, with H % Hg == 0.

Reference: pure PyTorch chunked loop over batch/head reproducing the above
exactly in fp32, then cast back to the input dtype.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_chunk_fwd_kernel_o.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/chunk_o.py"

DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing  # noqa: E402


def _qaic_safe_do_bench(fn, warmup=25, rep=100, grad_to_none=None,
                         quantiles=None, return_mode="mean"):
    """Wall-clock replacement for `triton.testing.do_bench`.

    The stock implementation times kernels via `torch.Event.elapsed_time`,
    which is broken for the QAIC backend in this environment
    (`RuntimeError: expected other to be a torch.Event object`). Triton's
    `@triton.autotune` decorator calls this during its very first config
    search, so every autotuned FLA kernel hits this crash before any of our
    kernel-under-test logic even runs. We swap in a simple `time.perf_counter`
    based timer (device-synced) so autotuning can proceed; this only affects
    *which* config is picked / timed, not kernel correctness.
    """
    import time

    fn()
    torch.qaic.synchronize()
    n_repeat = 3
    times = []
    for _ in range(n_repeat):
        start = time.perf_counter()
        fn()
        torch.qaic.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    if quantiles is not None:
        import numpy as np

        return list(np.quantile(times, quantiles))
    return sum(times) / len(times)


triton.testing.do_bench = _qaic_safe_do_bench

from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o  # noqa: E402

# ---------------------------------------------------------------------------
# Global inputs (single chunk: B=1, T=64=chunk_size, Hg=1, H=2, K=32, V=32)
# ---------------------------------------------------------------------------
B = 1
T = 64
Hg = 1
H = 2
K = 32
V = 32
CHUNK_SIZE = 64
NT = 1  # cdiv(T, CHUNK_SIZE)
DTYPE = torch.float32
SCALE = K**-0.5

torch.manual_seed(42)
Q = torch.randn(B, T, Hg, K, dtype=DTYPE, device=DEVICE) * 0.1
KEY = torch.randn(B, T, Hg, K, dtype=DTYPE, device=DEVICE) * 0.1
VAL = torch.randn(B, T, H, V, dtype=DTYPE, device=DEVICE) * 0.1
STATE_H = torch.randn(B, NT, H, V, K, dtype=DTYPE, device=DEVICE) * 0.1
# Per-(token, head) cumulative log-decay within the chunk (monotonically
# non-increasing so that exp(g_i - g_j) with i>=j decays smoothly).
_log_decay = -0.02 * torch.rand(B, T, H, dtype=DTYPE, device=DEVICE)
G = torch.cumsum(_log_decay, dim=1)


def pytorch_ref(q, k, v, h, g, scale):
    """Pure PyTorch reference for chunk_fwd_kernel_o (single chunk, GQA)."""
    b_, t_, hg_, k_ = q.shape
    h_ = v.shape[2]
    v_ = v.shape[3]
    rep = h_ // hg_

    qf = q.float()
    kf = k.float()
    vf = v.float()
    hf = h.float()
    gf = g.float() if g is not None else None

    out = torch.zeros(b_, t_, h_, v_, dtype=torch.float32, device=q.device)
    causal_mask = torch.tril(torch.ones(t_, t_, dtype=torch.bool, device=q.device))

    for bi in range(b_):
        for hh in range(h_):
            hg_idx = hh // rep
            Qb = qf[bi, :, hg_idx, :]  # [T, K]
            Kb = kf[bi, :, hg_idx, :]  # [T, K]
            Vb = vf[bi, :, hh, :]  # [T, V]
            Hb = hf[bi, 0, hh, :, :]  # [V, K]  (single chunk -> NT=1)

            A = Qb @ Kb.transpose(0, 1)  # [T, T]
            o_inter = Qb @ Hb.transpose(0, 1)  # [T, K] @ [K, V] -> [T, V]

            if gf is not None:
                g_ = gf[bi, :, hh]  # [T]
                o_inter = o_inter * torch.exp(g_)[:, None]
                A = A * torch.exp(g_[:, None] - g_[None, :])

            A = torch.where(causal_mask, A, torch.zeros_like(A))
            o_intra = A @ Vb  # [T, T] @ [T, V] -> [T, V]

            out[bi, :, hh, :] = scale * o_intra + scale * o_inter

    return out.to(v.dtype)


def kernel_impl(q, k, v, h, g, scale):
    return chunk_fwd_o(q, k, v, h, g=g, scale=scale, chunk_size=CHUNK_SIZE)


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
        ref_out = pytorch_ref(Q, KEY, VAL, STATE_H, G, SCALE)
        kernel_out = kernel_impl(Q, KEY, VAL, STATE_H, G, SCALE)

        ref_cpu = ref_out.cpu().float()
        kernel_cpu = kernel_out.cpu().float()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-6)).mean().item()

        stats = {
            "input_shapes": {
                "q": tuple(Q.shape),
                "k": tuple(KEY.shape),
                "v": tuple(VAL.shape),
                "h": tuple(STATE_H.shape),
                "g": tuple(G.shape),
            },
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(Q.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(Q.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": rel_err,
        }
        pt_stats = _bench(lambda: pytorch_ref(Q, KEY, VAL, STATE_H, G, SCALE))
        kern_stats = _bench(lambda: kernel_impl(Q, KEY, VAL, STATE_H, G, SCALE))
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
            "Kernel: chunk_fwd_kernel_o\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            for name, shape in stats["input_shapes"].items():
                lines.append(f"- {name} shape: {shape}\n")
            lines.append(f"- dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- shape: {stats['output_shape']}\n")
            lines.append(f"- dtype: {stats['output_dtype']}\n")
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
