"""
Standalone QAIC validation for `kda_gate_cumsum_fwd_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - kda_gate_cumsum_fwd_kernel  (Fused kernel computing the KDA softplus
    decay gate followed by an in-chunk causal cumulative sum.)

Launcher under test:
  `fused_kda_gate_chunk_cumsum(raw_g, A_log, g_bias, beta, threshold,
   cu_seqlens, chunk_indices, chunk_size, output_dtype)`.

Semantics (per source):
  raw_g: [B, T, H, D]
  b_a = -exp(A_log[h])                      (scalar per head)
  b_g = raw_g (+ g_bias[h, :] if provided)
  softplus(b_g * beta, threshold) / beta     (linear-threshold switch)
  b_gate = b_a * softplus_result
  y = causal_cumsum_within_chunk(b_gate) * RCP_LN2   (matmul w/ lower-tri
      ones matrix, i.e. y[t] = sum_{s<=t, same chunk} b_gate[s] * RCP_LN2)
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_kda_gate_cumsum_fwd_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# kda_gate_cumsum_fwd_kernel is wrapped in @triton.autotune. Triton's generic
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

from vllm.model_executor.layers.fla.ops.kda import (  # noqa: E402
    fused_kda_gate_chunk_cumsum,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
torch.manual_seed(42)

RCP_LN2 = 1.0 / math.log(2.0)

B, T, H, D = 1, 64, 1, 16  # T == chunk_size (single chunk)
CHUNK_SIZE = 64
BETA = 1.0
THRESHOLD = 20.0

RAW_G = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
A_LOG = torch.randn(H, dtype=torch.float32, device=DEVICE)
G_BIAS = None  # HAS_BIAS = False


def pytorch_ref(raw_g, A_log, g_bias=None, beta=1.0, threshold=20.0, chunk_size=64):
    """Pure PyTorch reference for `kda_gate_cumsum_fwd_kernel`."""
    Bn, Tn, Hn, Dn = raw_g.shape
    g = raw_g.to(torch.float32)
    if g_bias is not None:
        g = g + g_bias.view(1, 1, Hn, Dn)

    A_log_r = A_log.reshape(-1)
    b_a = -torch.exp(A_log_r.to(torch.float32)).view(1, 1, Hn, 1)

    g_scaled = g * beta
    use_linear = g_scaled > threshold
    softplus = torch.where(
        use_linear,
        g,
        (1.0 / beta) * torch.log1p(torch.exp(g_scaled)),
    )
    gate = b_a * softplus  # [B, T, H, D]

    NT = (Tn + chunk_size - 1) // chunk_size
    y = torch.empty_like(gate)
    for it in range(NT):
        t0 = it * chunk_size
        t1 = min(t0 + chunk_size, Tn)
        chunk = gate[:, t0:t1, :, :]  # [B, bt, H, D]
        bt = t1 - t0
        tri = torch.tril(torch.ones(bt, bt, dtype=torch.float32, device=gate.device))
        # causal cumsum within chunk along time: y[t] = sum_{s<=t} chunk[s]
        # chunk shape [B, bt, H, D] -> matmul over time dim per (B,H,D)
        chunk_bhtd = chunk.permute(0, 2, 3, 1)  # [B, H, D, bt]
        y_bhtd = chunk_bhtd @ tri.T  # [B,H,D,bt] @ [bt,bt] -> sum_{s<=t}
        y_chunk = y_bhtd.permute(0, 3, 1, 2)  # [B, bt, H, D]
        y[:, t0:t1, :, :] = y_chunk * RCP_LN2
    return y


def kernel_impl(raw_g, A_log, g_bias=None, beta=1.0, threshold=20.0, chunk_size=64):
    return fused_kda_gate_chunk_cumsum(
        raw_g,
        A_log=A_log,
        g_bias=g_bias,
        beta=beta,
        threshold=threshold,
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
            RAW_G, A_LOG, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD, chunk_size=CHUNK_SIZE
        )
        kernel_out = kernel_impl(
            RAW_G, A_LOG, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD, chunk_size=CHUNK_SIZE
        )

        ref_cpu = ref_out.detach().cpu()
        kernel_cpu = kernel_out.detach().cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-6)).mean().item()
        stats = {
            "input_shape": tuple(RAW_G.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(RAW_G.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(RAW_G.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": rel_err,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                RAW_G, A_LOG, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD, chunk_size=CHUNK_SIZE
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                RAW_G, A_LOG, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD, chunk_size=CHUNK_SIZE
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
            "Kernel: kda_gate_cumsum_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- raw_g shape: {stats['input_shape']}, dtype: {stats['input_dtype']}\n")
            lines.append(f"- A_log shape: {tuple(A_LOG.shape)}\n")
            lines.append(f"- chunk_size: {CHUNK_SIZE}, beta: {BETA}, threshold: {THRESHOLD}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}, dtype: {stats['output_dtype']}\n")
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
