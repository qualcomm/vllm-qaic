"""
Standalone QAIC validation for `fused_olmo_hybrid_gdn_gating_kernel`.

Source under test:
vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py
  - fused_olmo_hybrid_gdn_gating_kernel  (Fused kernel computing the
    softplus-based decay gate `g` and sigmoid beta-output gate for OLMo
    hybrid GDN, with an optional allow_neg_eigval doubling of beta.)

Per (batch, head):
    x = a + dt_bias
    softplus_x = x if beta*x > threshold else (1/beta) * log(1 + exp(beta*x))
    g = -exp(A_log) * softplus_x
    beta_output = sigmoid(b)
    if allow_neg_eigval: beta_output *= 2.0

Reference: pure PyTorch reimplementation of the above (no triton/vllm calls).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_olmo_hybrid_gdn_gating_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/gdn/olmo_gdn_linear_attn.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (  # noqa: E402
    fused_olmo_hybrid_gdn_gating,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
BATCH = 4
NUM_HEADS = 8
ALLOW_NEG_EIGVAL = False
BETA = 1.0
THRESHOLD = 20.0

torch.manual_seed(42)

A_LOG = torch.randn(NUM_HEADS, dtype=torch.float32, device=DEVICE)
A_INPUT = torch.randn(BATCH, NUM_HEADS, dtype=torch.float32, device=DEVICE)
B_INPUT = torch.randn(BATCH, NUM_HEADS, dtype=torch.float32, device=DEVICE)
DT_BIAS = torch.randn(NUM_HEADS, dtype=torch.float32, device=DEVICE)


def pytorch_ref(A_log, a, b, dt_bias, allow_neg_eigval, beta, threshold):
    """Pure PyTorch reference implementation.

    Requirements (Claude.md):
      - Pure PyTorch only.
      - No custom kernel calls.
      - No Triton kernel calls.
      - No vLLM kernel calls.
      - No QAIC custom operator calls.
    """
    A_log = A_log.float()
    a = a.float()
    b = b.float()
    dt_bias = dt_bias.float()

    x = a + dt_bias.unsqueeze(0)
    beta_x = beta * x
    softplus_x = torch.where(
        beta_x <= threshold,
        (1.0 / beta) * torch.log(1.0 + torch.exp(beta_x)),
        x,
    )
    g = -torch.exp(A_log).unsqueeze(0) * softplus_x
    g = g.unsqueeze(0)  # [1, batch, num_heads]

    beta_output = torch.sigmoid(b)
    if allow_neg_eigval:
        beta_output = beta_output * 2.0
    beta_output = beta_output.unsqueeze(0)  # [1, batch, num_heads]

    return g, beta_output


def kernel_impl(A_log, a, b, dt_bias, allow_neg_eigval, beta, threshold):
    """Kernel wrapper: launch only.

    Requirements (Claude.md):
      - Kernel launch only.
      - Minimal setup logic.
      - No reference implementation logic.
      - No correctness-check logic.
      - No validation logic.
    """
    return fused_olmo_hybrid_gdn_gating(
        A_log, a, b, dt_bias, allow_neg_eigval=allow_neg_eigval,
        beta=beta, threshold=threshold,
    )


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_g, ref_beta = pytorch_ref(
            A_LOG, A_INPUT, B_INPUT, DT_BIAS, ALLOW_NEG_EIGVAL, BETA, THRESHOLD
        )
        kernel_g, kernel_beta = kernel_impl(
            A_LOG, A_INPUT, B_INPUT, DT_BIAS, ALLOW_NEG_EIGVAL, BETA, THRESHOLD
        )

        ref_g_cpu = ref_g.cpu()
        ref_beta_cpu = ref_beta.cpu()
        kernel_g_cpu = kernel_g.cpu()
        kernel_beta_cpu = kernel_beta.cpu()

        torch.testing.assert_close(kernel_g_cpu, ref_g_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            kernel_beta_cpu, ref_beta_cpu, rtol=1e-3, atol=1e-3
        )

        diff_g = (kernel_g_cpu - ref_g_cpu).abs()
        diff_beta = (kernel_beta_cpu - ref_beta_cpu).abs()
        max_abs_diff = max(diff_g.max().item(), diff_beta.max().item())
        mean_abs_diff = (diff_g.mean().item() + diff_beta.mean().item()) / 2.0
        rel_err_g = (
            diff_g.max() / (ref_g_cpu.abs().max() + 1e-8)
        ).item()
        rel_err_beta = (
            diff_beta.max() / (ref_beta_cpu.abs().max() + 1e-8)
        ).item()

        stats = {
            "a_shape": tuple(A_INPUT.shape),
            "b_shape": tuple(B_INPUT.shape),
            "A_log_shape": tuple(A_LOG.shape),
            "dt_bias_shape": tuple(DT_BIAS.shape),
            "g_shape": tuple(kernel_g.shape),
            "beta_output_shape": tuple(kernel_beta.shape),
            "dtype": str(A_INPUT.dtype),
            "device": str(A_INPUT.device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "rel_err_g": rel_err_g,
            "rel_err_beta": rel_err_beta,
            "allow_neg_eigval": ALLOW_NEG_EIGVAL,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                A_LOG, A_INPUT, B_INPUT, DT_BIAS, ALLOW_NEG_EIGVAL, BETA, THRESHOLD
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                A_LOG, A_INPUT, B_INPUT, DT_BIAS, ALLOW_NEG_EIGVAL, BETA, THRESHOLD
            )
        )
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: fused_olmo_hybrid_gdn_gating_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- a shape: {stats['a_shape']}\n")
            lines.append(f"- b shape: {stats['b_shape']}\n")
            lines.append(f"- A_log shape: {stats['A_log_shape']}\n")
            lines.append(f"- dt_bias shape: {stats['dt_bias_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- allow_neg_eigval: {stats['allow_neg_eigval']}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- g shape: {stats['g_shape']}\n")
            lines.append(f"- beta_output shape: {stats['beta_output_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err (g): {stats['rel_err_g']}\n")
            lines.append(f"- rel_err (beta_output): {stats['rel_err_beta']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
