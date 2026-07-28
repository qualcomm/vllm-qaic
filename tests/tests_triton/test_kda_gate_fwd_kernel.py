"""
Standalone QAIC validation for `kda_gate_fwd_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/kda.py
  - kda_gate_fwd_kernel  (Computes the KDA gate value via a softplus-based
    decay transform per head/token block: for input g reshaped to
    [-1, H, D], per-head b_a = -exp(A[h]), then
    softplus(g * beta, threshold) / beta, output = b_a * softplus_result.)

Launcher under test: `fused_kda_gate(g, A, head_k_dim, g_bias, beta, threshold)`.

Reference: pure PyTorch softplus (with linear-threshold switch) times
-exp(A) per head, no cumsum involved (unlike kda_gate_cumsum_fwd_kernel).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_kda_gate_fwd_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/kda.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import triton.testing as _triton_testing  # noqa: E402

# kda_gate_fwd_kernel is wrapped in @triton.autotune. Triton's generic
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

from vllm.model_executor.layers.fla.ops.kda import fused_kda_gate  # noqa: E402

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
torch.manual_seed(42)

T = 32
H = 2
D = 16  # head_k_dim
BETA = 1.0
THRESHOLD = 20.0

G_INPUT = torch.randn(T, H * D, dtype=torch.float32, device=DEVICE)
A_PARAM = torch.randn(H, dtype=torch.float32, device=DEVICE)
G_BIAS = None
HEAD_K_DIM = D


def pytorch_ref(g, A, head_k_dim, g_bias=None, beta=1.0, threshold=20.0):
    """Pure PyTorch reference for `kda_gate_fwd_kernel`.

    g: [..., H*D] -> reshaped to [-1, H, D]
    A: [H]
    output: [..., H, D]
    """
    orig_shape = g.shape[:-1]
    g2 = g.reshape(-1, g.shape[-1]).to(torch.float32)
    Tn = g2.shape[0]
    Hn = A.numel()
    Dn = head_k_dim
    g2 = g2.view(Tn, Hn, Dn)

    if g_bias is not None:
        g2 = g2 + g_bias.view(Hn, Dn)[None, :, :]

    b_a = -torch.exp(A.to(torch.float32)).view(1, Hn, 1)

    g_scaled = g2 * beta
    use_linear = g_scaled > threshold
    softplus = torch.where(
        use_linear,
        g2,
        (1.0 / beta) * torch.log1p(torch.exp(g_scaled)),
    )
    y = b_a * softplus
    return y.view(*orig_shape, Hn, Dn)


def kernel_impl(g, A, head_k_dim, g_bias=None, beta=1.0, threshold=20.0):
    return fused_kda_gate(g, A, head_k_dim, g_bias=g_bias, beta=beta, threshold=threshold)


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
            G_INPUT, A_PARAM, HEAD_K_DIM, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD
        )
        kernel_out = kernel_impl(
            G_INPUT, A_PARAM, HEAD_K_DIM, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD
        )

        ref_cpu = ref_out.detach().cpu()
        kernel_cpu = kernel_out.detach().cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        rel_err = (diff / (ref_cpu.abs() + 1e-6)).mean().item()
        stats = {
            "input_shape": tuple(G_INPUT.shape),
            "output_shape": tuple(kernel_out.shape),
            "input_dtype": str(G_INPUT.dtype),
            "output_dtype": str(kernel_out.dtype),
            "device": str(G_INPUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": rel_err,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                G_INPUT, A_PARAM, HEAD_K_DIM, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                G_INPUT, A_PARAM, HEAD_K_DIM, g_bias=G_BIAS, beta=BETA, threshold=THRESHOLD
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
            "Kernel: kda_gate_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- g shape: {stats['input_shape']}, dtype: {stats['input_dtype']}\n")
            lines.append(f"- A shape: {tuple(A_PARAM.shape)}\n")
            lines.append(f"- head_k_dim: {HEAD_K_DIM}, beta: {BETA}, threshold: {THRESHOLD}\n")
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
