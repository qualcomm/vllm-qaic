"""
Standalone QAIC validation for `_fused_post_conv_kernel`.

Source under test:
vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py
  - _fused_post_conv_kernel  (fused post short-conv1d split/l2norm/gating
    for Gated DeltaNet prefill: q/k/v split + optional L2 normalization of
    q,k + softplus-gating for g + sigmoid for beta, all in one kernel)

conv_output: [L, qkv_dim], qkv_dim = 2*H*K + HV*V.
  q = conv_output[:, :H*K].view(L, H, K)
  k = conv_output[:, H*K:2*H*K].view(L, H, K)
  v = conv_output[:, 2*H*K:].view(L, HV, V)
If apply_l2norm: q, k are L2-normalized along the last dim with eps=1e-6:
  x / sqrt(sum(x^2) + eps)
Gating (per token, per HV-head):
  x = a + dt_bias
  softplus(x) = x if x > threshold(20.0) else log(1+exp(x))
  g = -exp(A_log) * softplus(x); if output_g_exp: g = exp(g)
  beta = sigmoid(b)

Reference: pure PyTorch reimplementation of the same split/l2norm/gating
logic. Five outputs (q, k, v, g, beta) are all compared; the worst-case
(max) abs diff/mean abs diff across all five feeds the reported stats.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_post_conv_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.model_executor.layers.fla.ops.fused_gdn_prefill_post_conv import (  # noqa: E402
    fused_post_conv_prep,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"

L = 32
H = 2  # num_k_heads
K = 16  # head_k_dim
V = 16  # head_v_dim
HV = 4  # number of "value" heads (A_log.numel())
QKV_DIM = 2 * H * K + HV * V  # 2*2*16 + 4*16 = 128
APPLY_L2NORM = True
OUTPUT_G_EXP = False

torch.manual_seed(42)
CONV_OUTPUT = torch.randn(L, QKV_DIM, dtype=torch.float32, device=DEVICE)
A = torch.randn(L, HV, dtype=torch.float32, device=DEVICE)
B_GATE = torch.randn(L, HV, dtype=torch.float32, device=DEVICE)
A_LOG = torch.randn(HV, dtype=torch.float32, device=DEVICE)
DT_BIAS = torch.randn(HV, dtype=torch.float32, device=DEVICE)


def pytorch_ref(
    conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
    apply_l2norm=True, output_g_exp=False,
):
    """Pure PyTorch reference for the fused post-conv1d prep."""
    L_ = conv_output.shape[0]
    H_, K_, V_ = num_k_heads, head_k_dim, head_v_dim
    HV_ = A_log.shape[0]

    q = conv_output[:, : H_ * K_].reshape(L_, H_, K_).clone()
    k = conv_output[:, H_ * K_ : 2 * H_ * K_].reshape(L_, H_, K_).clone()
    v = conv_output[:, 2 * H_ * K_ :].reshape(L_, HV_, V_).clone()

    if apply_l2norm:
        eps = 1e-6
        q_sq = (q.to(torch.float32) ** 2).sum(dim=-1, keepdim=True)
        q = (q.to(torch.float32) / torch.sqrt(q_sq + eps)).to(q.dtype)
        k_sq = (k.to(torch.float32) ** 2).sum(dim=-1, keepdim=True)
        k = (k.to(torch.float32) / torch.sqrt(k_sq + eps)).to(k.dtype)

    x = a.to(torch.float32) + dt_bias.to(torch.float32)
    threshold = 20.0
    sp = torch.where(x > threshold, x, F.softplus(x))
    g = -torch.exp(A_log.to(torch.float32)) * sp
    if output_g_exp:
        g = torch.exp(g)
    beta = torch.sigmoid(b.to(torch.float32))

    return q, k, v, g, beta


def kernel_impl(
    conv_output, a, b, A_log, dt_bias, num_k_heads, head_k_dim, head_v_dim,
    apply_l2norm=True, output_g_exp=False,
):
    return fused_post_conv_prep(
        conv_output, a, b, A_log, dt_bias,
        num_k_heads, head_k_dim, head_v_dim,
        apply_l2norm=apply_l2norm, output_g_exp=output_g_exp,
    )


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
        ref_q, ref_k, ref_v, ref_g, ref_beta = pytorch_ref(
            CONV_OUTPUT, A, B_GATE, A_LOG, DT_BIAS, H, K, V,
            apply_l2norm=APPLY_L2NORM, output_g_exp=OUTPUT_G_EXP,
        )
        ker_q, ker_k, ker_v, ker_g, ker_beta = kernel_impl(
            CONV_OUTPUT, A, B_GATE, A_LOG, DT_BIAS, H, K, V,
            apply_l2norm=APPLY_L2NORM, output_g_exp=OUTPUT_G_EXP,
        )

        names = ["q", "k", "v", "g", "beta"]
        ref_outs = [ref_q, ref_k, ref_v, ref_g, ref_beta]
        ker_outs = [ker_q, ker_k, ker_v, ker_g, ker_beta]

        per_output_stats = {}
        max_abs_diff = 0.0
        mean_abs_diffs = []
        worst_rel_err = 0.0

        for name, ref_o, ker_o in zip(names, ref_outs, ker_outs):
            ref_cpu = ref_o.cpu()
            ker_cpu = ker_o.cpu()
            torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
            diff = (ker_cpu - ref_cpu).abs()
            rel_err = (diff / (ref_cpu.abs() + 1e-8)).mean().item()
            per_output_stats[name] = {
                "shape": tuple(ker_o.shape),
                "dtype": str(ker_o.dtype),
                "max_abs_diff": diff.max().item(),
                "mean_abs_diff": diff.mean().item(),
                "rel_error": rel_err,
            }
            max_abs_diff = max(max_abs_diff, diff.max().item())
            mean_abs_diffs.append(diff.mean().item())
            worst_rel_err = max(worst_rel_err, rel_err)

        stats = {
            "input_shape": tuple(CONV_OUTPUT.shape),
            "input_dtype": str(CONV_OUTPUT.dtype),
            "device": str(CONV_OUTPUT.device),
            "per_output": per_output_stats,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": sum(mean_abs_diffs) / len(mean_abs_diffs),
            "rel_error": worst_rel_err,
        }
        pt_stats = _bench(lambda: pytorch_ref(
            CONV_OUTPUT, A, B_GATE, A_LOG, DT_BIAS, H, K, V,
            apply_l2norm=APPLY_L2NORM, output_g_exp=OUTPUT_G_EXP))
        kern_stats = _bench(lambda: kernel_impl(
            CONV_OUTPUT, A, B_GATE, A_LOG, DT_BIAS, H, K, V,
            apply_l2norm=APPLY_L2NORM, output_g_exp=OUTPUT_G_EXP))
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
            "Kernel: _fused_post_conv_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- conv_output shape: {stats['input_shape']}\n")
            lines.append(f"- input dtype: {stats['input_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(
                f"- H={H} K={K} V={V} HV={HV} apply_l2norm={APPLY_L2NORM} "
                f"output_g_exp={OUTPUT_G_EXP}\n\n"
            )
            lines.append("Per-output stats:\n")
            for name, s in stats["per_output"].items():
                lines.append(
                    f"- {name}: shape={s['shape']} dtype={s['dtype']} "
                    f"max_abs_diff={s['max_abs_diff']} "
                    f"mean_abs_diff={s['mean_abs_diff']} "
                    f"rel_error={s['rel_error']}\n"
                )
            lines.append("\nOverall (worst-case across outputs):\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_error: {stats['rel_error']}\n")
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
