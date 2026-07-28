"""
Standalone QAIC validation for `_silu_mul_fp8_quant_deep_gemm`.

Source under test:
vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py
  - _silu_mul_fp8_quant_deep_gemm  (@triton.jit)
  - persistent_masked_m_silu_mul_quant  (launcher; Triton fallback path)

The kernel takes batched MoE intermediate activations y of shape (E, T, 2*H).
For each expert e and valid token t it applies SwiGLU:
    gate = y[e, t, :H];  up = y[e, t, H:]
    out  = silu(gate) * up = (gate * sigmoid(gate)) * up
then quantizes `out` to FP8 with a per-(expert, token, group) absmax scale
(GROUP_SIZE = 128 elements per group along H):
    y_s = max(absmax(out_group), eps) * (1 / fp8_max)
    y_q = clamp(out / y_s, fp8_min, fp8_max)  cast to fp8
Outputs: y_q (E, T, H) fp8 and y_s (E, T, G) fp32.

Comparison choice: the PyTorch reference performs the IDENTICAL per-group
absmax FP8 quantization (same eps, fp8_min/max, same fp8 dtype), so we compare
(a) the FP8 codes y_q (viewed as float) and (b) the fp32 scales y_s directly
under 1e-3 tolerances -- rather than dequantizing against a full-precision
value (which would exceed FP8 quant error).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_silu_mul_fp8_quant_deep_gemm.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py"
)

# Global inputs (tiny). GROUP_SIZE is fixed at 128 by the kernel, and H must be
# a multiple of 128, so use H = 128 (single group, G = 1).
NUM_EXPERTS = 2
NUM_TOKENS = 4
H = 128
GROUP_SIZE = 128
EPS = 1e-10
DEVICE = "qaic"

_IS_CHILD = os.environ.get("SMFQ_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        persistent_masked_m_silu_mul_quant,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        get_fp8_min_max,
    )
    from vllm.platforms import current_platform

    torch.manual_seed(42)
    # y: (E, T, 2*H)
    Y = torch.randn(NUM_EXPERTS, NUM_TOKENS, 2 * H, dtype=torch.float32, device=DEVICE)
    # All tokens valid for every expert.
    TOKENS_PER_EXPERT = torch.full(
        (NUM_EXPERTS,), NUM_TOKENS, dtype=torch.int32, device=DEVICE
    )
    FP8_DTYPE = current_platform.fp8_dtype()
    FP8_MIN, FP8_MAX = get_fp8_min_max()


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


def pytorch_ref(y, tokens_per_expert):
    """Pure PyTorch SwiGLU + per-group absmax FP8 quantization.

    Returns (y_q_float, y_s) matching the kernel's outputs:
      * y_q_float: fp8 codes cast back to float32, shape (E, T, H)
      * y_s:       fp32 per-group scales, shape (E, T, G)
    """
    y = y.cpu()
    E, T, H2 = y.shape
    G = H // GROUP_SIZE

    gate = y[:, :, :H]
    up = y[:, :, H:]
    out = (gate * torch.sigmoid(gate)) * up  # silu(gate) * up  -> (E, T, H)

    y_q = torch.zeros(E, T, H, dtype=torch.float32)
    y_s = torch.zeros(E, T, G, dtype=torch.float32)
    for e in range(E):
        n = int(tokens_per_expert[e].item())
        for t in range(n):
            for g in range(G):
                grp = out[e, t, g * GROUP_SIZE : (g + 1) * GROUP_SIZE]
                absmax = grp.abs().max()
                scale = torch.maximum(
                    absmax, torch.tensor(EPS, dtype=torch.float32)
                ) * (1.0 / FP8_MAX)
                q = torch.clamp(grp / scale, FP8_MIN, FP8_MAX)
                # Round to fp8 grid to match the kernel's fp8 store.
                q_fp8 = q.to(FP8_DTYPE).to(torch.float32)
                y_q[e, t, g * GROUP_SIZE : (g + 1) * GROUP_SIZE] = q_fp8
                y_s[e, t, g] = scale
    return y_q, y_s


def kernel_impl(y, tokens_per_expert):
    y_q, y_s = persistent_masked_m_silu_mul_quant(
        y.clone(),
        tokens_per_expert.clone(),
        group_size=GROUP_SIZE,
    )
    # y_q is fp8; view as float for comparison. y_s is fp32.
    return y_q.to(torch.float32), y_s.to(torch.float32)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_yq, ref_ys = pytorch_ref(Y, TOKENS_PER_EXPERT)
        kern_yq, kern_ys = kernel_impl(Y, TOKENS_PER_EXPERT)

        ref_yq = ref_yq.cpu()
        ref_ys = ref_ys.cpu()
        kern_yq = kern_yq.cpu()
        kern_ys = kern_ys.cpu()

        torch.testing.assert_close(kern_yq, ref_yq, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kern_ys, ref_ys, rtol=1e-3, atol=1e-3)

        diff_q = (kern_yq - ref_yq).abs()
        diff_s = (kern_ys - ref_ys).abs()
        stats = {
            "input_shape": tuple(Y.shape),
            "yq_shape": tuple(kern_yq.shape),
            "ys_shape": tuple(kern_ys.shape),
            "dtype": str(Y.dtype),
            "fp8_dtype": str(FP8_DTYPE),
            "device": str(Y.device),
            "max_abs_diff_yq": diff_q.max().item(),
            "mean_abs_diff_yq": diff_q.mean().item(),
            "max_abs_diff_ys": diff_s.max().item(),
            "mean_abs_diff_ys": diff_s.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(Y, TOKENS_PER_EXPERT))
        kern_stats = _bench(lambda: kernel_impl(Y, TOKENS_PER_EXPERT))
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
            "Kernel: _silu_mul_fp8_quant_deep_gemm\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- y shape: {stats['input_shape']}\n")
            lines.append(
                f"- num_experts={NUM_EXPERTS}, tokens={NUM_TOKENS}, "
                f"H={H}, group_size={GROUP_SIZE}\n"
            )
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- fp8_dtype: {stats['fp8_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- y_q shape: {stats['yq_shape']}\n")
            lines.append(f"- y_s shape: {stats['ys_shape']}\n")
            lines.append(f"- max_abs_diff (y_q codes): {stats['max_abs_diff_yq']}\n")
            lines.append(f"- mean_abs_diff (y_q codes): {stats['mean_abs_diff_yq']}\n")
            lines.append(f"- max_abs_diff (y_s scales): {stats['max_abs_diff_ys']}\n")
            lines.append(f"- mean_abs_diff (y_s scales): {stats['mean_abs_diff_ys']}\n")
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


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("SMFQ_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, SMFQ_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: _silu_mul_fp8_quant_deep_gemm\n"
            f"Kernel file: {KERNEL_FILE_PATH}\n"
            f"Device target: QAIC (device='{DEVICE}')\n"
            "Status: FAILURE\n\n"
            "Error:\n"
            f"Child killed by signal (exit {proc.returncode}) during "
            "Triton->Hexagon compile/execution.\n"
            "\n------------------------------------\n\n"
        )
    sys.exit(proc.returncode if proc.returncode >= 0 else 1)


if __name__ == "__main__":
    _run_with_crash_guard()
