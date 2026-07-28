"""
Standalone QAIC validation for `_fused_mtp_input_rmsnorm_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py
  - _fused_mtp_input_rmsnorm_kernel  (fused MTP-input enorm + hnorm RMSNorm)
  - launcher: fused_mtp_input_rmsnorm(inputs_embeds, positions,
        previous_hidden_states, enorm_weight, hnorm_weight, eps, hc_mult)

The kernel runs a grid (T, hc_mult+1). Task 0 is the enorm path: it zero-masks
inputs_embeds rows whose position == 0, then applies RMSNorm with enorm_weight.
Tasks 1..hc_mult are the hnorm path: RMSNorm of previous_hidden_states[:, k, :]
with hnorm_weight. RMSNorm is computed in fp32:
    y = x * rsqrt(mean(x^2) + eps) * w
Math is preserved for pos==0: x is zeroed -> variance 0 -> output 0.

Config tested: T=4 tokens (one at position 0), HIDDEN=128, hc_mult=2, bf16 IO.
Reference: pure PyTorch RMSNorm with the same masking. Two outputs
(enorm_out [T,H], hnorm_out [T,hc_mult,H]) are compared with assert_close.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_mtp_input_rmsnorm_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (  # noqa: E402
    fused_mtp_input_rmsnorm,
)

torch.manual_seed(42)

T = 4
HIDDEN = 128
HC_MULT = 2
EPS = 1e-6

# ---- Global shared inputs (used by BOTH implementations) ----
INPUTS_EMBEDS = torch.randn(T, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
# Include position 0 (masked to zero row) and non-zero positions.
POSITIONS = torch.tensor([0, 1, 5, 9], dtype=torch.int64, device=DEVICE)
PREV_HIDDEN = torch.randn(T, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
ENORM_WEIGHT = torch.randn(HIDDEN, dtype=torch.bfloat16, device=DEVICE)
HNORM_WEIGHT = torch.randn(HIDDEN, dtype=torch.bfloat16, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns latency stats (ms)."""
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
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps) * w.float()
    return y


def pytorch_ref(inputs_embeds, positions, prev_hidden, enorm_w, hnorm_w, eps):
    """Pure PyTorch ONLY. enorm (mask pos==0 then RMSNorm) + hnorm RMSNorm."""
    ie = inputs_embeds.cpu()
    pos = positions.cpu()
    masked = torch.where((pos == 0).unsqueeze(-1), torch.zeros_like(ie), ie)
    enorm_out = _rmsnorm(masked, enorm_w.cpu(), eps).to(inputs_embeds.dtype)
    ph = prev_hidden.cpu()
    hnorm_out = _rmsnorm(ph, hnorm_w.cpu(), eps).to(prev_hidden.dtype)
    return enorm_out, hnorm_out


def kernel_impl(inputs_embeds, positions, prev_hidden, enorm_w, hnorm_w, eps):
    """Kernel launch only."""
    return fused_mtp_input_rmsnorm(
        inputs_embeds, positions, prev_hidden, enorm_w, hnorm_w, eps, HC_MULT
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_e, ref_h = pytorch_ref(
            INPUTS_EMBEDS, POSITIONS, PREV_HIDDEN, ENORM_WEIGHT, HNORM_WEIGHT, EPS
        )
        kern_e, kern_h = kernel_impl(
            INPUTS_EMBEDS, POSITIONS, PREV_HIDDEN, ENORM_WEIGHT, HNORM_WEIGHT, EPS
        )

        re, ke = ref_e.cpu().float(), kern_e.cpu().float()
        rh, kh = ref_h.cpu().float(), kern_h.cpu().float()
        torch.testing.assert_close(ke, re, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(kh, rh, rtol=1e-2, atol=1e-2)

        de = (ke - re).abs()
        dh = (kh - rh).abs()
        stats = {
            "input_shape": tuple(INPUTS_EMBEDS.shape),
            "output_shape": tuple(kern_e.shape),
            "hnorm_output_shape": tuple(kern_h.shape),
            "in_dtype": str(INPUTS_EMBEDS.dtype),
            "out_dtype": str(kern_e.dtype),
            "device": str(INPUTS_EMBEDS.device),
            "max_abs_diff": max(de.max().item(), dh.max().item()),
            "mean_abs_diff": float(
                (de.sum() + dh.sum()).item() / (de.numel() + dh.numel())
            ),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                INPUTS_EMBEDS, POSITIONS, PREV_HIDDEN, ENORM_WEIGHT, HNORM_WEIGHT, EPS
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                INPUTS_EMBEDS, POSITIONS, PREV_HIDDEN, ENORM_WEIGHT, HNORM_WEIGHT, EPS
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
            "Kernel: _fused_mtp_input_rmsnorm_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- enorm output shape: {stats['output_shape']}\n")
            lines.append(f"- hnorm output shape: {stats['hnorm_output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
