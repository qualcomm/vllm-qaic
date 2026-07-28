"""
Standalone QAIC validation for `_correct_attn_cp_out_kernel`.

Source under test:
vllm/v1/attention/ops/common.py
  - _correct_attn_cp_out_kernel  (@triton.jit)
  - correct_attn_out             (public launcher)

The kernel corrects one context-parallel rank's local attention output using
the all-gathered per-rank log-sum-exp (LSE) values. For each (batch b, head h):

  lse[n] = lses[n, b, h]                      # over N ranks
  lse[n] := -inf where nan or +inf
  m       = max_n lse[n];  m := 0 if m == -inf
  final_lse = log( sum_n exp(lse[n] - m) ) + m     # base e
  factor    = exp( lses[cp_rank, b, h] - final_lse )   (nan/inf -> -inf -> 0)
  new_out[b, h, :] = out[b, h, :] * factor

and writes `final_lse` into a [B, H] buffer. This is the per-rank rescale that
precedes the cross-rank reduction in context-parallel attention.

Config tested: IS_BASE_E=True (natural-log LSE), N=4 ranks, cp_rank=1.
Reference: pure PyTorch replication of the LSE reduction + rescale.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_correct_attn_cp_out_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/common.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
N = 4  # number of CP ranks
B = 3  # batch (num tokens)
H = 2  # heads
D = 16  # head dim
CP_RANK = 1
IS_BASE_E = True

OUT = torch.randn(B, H, D, dtype=torch.float32, device=DEVICE)
# LSE values [N, B, H]; realistic magnitudes.
LSES = torch.randn(N, B, H, dtype=torch.float32, device=DEVICE) * 2.0


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


def pytorch_ref():
    """Pure PyTorch LSE reduction + per-rank output rescale."""
    out = OUT.cpu().clone()
    lses = LSES.cpu().float()  # [N, B, H]
    ninf = float("-inf")
    lse = torch.where(
        torch.isnan(lses) | (lses == float("inf")),
        torch.full_like(lses, ninf),
        lses,
    )
    lse_max, _ = lse.max(dim=0)  # [B, H]
    lse_max = torch.where(lse_max == ninf, torch.zeros_like(lse_max), lse_max)
    lse_shift = lse - lse_max.unsqueeze(0)
    lse_acc = torch.exp(lse_shift).sum(dim=0)  # [B, H]
    final_lse = torch.log(lse_acc) + lse_max  # [B, H]

    lse_tmp = lses[CP_RANK]  # [B, H]
    lse_finally = lse_tmp - final_lse
    lse_finally = torch.where(
        torch.isnan(lse_finally) | (lse_finally == float("inf")),
        torch.full_like(lse_finally, ninf),
        lse_finally,
    )
    factor = torch.exp(lse_finally)  # [B, H]
    new_out = out * factor.unsqueeze(-1)  # [B, H, D]
    return new_out, final_lse


def kernel_impl():
    """Launch only: correct_attn_out mutates its `out` in place, so clone."""
    out = OUT.clone()
    ctx = CPTritonContext()
    corrected, lse = correct_attn_out(
        out, LSES, CP_RANK, ctx, is_lse_base_on_e=IS_BASE_E
    )
    return corrected, lse


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_lse = pytorch_ref()
        ker_out, ker_lse = kernel_impl()

        ker_out_c = ker_out.cpu()
        ker_lse_c = ker_lse.cpu()
        torch.testing.assert_close(
            ker_out_c.float(), ref_out.float(), rtol=1e-3, atol=1e-3
        )
        torch.testing.assert_close(
            ker_lse_c.float(), ref_lse.float(), rtol=1e-3, atol=1e-3
        )

        diff = (ker_out_c.float() - ref_out.float()).abs()
        stats = {
            "input_shape": tuple(OUT.shape),
            "output_shape": tuple(ker_out.shape),
            "in_dtype": str(OUT.dtype),
            "out_dtype": str(ker_out.dtype),
            "device": str(OUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: _correct_attn_cp_out_kernel (IS_BASE_E=True)\n",
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
            lines.append(f"- output shape: {stats['output_shape']}\n")
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
