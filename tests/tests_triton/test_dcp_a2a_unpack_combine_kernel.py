"""
Standalone QAIC validation for `_dcp_a2a_unpack_combine_kernel`.

Source under test:
vllm/v1/attention/ops/dcp_alltoall.py
  - _dcp_a2a_unpack_combine_kernel  (@triton.jit)
  - _dcp_a2a_unpack_combine         (internal launcher — used directly here so
    no torch.distributed / All-to-All is required)

After the All-to-All exchange, each rank holds a recv buffer of shape
[N, B, H_per_rank, D + lse_pack_dim] containing the N ranks' partial outputs
and their fp32 LSEs. The unpack+combine kernel does an exact LSE-weighted
reduction over the N dimension, for each (batch b, head h):

  lse_r     = recv[r, b, h, D]                       # fp32 (lse_pack_dim=1)
  lse_r    := -inf where nan or +inf
  lse_max   = max_r lse_r;  := 0 if all -inf
  lse_sum   = sum_r exp(lse_r - lse_max)             # base e
  global_lse= log(lse_sum) + lse_max
  weight_r  = exp(lse_r - global_lse)   (nan -> 0)
  out[b,h,:]= sum_r recv[r, b, h, 0:D] * weight_r
  (optionally) out_lse[b, h] = global_lse

This is the softmax-over-ranks recombination of context-parallel partial
attention outputs. We validate the fp32 path (lse_pack_dim=1, IS_BASE_E=True,
RETURN_LSE=True). Float comparison via assert_close.
Reference: pure PyTorch replication of the LSE-weighted combine.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dcp_a2a_unpack_combine_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/dcp_alltoall.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.dcp_alltoall import _dcp_a2a_unpack_combine  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
N = 4  # world size (num ranks)
B = 3  # batch (num tokens)
H_PER_RANK = 2
D = 16  # head dim
LSE_PACK_DIM = 1  # fp32 recv dtype
IS_BASE_E = True
RETURN_LSE = True
DTYPE = torch.float32

# recv buffer: [N, B, H_per_rank, D + lse_pack_dim]; last column is the fp32 LSE.
RECV = torch.randn(N, B, H_PER_RANK, D + LSE_PACK_DIM, dtype=DTYPE, device=DEVICE)
# Give the LSE column realistic magnitudes.
RECV[..., D] = torch.randn(N, B, H_PER_RANK, device=DEVICE) * 2.0


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
    """Pure PyTorch LSE-weighted combine over the N-rank dimension."""
    recv = RECV.cpu().float()  # [N, B, H, D+1]
    out_data = recv[..., 0:D]  # [N, B, H, D]
    lse = recv[..., D]  # [N, B, H]
    ninf = float("-inf")
    lse = torch.where(
        torch.isnan(lse) | (lse == float("inf")),
        torch.full_like(lse, ninf),
        lse,
    )
    lse_max, _ = lse.max(dim=0)  # [B, H]
    lse_max = torch.where(lse_max == ninf, torch.zeros_like(lse_max), lse_max)
    lse_sum = torch.exp(lse - lse_max.unsqueeze(0)).sum(dim=0)  # [B, H]
    global_lse = torch.log(lse_sum) + lse_max  # [B, H]

    weight = torch.exp(lse - global_lse.unsqueeze(0))  # [N, B, H]
    weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)
    out = (out_data * weight.unsqueeze(-1)).sum(dim=0)  # [B, H, D]
    return out, global_lse


def kernel_impl():
    """Launch only: unpack+combine the recv buffer."""
    out, out_lse = _dcp_a2a_unpack_combine(
        RECV,
        D,
        LSE_PACK_DIM,
        RETURN_LSE,
        IS_BASE_E,
    )
    return out, out_lse


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
            "input_shape": tuple(RECV.shape),
            "output_shape": tuple(ker_out.shape),
            "in_dtype": str(RECV.dtype),
            "out_dtype": str(ker_out.dtype),
            "device": str(RECV.device),
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
            "Kernel: _dcp_a2a_unpack_combine_kernel (fp32, LSE_PACK_DIM=1)\n",
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
