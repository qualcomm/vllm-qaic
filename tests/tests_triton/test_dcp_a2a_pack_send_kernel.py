"""
Standalone QAIC validation for `_dcp_a2a_pack_send_kernel`.

Source under test:
vllm/v1/attention/ops/dcp_alltoall.py
  - _dcp_a2a_pack_send_kernel  (@triton.jit)
  - _dcp_a2a_pack_send         (internal launcher — used directly here so no
    torch.distributed / All-to-All is required)

The pack kernel gathers this rank's partial attention output [B, H, D] and its
fp32 LSE [B, H] into a per-destination-rank send buffer of shape
[N, B, H_per_rank, D + lse_pack_dim]. With an fp32 output dtype the LSE packs
into a single trailing fp32 element (lse_pack_dim = 1), so for each
(batch b, local head lh) and destination rank r:

  src_head = r * H_per_rank + lh
  send[r, b, lh, 0:D] = out[b, src_head, 0:D]
  send[r, b, lh, D]   = lse[b, src_head]

i.e. it re-lays-out the H = N*H_per_rank heads into a rank-major send payload
with the LSE appended after each head's D values. We validate the fp32 path
(lse_pack_dim=1) so no uint16 bit-splitting is exercised; the packed buffer is
compared for EXACT float equality.
Reference: pure PyTorch replication of the pack layout.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dcp_a2a_pack_send_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/dcp_alltoall.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.dcp_alltoall import _dcp_a2a_pack_send  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
N = 4  # world size (num ranks)
B = 3  # batch (num tokens)
H_PER_RANK = 2
H = N * H_PER_RANK  # total heads
D = 16  # head dim
LSE_PACK_DIM = 1  # fp32 output => LSE packs as single fp32 element
DTYPE = torch.float32

OUT = torch.randn(B, H, D, dtype=DTYPE, device=DEVICE)
LSE = torch.randn(B, H, dtype=torch.float32, device=DEVICE) * 2.0


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
    """Pure PyTorch replication of the pack layout (lse_pack_dim=1)."""
    out = OUT.cpu()
    lse = LSE.cpu()
    send = torch.zeros(N, B, H_PER_RANK, D + LSE_PACK_DIM, dtype=DTYPE)
    for r in range(N):
        for b in range(B):
            for lh in range(H_PER_RANK):
                src_head = r * H_PER_RANK + lh
                send[r, b, lh, 0:D] = out[b, src_head, 0:D]
                send[r, b, lh, D] = lse[b, src_head]
    return send


def kernel_impl():
    """Launch only: pack into a freshly-allocated send buffer."""
    send_buffer = torch.zeros(
        N, B, H_PER_RANK, D + LSE_PACK_DIM, dtype=DTYPE, device=DEVICE
    )
    _dcp_a2a_pack_send(
        OUT,
        LSE,
        send_buffer,
        N,
        H_PER_RANK,
        D,
        LSE_PACK_DIM,
    )
    return send_buffer


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl()

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(
            kernel_cpu.float(), ref_cpu.float(), rtol=0, atol=0
        )

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        stats = {
            "input_shape": tuple(OUT.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(OUT.dtype),
            "out_dtype": str(kernel_out.dtype),
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
            "Kernel: _dcp_a2a_pack_send_kernel (fp32, LSE_PACK_DIM=1)\n",
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
