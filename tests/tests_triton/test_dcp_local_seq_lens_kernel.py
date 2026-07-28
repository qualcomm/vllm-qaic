"""
Standalone QAIC validation for `_dcp_local_seq_lens_kernel`.

Source under test:
vllm/v1/worker/gpu/cp_utils.py
  - _dcp_local_seq_lens_kernel  (launched via `prepare_dcp_local_seq_lens`)

Computes the per-request local sequence length for a decode-context-parallel
(DCP) rank, distributing KV cache round-robin across ranks:
  rounds        = seq_len // (dcp_size * cp_interleave)
  remainder     = seq_len %  (dcp_size * cp_interleave)
  remainder     = clamp(remainder - dcp_rank * cp_interleave, 0, cp_interleave)
  local_seq_len = rounds * cp_interleave + remainder
Positions in [num_reqs, max_num_reqs) are padded with 0.

Config tested: max_num_reqs=8, num_reqs=5, dcp_size=4, dcp_rank=1,
cp_interleave=2, single block (BLOCK_SIZE=128). The launcher
`prepare_dcp_local_seq_lens` early-returns when dcp_size==1, so we launch the
@triton.jit kernel directly to exercise the distribution math. Integer output
-> EXACT (torch.equal) comparison. Reference: pure PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_dcp_local_seq_lens_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/cp_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.cp_utils import _dcp_local_seq_lens_kernel  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
MAX_NUM_REQS = 8
NUM_REQS = 5
DCP_SIZE = 4
DCP_RANK = 1
CP_INTERLEAVE = 2
BLOCK_SIZE = 128  # matches launcher default; one block covers MAX_NUM_REQS

# seq_lens for real requests; entries beyond NUM_REQS are ignored / padded to 0.
SEQ_LENS = torch.tensor(
    [10, 17, 33, 8, 100, 0, 0, 0], dtype=torch.int32, device=DEVICE
)


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
    seq_lens = SEQ_LENS.cpu().tolist()
    out = torch.zeros(MAX_NUM_REQS, dtype=SEQ_LENS.dtype)
    denom = DCP_SIZE * CP_INTERLEAVE
    for i in range(MAX_NUM_REQS):
        if i >= NUM_REQS:
            out[i] = 0
            continue
        sl = seq_lens[i]
        rounds = sl // denom
        remainder = sl % denom
        remainder = max(remainder - DCP_RANK * CP_INTERLEAVE, 0)
        remainder = min(remainder, CP_INTERLEAVE)
        out[i] = rounds * CP_INTERLEAVE + remainder
    return out


def kernel_impl():
    out = torch.zeros(MAX_NUM_REQS, dtype=SEQ_LENS.dtype, device=DEVICE)
    num_blocks = (MAX_NUM_REQS + BLOCK_SIZE - 1) // BLOCK_SIZE
    _dcp_local_seq_lens_kernel[(num_blocks,)](
        out,
        SEQ_LENS,
        DCP_SIZE,
        DCP_RANK,
        CP_INTERLEAVE,
        NUM_REQS,
        MAX_NUM_REQS,
        BLOCK_SIZE,
    )
    return out


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
        assert torch.equal(kernel_cpu, ref_cpu), (
            f"local_seq_lens mismatch: ref={ref_cpu.tolist()} "
            f"kern={kernel_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(SEQ_LENS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(SEQ_LENS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(SEQ_LENS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0,
        }
        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: _dcp_local_seq_lens_kernel\n",
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
            lines.append("- relative_error: 0.0 (exact integer match)\n")
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
