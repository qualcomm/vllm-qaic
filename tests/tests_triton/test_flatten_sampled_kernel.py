"""
Standalone QAIC validation for `_flatten_sampled_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
  - _flatten_sampled_kernel  (flattens the per-request [num_reqs, spec+1]
    `sampled` matrix into a packed [num_logits] vector).

For each request `req_idx` it copies `sampled[req_idx, 0:num_sampled[req_idx]]`
into `flat_sampled[start:start+num_sampled]` where `start = cu_num_logits[req_idx]`.
Purely a deterministic integer gather/scatter with NO RNG, so we compare the
launched Triton kernel output against a pure-PyTorch reference EXACTLY.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_flatten_sampled_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (  # noqa: E402
    _flatten_sampled_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 4
NUM_SPEC_STEPS = 3
# Per-request logits budget (cu_num_logits deltas).
NUM_LOGITS_PER_REQ = [4, 2, 3, 4]
# Number of sampled tokens per request (<= its logits budget, <= spec+1).
NUM_SAMPLED_PER_REQ = [3, 2, 1, 4]

_cu = [0]
for _n in NUM_LOGITS_PER_REQ:
    _cu.append(_cu[-1] + _n)
CU_NUM_LOGITS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
NUM_LOGITS = _cu[-1]

NUM_SAMPLED = torch.tensor(NUM_SAMPLED_PER_REQ, dtype=torch.int32, device=DEVICE)
VOCAB = 128
SAMPLED = torch.randint(
    0, VOCAB, (NUM_REQS, NUM_SPEC_STEPS + 1), dtype=torch.int64, device=DEVICE
)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
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


def pytorch_ref(sampled, num_sampled, cu_num_logits):
    sampled = sampled.cpu()
    num_sampled = num_sampled.cpu()
    cu = cu_num_logits.cpu()
    flat = torch.zeros(NUM_LOGITS, dtype=sampled.dtype)
    for r in range(NUM_REQS):
        start = int(cu[r].item())
        ns = int(num_sampled[r].item())
        for i in range(ns):
            flat[start + i] = sampled[r, i]
    return flat


def kernel_impl(sampled, num_sampled, cu_num_logits):
    flat = torch.zeros(NUM_LOGITS, dtype=sampled.dtype, device=sampled.device)
    _flatten_sampled_kernel[(NUM_REQS,)](
        flat,
        sampled,
        sampled.stride(0),
        num_sampled,
        cu_num_logits,
        num_warps=1,
    )
    return flat


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(SAMPLED, NUM_SAMPLED, CU_NUM_LOGITS)
        kernel_out = kernel_impl(SAMPLED, NUM_SAMPLED, CU_NUM_LOGITS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), "flat_sampled mismatch"

        diff = (kernel_cpu.float() - ref_cpu.float()).abs()
        stats = {
            "input_shape": tuple(SAMPLED.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(SAMPLED.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(SAMPLED.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(SAMPLED, NUM_SAMPLED, CU_NUM_LOGITS))
        kern_stats = _bench(lambda: kernel_impl(SAMPLED, NUM_SAMPLED, CU_NUM_LOGITS))
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
            "Kernel: _flatten_sampled_kernel\n",
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
