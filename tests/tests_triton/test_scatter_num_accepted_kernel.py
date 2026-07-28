"""
Standalone QAIC validation for `_scatter_num_accepted_kernel`.

Source under test:
vllm/v1/worker/gpu/model_states/mamba_hybrid.py
  - _scatter_num_accepted_kernel  (launched directly from
    MambaHybridModelState.postprocess_state)

Per row (program_id 0):
    req_state_idx = idx_mapping[row]
    if req_state_idx < 0: skip (PP -1 sentinel)
    num_accepted[req_state_idx] = max(num_sampled[row], 1)

The kernel is launched directly with grid=(num_reqs,); no separate Python
launcher exists, so `kernel_impl` replicates the launch line from
postprocess_state.

Config tested: 5 rows including a -1 sentinel (skipped) and a zero
num_sampled (clamped to 1), scattering into a size-8 num_accepted buffer
pre-filled with 1s. Integer output -> EXACT integer equality.
Reference: pure-PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_scatter_num_accepted_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.model_states.mamba_hybrid import (  # noqa: E402
    _scatter_num_accepted_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
MAX_NUM_REQS = 8
NUM_REQS = 5
# row -> req_state_idx; row 2 is a -1 sentinel (skipped).
IDX_MAPPING = torch.tensor([0, 3, -1, 5, 2], dtype=torch.int32, device=DEVICE)
# num_sampled per row; row 4 has 0 -> clamped to 1.
NUM_SAMPLED = torch.tensor([2, 4, 9, 0, 0], dtype=torch.int32, device=DEVICE)
NUM_ACCEPTED_BASE = torch.ones(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)


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
    """Pure PyTorch replication of the scatter."""
    num_accepted = NUM_ACCEPTED_BASE.clone().cpu()
    idx_mapping = IDX_MAPPING.cpu()
    num_sampled = NUM_SAMPLED.cpu()
    for row in range(NUM_REQS):
        rsi = int(idx_mapping[row])
        if rsi < 0:
            continue
        num_accepted[rsi] = max(int(num_sampled[row]), 1)
    return num_accepted


def kernel_impl():
    """Kernel launch only (mirrors postprocess_state launch)."""
    num_accepted = NUM_ACCEPTED_BASE.clone()
    _scatter_num_accepted_kernel[(NUM_REQS,)](
        IDX_MAPPING, NUM_SAMPLED, num_accepted
    )
    return num_accepted


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
            f"ref={ref_cpu.tolist()} k={kernel_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(IDX_MAPPING.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(IDX_MAPPING.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(IDX_MAPPING.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "num_accepted": kernel_cpu.tolist(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup

        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _scatter_num_accepted_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape (idx_mapping): {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- num_accepted: {stats['num_accepted']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact-match comparison)\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
                    "Timing:\n",
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
                ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
