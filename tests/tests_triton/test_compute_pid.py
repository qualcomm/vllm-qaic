"""
Standalone QAIC validation for `_compute_pid` (device helper).

Source under test:
vllm/model_executor/layers/batch_invariant.py
  - _compute_pid  (grouped-launch tile-id -> (pid_m, pid_n) mapping)

`_compute_pid` is a `@triton.jit` DEVICE HELPER (not launchable on its own),
so we wrap it in a tiny standalone `@triton.jit` kernel that evaluates it for
each linear tile id and stores the resulting (pid_m, pid_n) coordinates. These
coordinates re-order tiles for L2-locality in the persistent matmul.

Reference: pure-PyTorch replication of the group_size_m grouping arithmetic.
Integer exact comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.batch_invariant import _compute_pid

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_pid.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/batch_invariant.py"

DEVICE = "qaic"
torch.manual_seed(42)

# Grid geometry mirroring a small persistent matmul launch.
NUM_PID_M = 4
NUM_PID_N = 4
GROUP_SIZE_M = 2
NUM_PID_IN_GROUP = GROUP_SIZE_M * NUM_PID_N
NUM_TILES = NUM_PID_M * NUM_PID_N

TILE_IDS = torch.arange(NUM_TILES, dtype=torch.int32, device=DEVICE)


@triton.jit
def _compute_pid_wrapper(
    pid_m_ptr,
    pid_n_ptr,
    num_pid_in_group,
    num_pid_m,
    GROUP_SIZE_M: tl.constexpr,
):
    tid = tl.program_id(0)
    pid_m, pid_n = _compute_pid(tid, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
    tl.store(pid_m_ptr + tid, pid_m)
    tl.store(pid_n_ptr + tid, pid_n)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(tile_ids):
    """Pure PyTorch replication of _compute_pid grouped index arithmetic."""
    tile_ids = tile_ids.cpu().to(torch.int64)
    pid_m = torch.empty_like(tile_ids)
    pid_n = torch.empty_like(tile_ids)
    for i in range(tile_ids.numel()):
        tile_id = int(tile_ids[i].item())
        group_id = tile_id // NUM_PID_IN_GROUP
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(NUM_PID_M - first_pid_m, GROUP_SIZE_M)
        pid_m[i] = first_pid_m + (tile_id % group_size_m)
        pid_n[i] = (tile_id % NUM_PID_IN_GROUP) // group_size_m
    return pid_m, pid_n


def kernel_impl(tile_ids):
    pid_m = torch.empty(NUM_TILES, dtype=torch.int32, device=tile_ids.device)
    pid_n = torch.empty(NUM_TILES, dtype=torch.int32, device=tile_ids.device)
    _compute_pid_wrapper[(NUM_TILES,)](
        pid_m,
        pid_n,
        NUM_PID_IN_GROUP,
        NUM_PID_M,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    return pid_m, pid_n


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_m, ref_n = pytorch_ref(TILE_IDS)
        k_m, k_n = kernel_impl(TILE_IDS)

        k_m = k_m.cpu().to(torch.int64)
        k_n = k_n.cpu().to(torch.int64)

        assert torch.equal(k_m, ref_m), "pid_m mismatch"
        assert torch.equal(k_n, ref_n), "pid_n mismatch"

        stats = {
            "input_shape": tuple(TILE_IDS.shape),
            "output_shape": tuple(k_m.shape),
            "in_dtype": str(TILE_IDS.dtype),
            "out_dtype": str(k_m.dtype),
            "device": str(TILE_IDS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
        }
        pt_stats = _bench(lambda: pytorch_ref(TILE_IDS))
        kern_stats = _bench(lambda: kernel_impl(TILE_IDS))
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
            "Kernel: _compute_pid (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"Input shape: {stats['input_shape']} ({stats['in_dtype']})\n")
            lines.append(f"Output shape: {stats['output_shape']} ({stats['out_dtype']})\n")
            lines.append(f"Device: {stats['device']}\n")
            lines.append(f"Max abs diff: {stats['max_abs_diff']}\n")
            lines.append(f"Mean abs diff: {stats['mean_abs_diff']}\n")
            lines.append("Rel error: exact-integer compare\n")
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


if __name__ == "__main__":
    main()
