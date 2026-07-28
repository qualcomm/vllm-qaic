"""
Standalone QAIC validation for the `apply_expert_map` @triton.jit device helper.

Source under test:
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
  - apply_expert_map(expert_id, expert_map)

`apply_expert_map` is a device-side helper (called from inside the ep-scatter /
ep-gather kernels) that remaps a *global* expert id to its *local* (EP-shard)
id via a lookup table `expert_map`. A sentinel value of -1 (invalid expert) is
passed through unchanged.

Exact source logic:
    if expert_id != -1:
        expert_id = tl.load(expert_map + expert_id)
    return expert_id

Because the helper uses scalar control-flow (`if expert_id != -1`), we launch a
minimal `@triton.jit` wrapper (`_apply_expert_map_launcher`) with one program
per element (grid = (N,)), each program remapping a single global id.

Reference: pure PyTorch gather  out[i] = expert_map[gid[i]] if gid[i] != -1
else -1.  Integer output -> exact-match comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import apply_expert_map
from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_apply_expert_map.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py"

DEVICE = "qaic"
NUM_EXPERTS = 4  # local (EP-shard) experts
N = 8  # number of global expert ids to remap

torch.manual_seed(42)
# A random EP expert_map: global-id -> local-id lookup table.
EXPERT_MAP = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)
# Global expert ids to remap; include -1 sentinels (invalid) that pass through.
GLOBAL_IDS = torch.tensor(
    [0, 3, -1, 1, 2, -1, 3, 0], dtype=torch.int32, device=DEVICE
)


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


def pytorch_ref(global_ids, expert_map):
    """Pure PyTorch expert-id remap: gather expert_map[gid], keep -1 as -1."""
    global_ids = global_ids.cpu()
    expert_map = expert_map.cpu()
    out = torch.empty_like(global_ids)
    for i in range(global_ids.numel()):
        gid = int(global_ids[i].item())
        out[i] = gid if gid == -1 else int(expert_map[gid].item())
    return out


@triton.jit
def _apply_expert_map_launcher(gid_ptr, expert_map, out_ptr):
    pid = tl.program_id(0)
    gid = tl.load(gid_ptr + pid)
    res = apply_expert_map(gid, expert_map)
    tl.store(out_ptr + pid, res)


def kernel_impl(global_ids, expert_map):
    out = torch.empty_like(global_ids)
    _apply_expert_map_launcher[(global_ids.numel(),)](global_ids, expert_map, out)
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(GLOBAL_IDS, EXPERT_MAP)
        kernel_out = kernel_impl(GLOBAL_IDS, EXPERT_MAP)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        exact = bool(torch.equal(kernel_cpu, ref_cpu))
        assert exact, "integer remap mismatch"
        diff = (kernel_cpu.to(torch.int64) - ref_cpu.to(torch.int64)).abs()
        stats = {
            "input_shape": tuple(GLOBAL_IDS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(GLOBAL_IDS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(GLOBAL_IDS.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.to(torch.float32).mean().item()),
            "exact_match": exact,
        }

        pt_stats = _bench(lambda: pytorch_ref(GLOBAL_IDS, EXPERT_MAP))
        kern_stats = _bench(lambda: kernel_impl(GLOBAL_IDS, EXPERT_MAP))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: apply_expert_map (device helper)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- global_ids shape: {stats['input_shape']}\n",
                f"- expert_map: {EXPERT_MAP.cpu().tolist()}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- out shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- exact_match: {stats['exact_match']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
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
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
