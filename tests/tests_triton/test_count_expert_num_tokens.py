"""
Standalone QAIC validation for `_count_expert_num_tokens`.

Source under test:
vllm/model_executor/layers/fused_moe/utils.py
  - _count_expert_num_tokens  (per-expert token counting for MoE routing)

For each local expert i, counts how many entries in `topk_ids` route to it.
When an `expert_map` is provided (expert-parallel), each global expert id is
first remapped to its local id (or -1 if not on this rank) before counting.
Launched via the file's `count_expert_num_tokens` wrapper.

Reference: pure PyTorch bincount (with optional expert-map remap).
Integer exact comparison.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.utils import count_expert_num_tokens

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_count_expert_num_tokens.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/utils.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 8
TOP_K = 2
NUM_EXPERTS = 4  # global number of experts
NUM_LOCAL_EXPERTS = 4  # experts on this rank

# Each token's top-k expert assignments (signed; -1 would be "invalid").
TOPK_IDS = torch.randint(
    0, NUM_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int32, device=DEVICE
)
# No expert-parallel remap in this validation.
EXPERT_MAP = None


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(topk_ids, num_local_experts, expert_map):
    """Pure PyTorch per-expert token count via bincount."""
    ids = topk_ids.cpu().reshape(-1).to(torch.int64)
    if expert_map is not None:
        emap = expert_map.cpu().to(torch.int64)
        valid = ids >= 0
        remapped = torch.full_like(ids, -1)
        remapped[valid] = emap[ids[valid]]
        ids = remapped
    counts = torch.zeros(num_local_experts, dtype=torch.int32)
    for e in range(num_local_experts):
        counts[e] = int((ids == e).sum().item())
    return counts


def kernel_impl(topk_ids, num_local_experts, expert_map):
    return count_expert_num_tokens(topk_ids, num_local_experts, expert_map)


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
        ref_out = pytorch_ref(TOPK_IDS, NUM_LOCAL_EXPERTS, EXPERT_MAP)
        kernel_out = kernel_impl(TOPK_IDS, NUM_LOCAL_EXPERTS, EXPERT_MAP)

        kernel_cpu = kernel_out.cpu().to(torch.int32)

        assert torch.equal(kernel_cpu, ref_out), "expert token count mismatch"

        stats = {
            "input_shape": tuple(TOPK_IDS.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(TOPK_IDS.dtype),
            "device": str(TOPK_IDS.device),
            "counts": kernel_cpu.tolist(),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
        }
        pt_stats = _bench(lambda: pytorch_ref(TOPK_IDS, NUM_LOCAL_EXPERTS, EXPERT_MAP))
        kern_stats = _bench(lambda: kernel_impl(TOPK_IDS, NUM_LOCAL_EXPERTS, EXPERT_MAP))
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
            "Kernel: _count_expert_num_tokens\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"Input shape (topk_ids): {stats['input_shape']}\n")
            lines.append(f"Output shape: {stats['output_shape']}\n")
            lines.append(f"Dtype: {stats['dtype']}\n")
            lines.append(f"Device: {stats['device']}\n")
            lines.append(f"Per-expert counts: {stats['counts']}\n")
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
