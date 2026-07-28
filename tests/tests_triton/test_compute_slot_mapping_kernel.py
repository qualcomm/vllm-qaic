"""
Standalone QAIC validation for `_compute_slot_mapping_kernel`.

Source under test:
vllm/v1/worker/block_table.py
  - _compute_slot_mapping_kernel  (launched via BlockTable.compute_slot_mapping)

Grid is (num_reqs + 1,). For each request the kernel maps every token position
to a KV cache slot:
  block_indices = pos // (block_size * TOTAL_CP_WORLD_SIZE)
  block_numbers = block_table[req, block_indices]
  virtual_block_offset = pos - block_indices * virtual_block_size
  local offset (with CP interleave); slot = block_numbers * block_size + offset
  slot = PAD_ID where the position is not owned by this CP rank.
The final program id (num_reqs) pads slot_mapping[num_tokens:max_num_tokens]
with PAD_ID for CUDA graph compatibility.

Config tested: 3 requests, no context parallelism (TOTAL_CP_WORLD_SIZE=1,
TOTAL_CP_RANK=0, CP_KV_CACHE_INTERLEAVE_SIZE=1) so is_local is always true and
slot = block_number * block_size + pos % block_size. max_num_tokens > num_tokens
so the CUDA-graph padding path is also exercised. The full BlockTable launcher
requires distributed process groups, so we launch the @triton.jit kernel
directly. int64 index output -> EXACT (torch.equal) comparison.
Reference: pure PyTorch replication.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_slot_mapping_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/block_table.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402
from vllm.v1.worker.block_table import _compute_slot_mapping_kernel  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
BLOCK_SIZE = 8
MAX_BLOCKS_PER_REQ = 16
TOTAL_CP_WORLD_SIZE = 1
TOTAL_CP_RANK = 0
CP_KV_CACHE_INTERLEAVE_SIZE = 1
KERNEL_BLOCK = 1024

QUERY_LENS = [4, 5, 3]
_qsl = [0]
for q in QUERY_LENS:
    _qsl.append(_qsl[-1] + q)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)
NUM_TOKENS = _qsl[-1]  # 12
MAX_NUM_TOKENS = 16  # > NUM_TOKENS to exercise the padding path

# Positions per token (int64). Chosen to span multiple blocks.
POSITIONS = torch.tensor(
    [0, 1, 8, 20, 3, 9, 17, 40, 63, 2, 10, 30],
    dtype=torch.int64,
    device=DEVICE,
)
BLOCK_TABLE = torch.randint(
    0, 200, (NUM_REQS, MAX_BLOCKS_PER_REQ), dtype=torch.int32, device=DEVICE
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
    qsl = QUERY_START_LOC.cpu().tolist()
    pos = POSITIONS.cpu().tolist()
    bt = BLOCK_TABLE.cpu()

    slot_mapping = torch.full((MAX_NUM_TOKENS,), PAD_SLOT_ID, dtype=torch.int64)
    virtual_block_size = BLOCK_SIZE * TOTAL_CP_WORLD_SIZE
    for req in range(NUM_REQS):
        start = qsl[req]
        end = qsl[req + 1]
        for t in range(start, end):
            p = pos[t]
            block_index = p // virtual_block_size
            block_number = int(bt[req, block_index].item())
            vbo = p - block_index * virtual_block_size
            is_local = (
                (vbo // CP_KV_CACHE_INTERLEAVE_SIZE) % TOTAL_CP_WORLD_SIZE
            ) == TOTAL_CP_RANK
            local_off = (
                vbo // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (vbo % CP_KV_CACHE_INTERLEAVE_SIZE)
            slot = block_number * BLOCK_SIZE + local_off
            slot_mapping[t] = slot if is_local else PAD_SLOT_ID
    # Positions [num_tokens, max_num_tokens) stay PAD_ID (padding path).
    return slot_mapping


def kernel_impl():
    slot_mapping = torch.zeros(MAX_NUM_TOKENS, dtype=torch.int64, device=DEVICE)
    _compute_slot_mapping_kernel[(NUM_REQS + 1,)](
        NUM_TOKENS,
        MAX_NUM_TOKENS,
        QUERY_START_LOC,
        POSITIONS,
        BLOCK_TABLE,
        BLOCK_TABLE.stride(0),
        BLOCK_SIZE,
        slot_mapping,
        TOTAL_CP_WORLD_SIZE=TOTAL_CP_WORLD_SIZE,
        TOTAL_CP_RANK=TOTAL_CP_RANK,
        CP_KV_CACHE_INTERLEAVE_SIZE=CP_KV_CACHE_INTERLEAVE_SIZE,
        PAD_ID=PAD_SLOT_ID,
        BLOCK_SIZE=KERNEL_BLOCK,
    )
    return slot_mapping


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
            f"slot_mapping mismatch: ref={ref_cpu.tolist()} kern={kernel_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(POSITIONS.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(POSITIONS.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(POSITIONS.device),
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
            "Kernel: _compute_slot_mapping_kernel\n",
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
