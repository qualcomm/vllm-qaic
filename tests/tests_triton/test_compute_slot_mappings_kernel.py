"""
Standalone QAIC validation for `_compute_slot_mappings_kernel`.

Source under test:
vllm/v1/worker/gpu/block_table.py
  - _compute_slot_mappings_kernel  (for each kv-cache group and batch row,
    converts token positions into flat KV-cache slot ids using the request's
    block table: slot = block_number * block_size + block_offset. A trailing
    program pads the remaining slots [actual_num_tokens : max_num_tokens] to
    PAD_ID. This test covers the common case CP_SIZE == 1.)

The launcher is a private class method (BlockTables.compute_slot_mappings),
which launches:

    _compute_slot_mappings_kernel[(num_groups, num_reqs + 1)](
        self.max_num_batched_tokens,
        idx_mapping,
        query_start_loc,
        positions,
        self.block_table_ptrs,      # uint64 data_ptr() of each block table
        self.block_table_strides,   # stride(0) per group
        self.block_sizes_tensor,    # kernel block size per group
        self.slot_mappings,         # [num_groups, max_num_tokens]
        self.slot_mappings.stride(0),
        self.cp_rank,
        CP_SIZE=1,
        CP_INTERLEAVE=1,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=1024,
    )

We replicate that launch here. Reference: pure PyTorch slot computation using
the block tables and positions, padding the tail to PAD_ID. Output is integer,
so we compare with EXACT equality (no float tolerance).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_slot_mappings_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/block_table.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # noqa: E402
from vllm.v1.worker.gpu.block_table import (  # noqa: E402
    _compute_slot_mappings_kernel,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
TRITON_BLOCK_SIZE = 1024
CP_SIZE = 1
CP_INTERLEAVE = 1
CP_RANK = 0

NUM_GROUPS = 2
MAX_NUM_REQS = 4
MAX_NUM_BLOCKS = 8
BLOCK_SIZES = [16, 32]          # kernel block size per group
MAX_NUM_BATCHED_TOKENS = 32     # cols in slot_mappings

# Two requests with 3 and 5 query tokens -> 8 tokens total.
QUERY_LENS = [3, 5]
NUM_REQS = len(QUERY_LENS)
NUM_TOKENS = sum(QUERY_LENS)

torch.manual_seed(42)

# query_start_loc (indptr): [0, 3, 8]
QUERY_START_LOC = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(QUERY_LENS), dim=0).tolist()),
    dtype=torch.int32,
)

# Token positions within each request's sequence (prefill from 0).
POSITIONS = torch.tensor(
    [0, 1, 2] + [0, 1, 2, 3, 4], dtype=torch.int64
)

# Batch row -> request-state index in the persistent block tables.
IDX_MAPPING = torch.tensor([1, 0], dtype=torch.int32)

# Persistent block tables, one per group: [MAX_NUM_REQS, MAX_NUM_BLOCKS].
BLOCK_TABLES = [
    (torch.arange(
        MAX_NUM_REQS * MAX_NUM_BLOCKS, dtype=torch.int32
    ).reshape(MAX_NUM_REQS, MAX_NUM_BLOCKS) + g * 50)
    for g in range(NUM_GROUPS)
]
BLOCK_TABLES = list(BLOCK_TABLES)


def pytorch_ref(block_tables, num_blocks_unused=None):
    """Pure PyTorch slot-id computation with tail padded to PAD_ID."""
    qsl = QUERY_START_LOC.cpu().tolist()
    pos = POSITIONS.cpu()
    idx_mapping = IDX_MAPPING.cpu().tolist()
    out = torch.full(
        (NUM_GROUPS, MAX_NUM_BATCHED_TOKENS), PAD_SLOT_ID, dtype=torch.int64
    )
    for g in range(NUM_GROUPS):
        bt = block_tables[g].cpu()
        bs = BLOCK_SIZES[g]
        for b in range(NUM_REQS):
            req = idx_mapping[b]
            start, end = qsl[b], qsl[b + 1]
            p = pos[start:end]
            block_indices = p // bs
            block_offsets = p % bs
            block_numbers = bt[req, block_indices].to(torch.int64)
            out[g, start:end] = block_numbers * bs + block_offsets
    return out


def kernel_impl(block_tables, num_blocks_unused=None):
    """Kernel wrapper: launch only (replicates the class-method launch site)."""
    bt_gpu = [t.to(DEVICE) for t in block_tables]
    idx_mapping = IDX_MAPPING.to(DEVICE)
    qsl = QUERY_START_LOC.to(DEVICE)
    pos = POSITIONS.to(DEVICE)

    bt_ptrs = torch.tensor(
        [t.data_ptr() for t in bt_gpu], dtype=torch.uint64, device=DEVICE
    )
    bt_strides = torch.tensor(
        [t.stride(0) for t in bt_gpu], dtype=torch.int64, device=DEVICE
    )
    block_sizes = torch.tensor(BLOCK_SIZES, dtype=torch.int32, device=DEVICE)

    slot_mappings = torch.zeros(
        NUM_GROUPS, MAX_NUM_BATCHED_TOKENS, dtype=torch.int64, device=DEVICE
    )

    _compute_slot_mappings_kernel[(NUM_GROUPS, NUM_REQS + 1)](
        MAX_NUM_BATCHED_TOKENS,
        idx_mapping,
        qsl,
        pos,
        bt_ptrs,
        bt_strides,
        block_sizes,
        slot_mappings,
        slot_mappings.stride(0),
        CP_RANK,
        CP_SIZE=CP_SIZE,
        CP_INTERLEAVE=CP_INTERLEAVE,
        PAD_ID=PAD_SLOT_ID,
        TRITON_BLOCK_SIZE=TRITON_BLOCK_SIZE,
    )
    return slot_mappings


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(BLOCK_TABLES)
        kernel_out = kernel_impl(BLOCK_TABLES)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        exact_match = bool(torch.equal(kernel_cpu, ref_cpu))
        num_mismatch = int((kernel_cpu != ref_cpu).sum().item())
        assert exact_match, (
            f"exact int mismatch: num_mismatch={num_mismatch}\n"
            f"ref={ref_cpu.tolist()}\nkernel={kernel_cpu.tolist()}"
        )

        stats = {
            "num_groups": NUM_GROUPS,
            "block_table_shape": tuple(BLOCK_TABLES[0].shape),
            "block_sizes": BLOCK_SIZES,
            "num_reqs": NUM_REQS,
            "num_tokens": NUM_TOKENS,
            "query_start_loc": QUERY_START_LOC.tolist(),
            "idx_mapping": IDX_MAPPING.tolist(),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(kernel_out.dtype),
            "device": str(kernel_out.device),
            "exact_match": exact_match,
            "num_mismatch": num_mismatch,
            "max_abs_diff": 0,      # exact integer match
            "mean_abs_diff": 0.0,
        }

        pt_stats = _bench(lambda: pytorch_ref(BLOCK_TABLES))
        kern_stats = _bench(lambda: kernel_impl(BLOCK_TABLES))
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
            "Kernel: _compute_slot_mappings_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- num_groups: {stats['num_groups']}\n")
            lines.append(f"- block_table shape: {stats['block_table_shape']}\n")
            lines.append(f"- block_sizes: {stats['block_sizes']}\n")
            lines.append(f"- num_reqs / num_tokens: {stats['num_reqs']} / {stats['num_tokens']}\n")
            lines.append(f"- query_start_loc: {stats['query_start_loc']}\n")
            lines.append(f"- idx_mapping: {stats['idx_mapping']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Comparison: EXACT integer equality (incl. PAD_ID tail)\n")
            lines.append("Output:\n")
            lines.append(f"- slot_mappings shape: {stats['output_shape']}\n")
            lines.append(f"- exact_match: {stats['exact_match']}\n")
            lines.append(f"- num_mismatch: {stats['num_mismatch']}\n")
            lines.append(f"- max_abs_diff (int): {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff (int): {stats['mean_abs_diff']}\n")
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
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
