"""
Standalone QAIC validation for `_gather_block_tables_kernel`.

Source under test:
vllm/v1/worker/gpu/block_table.py
  - _gather_block_tables_kernel  (for each kv-cache group and batch row,
    gathers a request's block-table row from a persistent source block table
    into the destination (input) block table, using batch_idx -> req_idx
    mapping. Padded rows [batch_idx >= num_reqs] are zeroed. Pure integer
    index-gather/copy.)

The launcher is a private class method (BlockTables.gather_block_tables),
which launches:

    _gather_block_tables_kernel[(num_kv_cache_groups, num_reqs_padded)](
        idx_mapping,
        self.block_table_ptrs,          # uint64 data_ptr() of each src table
        self.input_block_table_ptrs,    # uint64 data_ptr() of each dst table
        self.block_table_strides,       # stride(0) == max_num_blocks per group
        self.num_blocks.gpu,
        self.num_blocks.gpu.stride(0),
        num_reqs,
        BLOCK_SIZE=1024,
    )

We replicate that launch here. For batch row b < num_reqs, req = idx_mapping[b]
and dst[b, 0:nb] = src[req, 0:nb] where nb = num_blocks[group, req]; the tail of
the dst row is left as-is. For b >= num_reqs the whole dst row is zeroed.

Reference: pure PyTorch gather/copy replicating that logic. Output is integer,
so we compare with EXACT equality (no float tolerance).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_gather_block_tables_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/block_table.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.block_table import (  # noqa: E402
    _gather_block_tables_kernel,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
BLOCK_SIZE = 1024

NUM_GROUPS = 2
MAX_NUM_REQS = 4          # rows in the persistent source/dst tables
MAX_NUM_BLOCKS = 6        # cols == stride(0) per group
NUM_REQS = 3              # actual requests this step
NUM_REQS_PADDED = 4       # batch rows launched (>= NUM_REQS -> zero the tail)

torch.manual_seed(42)

# Source (persistent) block tables, one per group: [MAX_NUM_REQS, MAX_NUM_BLOCKS].
SRC_TABLES = [
    torch.arange(
        g * 100, g * 100 + MAX_NUM_REQS * MAX_NUM_BLOCKS, dtype=torch.int32
    ).reshape(MAX_NUM_REQS, MAX_NUM_BLOCKS)
    for g in range(NUM_GROUPS)
]

# num_blocks[group, req]: how many valid blocks each request holds.
NUM_BLOCKS = torch.tensor(
    [
        [2, 4, 6, 0],   # group 0
        [3, 1, 5, 0],   # group 1
    ],
    dtype=torch.int32,
)

# Batch row b -> request index in the persistent tables.
IDX_MAPPING = torch.tensor([2, 0, 1], dtype=torch.int32)


def pytorch_ref(src_tables, num_blocks, idx_mapping):
    """Pure PyTorch gather/copy replicating the kernel semantics."""
    num_blocks = num_blocks.cpu()
    idx_mapping = idx_mapping.cpu().tolist()
    dst_tables = []
    for g in range(NUM_GROUPS):
        src = src_tables[g].cpu()
        # Destination is initialised to the same "stale" contents as the kernel
        # test uses (all zeros here), matching input_block_tables being zeros.
        dst = torch.zeros_like(src)
        for b in range(NUM_REQS_PADDED):
            if b >= NUM_REQS:
                dst[b, :] = 0
                continue
            req = idx_mapping[b]
            nb = int(num_blocks[g, req].item())
            dst[b, :nb] = src[req, :nb]
        dst_tables.append(dst)
    return dst_tables


def kernel_impl(src_tables, num_blocks, idx_mapping):
    """Kernel wrapper: launch only (replicates the class-method launch site)."""
    src_gpu = [t.to(DEVICE) for t in src_tables]
    dst_gpu = [torch.zeros_like(t) for t in src_gpu]
    num_blocks_gpu = num_blocks.to(DEVICE)
    idx_mapping_gpu = idx_mapping.to(DEVICE)

    src_ptrs = torch.tensor(
        [t.data_ptr() for t in src_gpu], dtype=torch.uint64, device=DEVICE
    )
    dst_ptrs = torch.tensor(
        [t.data_ptr() for t in dst_gpu], dtype=torch.uint64, device=DEVICE
    )
    strides = torch.tensor(
        [t.stride(0) for t in src_gpu], dtype=torch.int64, device=DEVICE
    )

    _gather_block_tables_kernel[(NUM_GROUPS, NUM_REQS_PADDED)](
        idx_mapping_gpu,
        src_ptrs,
        dst_ptrs,
        strides,
        num_blocks_gpu,
        num_blocks_gpu.stride(0),
        NUM_REQS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dst_gpu


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
        ref_out = pytorch_ref(SRC_TABLES, NUM_BLOCKS, IDX_MAPPING)
        kernel_out = kernel_impl(SRC_TABLES, NUM_BLOCKS, IDX_MAPPING)

        exact_match = True
        num_mismatch = 0
        for g in range(NUM_GROUPS):
            ref_cpu = ref_out[g].cpu()
            kern_cpu = kernel_out[g].cpu()
            exact_match = exact_match and bool(torch.equal(kern_cpu, ref_cpu))
            num_mismatch += int((kern_cpu != ref_cpu).sum().item())

        assert exact_match, (
            f"exact int mismatch: num_mismatch={num_mismatch} "
            f"ref={[t.cpu().tolist() for t in ref_out]} "
            f"kernel={[t.cpu().tolist() for t in kernel_out]}"
        )

        stats = {
            "num_groups": NUM_GROUPS,
            "src_table_shape": tuple(SRC_TABLES[0].shape),
            "num_reqs": NUM_REQS,
            "num_reqs_padded": NUM_REQS_PADDED,
            "idx_mapping": IDX_MAPPING.tolist(),
            "output_shape": tuple(kernel_out[0].shape),
            "dtype": str(kernel_out[0].dtype),
            "device": str(kernel_out[0].device),
            "exact_match": exact_match,
            "num_mismatch": num_mismatch,
            "max_abs_diff": 0,      # exact integer match
            "mean_abs_diff": 0.0,
        }

        pt_stats = _bench(lambda: pytorch_ref(SRC_TABLES, NUM_BLOCKS, IDX_MAPPING))
        kern_stats = _bench(lambda: kernel_impl(SRC_TABLES, NUM_BLOCKS, IDX_MAPPING))
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
            "Kernel: _gather_block_tables_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- num_groups: {stats['num_groups']}\n")
            lines.append(f"- src_table shape: {stats['src_table_shape']}\n")
            lines.append(f"- num_reqs / padded: {stats['num_reqs']} / {stats['num_reqs_padded']}\n")
            lines.append(f"- idx_mapping: {stats['idx_mapping']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Comparison: EXACT integer equality\n")
            lines.append("Output:\n")
            lines.append(f"- dst_block_table shape: {stats['output_shape']}\n")
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
