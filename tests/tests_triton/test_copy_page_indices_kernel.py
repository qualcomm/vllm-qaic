"""
Standalone QAIC validation for `_copy_page_indices_kernel`.

Source under test:
vllm/v1/attention/backends/flashinfer.py
  - _copy_page_indices_kernel  (flattens each request's block-table page
    indices into a single indptr-addressed `page_indices` buffer for
    FlashInfer paged attention. Pure integer index-gather/copy.)

The launcher is a private class method
(FlashInferMetadataBuilder._compute_flashinfer_kv_metadata), which launches:

    _copy_page_indices_kernel[(num_reqs,)](
        paged_kv_indices,          # flat output [num_actual_pages]
        block_table_tensor,        # [num_reqs, max_blocks_per_req]
        block_table_tensor.stride(0),
        paged_kv_indptr,           # cumulative num_blocks [num_reqs + 1]
        BLOCK_SIZE=1024,
    )

We replicate that launch here. For request r, the kernel copies
block_table[r, 0:num_blocks_r] into page_indices[indptr[r]:indptr[r+1]], where
num_blocks_r = indptr[r+1] - indptr[r].

Reference: pure PyTorch flattened index copy using the indptr offsets. Output
is integer, so we compare with EXACT equality (no float tolerance).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_copy_page_indices_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/flashinfer.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    _copy_page_indices_kernel,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
BLOCK_SIZE = 1024
# Per-request block counts: 3 requests with 2, 3, 1 pages respectively.
NUM_BLOCKS_PER_REQ = [2, 3, 1]
NUM_REQS = len(NUM_BLOCKS_PER_REQ)
MAX_BLOCKS_PER_REQ = max(NUM_BLOCKS_PER_REQ)
NUM_ACTUAL_PAGES = sum(NUM_BLOCKS_PER_REQ)

torch.manual_seed(42)

# Block table [num_reqs, max_blocks_per_req]; unused tail entries are junk.
BLOCK_TABLE = torch.tensor(
    [
        [10, 11, 99],   # req 0 uses first 2
        [20, 21, 22],   # req 1 uses first 3
        [30, 88, 77],   # req 2 uses first 1
    ],
    dtype=torch.int32,
    device=DEVICE,
)

# cu_num_blocks (indptr): [0, 2, 5, 6]
CU_NUM_BLOCKS = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(NUM_BLOCKS_PER_REQ), dim=0).tolist()),
    dtype=torch.int32,
    device=DEVICE,
)


def pytorch_ref(block_table, cu_num_blocks):
    """Pure PyTorch flattened index copy using indptr offsets."""
    block_table = block_table.cpu()
    cu = cu_num_blocks.cpu()
    num_reqs = block_table.shape[0]
    total = int(cu[num_reqs].item())
    out = torch.zeros(total, dtype=block_table.dtype)
    for r in range(num_reqs):
        start = int(cu[r].item())
        end = int(cu[r + 1].item())
        n = end - start
        out[start:end] = block_table[r, :n]
    return out


def kernel_impl(block_table, cu_num_blocks):
    """Kernel wrapper: launch only (replicates the class-method launch site)."""
    num_reqs = block_table.shape[0]
    num_actual_pages = int(cu_num_blocks.cpu()[num_reqs].item())
    page_indices = torch.zeros(
        num_actual_pages, dtype=block_table.dtype, device=block_table.device
    )
    _copy_page_indices_kernel[(num_reqs,)](
        page_indices,
        block_table,
        block_table.stride(0),
        cu_num_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return page_indices


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
        ref_out = pytorch_ref(BLOCK_TABLE, CU_NUM_BLOCKS)
        kernel_out = kernel_impl(BLOCK_TABLE, CU_NUM_BLOCKS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        # Integer data: require EXACT equality (no float tolerance).
        exact_match = bool(torch.equal(kernel_cpu, ref_cpu))
        assert exact_match, (
            f"exact int mismatch: ref={ref_cpu.tolist()} "
            f"kernel={kernel_cpu.tolist()}"
        )
        num_mismatch = int((kernel_cpu != ref_cpu).sum().item())

        stats = {
            "block_table_shape": tuple(BLOCK_TABLE.shape),
            "indptr": CU_NUM_BLOCKS.cpu().tolist(),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(kernel_out.dtype),
            "device": str(BLOCK_TABLE.device),
            "exact_match": exact_match,
            "num_mismatch": num_mismatch,
            "max_abs_diff": 0,      # exact integer match
            "mean_abs_diff": 0.0,
        }

        pt_stats = _bench(lambda: pytorch_ref(BLOCK_TABLE, CU_NUM_BLOCKS))
        kern_stats = _bench(lambda: kernel_impl(BLOCK_TABLE, CU_NUM_BLOCKS))
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
            "Kernel: _copy_page_indices_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- block_table shape: {stats['block_table_shape']}\n")
            lines.append(f"- indptr (cu_num_blocks): {stats['indptr']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Comparison: EXACT integer equality\n")
            lines.append("Output:\n")
            lines.append(f"- page_indices shape: {stats['output_shape']}\n")
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
