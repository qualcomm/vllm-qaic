"""
Standalone QAIC validation for `_expand_page_indices_kernel`.

Source under test:
vllm/v1/attention/backends/mla/rocm_aiter_mla.py
  - _expand_page_indices_kernel

The aiter MLA decode kernel always operates with page_size=1 internally, so
block-table entries must be expanded into per-token flat page indices. For each
request (grid = (num_reqs,)) and each token t in [0, seq_lens[req]):
  block_idx       = t // KERNEL_BLOCK_SIZE
  offset_in_block = t %  KERNEL_BLOCK_SIZE
  block_id        = block_table[req, block_idx]
  flat            = block_id * KERNEL_BLOCK_SIZE + offset_in_block
  page_indices[cu_num_tokens[req] + t] = flat

cu_num_tokens is the exclusive-cumsum (indptr) of seq_lens. Integer arithmetic;
validated with EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.rocm_aiter_mla import _expand_page_indices_kernel

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_expand_page_indices_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/rocm_aiter_mla.py"
KERNEL_NAME = "_expand_page_indices_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
KERNEL_BLOCK_SIZE = 4
SEQ_LENS_LIST = [6, 3, 9]  # per-request token counts
NUM_REQS = len(SEQ_LENS_LIST)
MAX_NUM_BLOCKS = (max(SEQ_LENS_LIST) + KERNEL_BLOCK_SIZE - 1) // KERNEL_BLOCK_SIZE

SEQ_LENS = torch.tensor(SEQ_LENS_LIST, dtype=torch.int32, device=DEVICE)
# indptr (exclusive cumsum) with leading zero.
CU_NUM_TOKENS = torch.tensor(
    [0] + list(torch.cumsum(SEQ_LENS, 0).cpu().tolist()),
    dtype=torch.int32,
    device=DEVICE,
)
TOTAL_TOKENS = int(CU_NUM_TOKENS[-1].item())
# Distinct block ids per request so the mapping is unambiguous.
BLOCK_TABLE = torch.arange(
    NUM_REQS * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE
).reshape(NUM_REQS, MAX_NUM_BLOCKS) + 1


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


def pytorch_ref(block_table, cu_num_tokens, seq_lens):
    block_table = block_table.cpu()
    cu_num_tokens = cu_num_tokens.cpu()
    seq_lens = seq_lens.cpu()
    page_indices = torch.zeros(TOTAL_TOKENS, dtype=torch.int32)
    for req in range(NUM_REQS):
        start = int(cu_num_tokens[req].item())
        n = int(seq_lens[req].item())
        for t in range(n):
            block_idx = t // KERNEL_BLOCK_SIZE
            offset_in_block = t % KERNEL_BLOCK_SIZE
            block_id = int(block_table[req, block_idx].item())
            page_indices[start + t] = block_id * KERNEL_BLOCK_SIZE + offset_in_block
    return page_indices


def kernel_impl(block_table, cu_num_tokens, seq_lens):
    page_indices = torch.zeros(TOTAL_TOKENS, dtype=torch.int32, device=DEVICE)
    _expand_page_indices_kernel[(NUM_REQS,)](
        page_indices,
        block_table,
        block_table.stride(0),
        cu_num_tokens,
        seq_lens,
        KERNEL_BLOCK_SIZE=KERNEL_BLOCK_SIZE,
        BLOCK_SIZE=1024,
    )
    return page_indices


def _exact(ref, ker):
    ref = ref.cpu()
    ker = ker.cpu()
    mism = int((ref != ker).sum().item())
    maxdiff = (
        int((ref.to(torch.int64) - ker.to(torch.int64)).abs().max().item())
        if ref.numel() > 0
        else 0
    )
    return mism, maxdiff


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(BLOCK_TABLE, CU_NUM_TOKENS, SEQ_LENS)
        ker = kernel_impl(BLOCK_TABLE, CU_NUM_TOKENS, SEQ_LENS)
        mism, maxdiff = _exact(ref, ker)
        assert mism == 0, f"{mism} mismatched elements"

        stats = {
            "num_reqs": NUM_REQS,
            "kernel_block_size": KERNEL_BLOCK_SIZE,
            "seq_lens": SEQ_LENS_LIST,
            "total_tokens": TOTAL_TOKENS,
            "block_table_shape": tuple(BLOCK_TABLE.shape),
            "dtype": str(BLOCK_TABLE.dtype),
            "device": str(BLOCK_TABLE.device),
            "mismatch_count": mism,
            "max_abs_int_diff": maxdiff,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(BLOCK_TABLE, CU_NUM_TOKENS, SEQ_LENS))
        kern_stats = _bench(lambda: kernel_impl(BLOCK_TABLE, CU_NUM_TOKENS, SEQ_LENS))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            f"Kernel: {KERNEL_NAME}\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
