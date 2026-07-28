"""
Standalone QAIC validation for `_convert_req_index_to_global_index_kernel`
(the sparse_utils version -- distinct from the rocm_aiter_mla_sparse one).

Source under test:
vllm/v1/attention/backends/mla/sparse_utils.py
  - _convert_req_index_to_global_index_kernel
  - triton_convert_req_index_to_global_index  (launcher)

Converts per-token sparse top-k local token indices into global paged KV-cache
slot indices, into a DENSE [num_tokens, NUM_TOPK_TOKENS] output, and (optionally)
tracks the per-row count of valid (non -1) entries.

Simplest config used here: HAS_PREFILL_WORKSPACE=False (no prefill-workspace
override) and return_valid_counts=True. For each token row / column:
  tok = token_indices[token_id, indice_id]
  block_id = tok // BLOCK_SIZE ; off = tok % BLOCK_SIZE
  valid_block = 0 <= block_id < max_num_blocks_per_req
  invalid = (tok < 0) or (not valid_block)
  out[token_id, indice_id] =
      -1 if invalid else block_table[req_id[token_id], block_id]*BLOCK_SIZE + off
  valid_count[token_id] = number of non-invalid entries in the row

Note: unlike the rocm version, invalid entries map to -1 (not 0), and the full
dense grid is always written. Both outputs validated with EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_convert_req_index_to_global_index_kernel_sparse_utils.txt"
)
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/sparse_utils.py"
KERNEL_NAME = "_convert_req_index_to_global_index_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
BLOCK_SIZE = 4
NUM_TOPK_TOKENS = 8
BLOCK_N = 4
MAX_NUM_BLOCKS = 4
NUM_REQUESTS = 2

REQ_ID = torch.tensor([0, 1, 1], dtype=torch.int32, device=DEVICE)
NUM_TOKENS = REQ_ID.shape[0]

# local token indices in [0, 16); scatter a few -1 and one OOB (>= 16 -> block 4)
TOKEN_INDICES = torch.randint(
    0, 16, (NUM_TOKENS, NUM_TOPK_TOKENS), dtype=torch.int32, device=DEVICE
)
TOKEN_INDICES[0, 2] = -1
TOKEN_INDICES[1, 0] = -1
TOKEN_INDICES[2, 3] = 60  # block_id 15 -> OOB -> -1

BLOCK_TABLE = (
    torch.arange(
        NUM_REQUESTS * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE
    ).reshape(NUM_REQUESTS, MAX_NUM_BLOCKS)
    + 1
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


def pytorch_ref(req_id, block_table, token_indices):
    req_id = req_id.cpu()
    block_table = block_table.cpu()
    token_indices = token_indices.cpu()
    out = torch.empty(NUM_TOKENS, NUM_TOPK_TOKENS, dtype=torch.int32)
    valid_counts = torch.zeros(NUM_TOKENS, dtype=torch.int32)
    for token_id in range(NUM_TOKENS):
        req = int(req_id[token_id].item())
        cnt = 0
        for indice_id in range(NUM_TOPK_TOKENS):
            tok = int(token_indices[token_id, indice_id].item())
            invalid = tok < 0
            block_id = tok // BLOCK_SIZE
            off = tok % BLOCK_SIZE
            valid_block = 0 <= block_id < MAX_NUM_BLOCKS
            invalid = invalid or (not valid_block)
            if invalid:
                out[token_id, indice_id] = -1
            else:
                out[token_id, indice_id] = (
                    int(block_table[req, block_id].item()) * BLOCK_SIZE + off
                )
                cnt += 1
        valid_counts[token_id] = cnt
    return out, valid_counts


def kernel_impl(req_id, block_table, token_indices):
    out, valid_counts = triton_convert_req_index_to_global_index(
        req_id,
        block_table,
        token_indices,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_TOPK_TOKENS=NUM_TOPK_TOKENS,
        BLOCK_N=BLOCK_N,
        HAS_PREFILL_WORKSPACE=False,
        return_valid_counts=True,
    )
    return out, valid_counts


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
        ref = pytorch_ref(REQ_ID, BLOCK_TABLE, TOKEN_INDICES)
        ker = kernel_impl(REQ_ID, BLOCK_TABLE, TOKEN_INDICES)
        names = ["global_indices", "valid_counts"]
        total_mism = 0
        max_diff = 0
        for n, r, k in zip(names, ref, ker):
            m, d = _exact(r, k)
            total_mism += m
            max_diff = max(max_diff, d)
        assert total_mism == 0, f"{total_mism} mismatched elements"

        stats = {
            "num_tokens": NUM_TOKENS,
            "block_size": BLOCK_SIZE,
            "num_topk_tokens": NUM_TOPK_TOKENS,
            "block_n": BLOCK_N,
            "has_prefill_workspace": False,
            "return_valid_counts": True,
            "dtype": str(TOKEN_INDICES.dtype),
            "device": str(TOKEN_INDICES.device),
            "mismatch_count": total_mism,
            "max_abs_int_diff": max_diff,
            "grid": f"({NUM_TOKENS}, {NUM_TOPK_TOKENS // BLOCK_N})",
        }
        pt_stats = _bench(lambda: pytorch_ref(REQ_ID, BLOCK_TABLE, TOKEN_INDICES))
        kern_stats = _bench(lambda: kernel_impl(REQ_ID, BLOCK_TABLE, TOKEN_INDICES))
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
    sys.exit(0 if main() == "SUCCESS" else 1)
