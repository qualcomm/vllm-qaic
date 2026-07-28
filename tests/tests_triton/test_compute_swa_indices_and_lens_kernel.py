"""
Standalone QAIC validation for `_compute_swa_indices_and_lens_kernel`.

Source under test:
vllm/v1/attention/backends/mla/sparse_swa.py
  - _compute_swa_indices_and_lens_kernel

Per decode token, computes the sliding-window KV length and the gathered
physical block slot indices from the block table. This kernel has no standalone
launcher; the grid/args are replicated from its launch site in
DeepseekSparseSWAMetadataBuilder.build (grid=(num_decode_tokens,),
TRITON_BLOCK_SIZE=1024).

For each decode token (grid = (num_decode_tokens,)):
  if not is_valid_token[t]: swa_lens[t]=0; leave swa_indices untouched.
  else:
    req = token_to_req_indices[t]
    query_len = query_start_loc[req+1] - query_start_loc[req]
    prefix_len = seq_lens[req] - query_len
    pos = prefix_len + t - query_start_loc[req]
    start_pos = max(pos - window_size + 1, 0) ; end_pos = pos + 1
    swa_len = end_pos - start_pos
    for off in [0, window_size):
      pos_off = start_pos + off
      slot = block_table[req, pos_off//block_size]*block_size + pos_off%block_size
      swa_indices[t, off] = slot if off < swa_len else -1

Both outputs (swa_indices, swa_lens) are integer arrays; EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_swa_indices_and_lens_kernel,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_swa_indices_and_lens_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/sparse_swa.py"
KERNEL_NAME = "_compute_swa_indices_and_lens_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
WINDOW_SIZE = 8
BLOCK_SIZE = 4
# Each decode token is its own single-token request (query_len == 1).
NUM_DECODE_TOKENS = 3
SEQ_LENS_LIST = [10, 5, 20]  # per-request context+1 lengths
MAX_NUM_BLOCKS = (max(SEQ_LENS_LIST) + BLOCK_SIZE - 1) // BLOCK_SIZE + 1

SEQ_LENS = torch.tensor(SEQ_LENS_LIST, dtype=torch.int32, device=DEVICE)
QUERY_START_LOC = torch.arange(
    NUM_DECODE_TOKENS + 1, dtype=torch.int32, device=DEVICE
)  # [0,1,2,3] -> query_len 1 each
TOKEN_TO_REQ = torch.arange(NUM_DECODE_TOKENS, dtype=torch.int32, device=DEVICE)
IS_VALID_TOKEN = torch.tensor([True, False, True], device=DEVICE)  # token 1 invalid
BLOCK_TABLE = (
    torch.arange(
        NUM_DECODE_TOKENS * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE
    ).reshape(NUM_DECODE_TOKENS, MAX_NUM_BLOCKS)
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


def pytorch_ref(seq_lens, query_start_loc, token_to_req, is_valid_token, block_table):
    seq_lens = seq_lens.cpu()
    query_start_loc = query_start_loc.cpu()
    token_to_req = token_to_req.cpu()
    is_valid_token = is_valid_token.cpu()
    block_table = block_table.cpu()

    swa_indices = torch.zeros(NUM_DECODE_TOKENS, WINDOW_SIZE, dtype=torch.int32)
    swa_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32)

    for t in range(NUM_DECODE_TOKENS):
        if not bool(is_valid_token[t].item()):
            swa_lens[t] = 0
            continue
        req = int(token_to_req[t].item())
        q_start = int(query_start_loc[req].item())
        q_end = int(query_start_loc[req + 1].item())
        query_len = q_end - q_start
        seq_len = int(seq_lens[req].item())
        prefix_len = seq_len - query_len
        pos = prefix_len + t - q_start
        start_pos = max(pos - WINDOW_SIZE + 1, 0)
        end_pos = pos + 1
        swa_len = end_pos - start_pos
        swa_lens[t] = swa_len
        for off in range(WINDOW_SIZE):
            pos_off = start_pos + off
            if off < swa_len:
                block_number = int(block_table[req, pos_off // BLOCK_SIZE].item())
                slot = block_number * BLOCK_SIZE + pos_off % BLOCK_SIZE
                swa_indices[t, off] = slot
            else:
                swa_indices[t, off] = -1
    return swa_indices, swa_lens


def kernel_impl(seq_lens, query_start_loc, token_to_req, is_valid_token, block_table):
    # Mirror the (max_tokens, 1, window_size) buffer layout of the launch site.
    swa_indices = torch.zeros(
        NUM_DECODE_TOKENS, 1, WINDOW_SIZE, dtype=torch.int32, device=DEVICE
    )
    swa_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32, device=DEVICE)
    _compute_swa_indices_and_lens_kernel[(NUM_DECODE_TOKENS,)](
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        WINDOW_SIZE,
        query_start_loc,
        seq_lens,
        token_to_req,
        is_valid_token,
        block_table,
        block_table.stride(0),
        BLOCK_SIZE,
        TRITON_BLOCK_SIZE=1024,
    )
    return swa_indices.view(NUM_DECODE_TOKENS, WINDOW_SIZE), swa_lens


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
        ref = pytorch_ref(
            SEQ_LENS, QUERY_START_LOC, TOKEN_TO_REQ, IS_VALID_TOKEN, BLOCK_TABLE
        )
        ker = kernel_impl(
            SEQ_LENS, QUERY_START_LOC, TOKEN_TO_REQ, IS_VALID_TOKEN, BLOCK_TABLE
        )
        names = ["swa_indices", "swa_lens"]
        total_mism = 0
        max_diff = 0
        for n, r, k in zip(names, ref, ker):
            m, d = _exact(r, k)
            total_mism += m
            max_diff = max(max_diff, d)
        assert total_mism == 0, f"{total_mism} mismatched elements"

        stats = {
            "num_decode_tokens": NUM_DECODE_TOKENS,
            "window_size": WINDOW_SIZE,
            "block_size": BLOCK_SIZE,
            "seq_lens": SEQ_LENS_LIST,
            "is_valid_token": IS_VALID_TOKEN.cpu().tolist(),
            "dtype": "torch.int32",
            "device": DEVICE,
            "mismatch_count": total_mism,
            "max_abs_int_diff": max_diff,
            "grid": f"({NUM_DECODE_TOKENS},)",
        }
        pt_stats = _bench(
            lambda: pytorch_ref(
                SEQ_LENS, QUERY_START_LOC, TOKEN_TO_REQ, IS_VALID_TOKEN, BLOCK_TABLE
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                SEQ_LENS, QUERY_START_LOC, TOKEN_TO_REQ, IS_VALID_TOKEN, BLOCK_TABLE
            )
        )
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
