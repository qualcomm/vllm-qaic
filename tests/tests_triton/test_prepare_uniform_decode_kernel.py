"""
Standalone QAIC validation for `_prepare_uniform_decode_kernel`.

Source under test:
vllm/v1/attention/backends/mla/indexer.py
  - _prepare_uniform_decode_kernel

For uniform-length multi-token decode, this kernel expands per-request decode
metadata into a uniform one-token-per-decode-step layout. For each expanded
index `idx` (grid covers num_decodes * max_decode_len):
  req_id    = idx // max_decode_len
  local_idx = idx %  max_decode_len
  decode_seq_lens[idx] = seq_lens[req_id] - max_decode_len + local_idx + 1
  expanded_block_table[idx] = block_table[req_id]   (row copy)
  decode_lens[idx] = 1

This is pure integer index arithmetic; we validate all three outputs with
EXACT integer equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.indexer import _prepare_uniform_decode_kernel

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prepare_uniform_decode_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/indexer.py"
KERNEL_NAME = "_prepare_uniform_decode_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
NUM_DECODES = 3
MAX_DECODE_LEN = 2
NUM_DECODE_TOKENS = NUM_DECODES * MAX_DECODE_LEN
NCOLS = 4  # columns of the (expanded) block table

SEQ_LENS = torch.tensor([10, 7, 12], dtype=torch.int32, device=DEVICE)
BLOCK_TABLE = torch.arange(
    NUM_DECODES * NCOLS, dtype=torch.int32, device=DEVICE
).reshape(NUM_DECODES, NCOLS)


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


def pytorch_ref(seq_lens, block_table):
    """Pure PyTorch integer expansion of the uniform decode metadata."""
    seq_lens = seq_lens.cpu()
    block_table = block_table.cpu()
    decode_seq_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32)
    expanded_bt = torch.zeros(NUM_DECODE_TOKENS, NCOLS, dtype=torch.int32)
    decode_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32)
    for idx in range(NUM_DECODE_TOKENS):
        req_id = idx // MAX_DECODE_LEN
        local_idx = idx % MAX_DECODE_LEN
        decode_seq_lens[idx] = int(seq_lens[req_id].item()) - MAX_DECODE_LEN + local_idx + 1
        expanded_bt[idx] = block_table[req_id]
        decode_lens[idx] = 1
    return decode_seq_lens, expanded_bt, decode_lens


def kernel_impl(seq_lens, block_table):
    """Kernel launch only."""
    decode_seq_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32, device=DEVICE)
    expanded_bt = torch.zeros(
        NUM_DECODE_TOKENS, NCOLS, dtype=torch.int32, device=DEVICE
    )
    decode_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32, device=DEVICE)
    _prepare_uniform_decode_kernel[(NUM_DECODE_TOKENS,)](
        seq_lens,
        decode_seq_lens,
        block_table,
        block_table.stride(0),
        expanded_bt,
        expanded_bt.stride(0),
        decode_lens,
        MAX_DECODE_LEN,
        BLOCK_SIZE=1024,
    )
    return decode_seq_lens, expanded_bt, decode_lens


def _exact(ref, ker):
    ref = ref.cpu()
    ker = ker.cpu()
    mism = int((ref != ker).sum().item())
    if ref.numel() > 0:
        maxdiff = int((ref.to(torch.int64) - ker.to(torch.int64)).abs().max().item())
    else:
        maxdiff = 0
    return mism, maxdiff


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(SEQ_LENS, BLOCK_TABLE)
        ker = kernel_impl(SEQ_LENS, BLOCK_TABLE)

        names = ["decode_seq_lens", "expanded_block_table", "decode_lens"]
        total_mism = 0
        max_diff = 0
        for n, r, k in zip(names, ref, ker):
            m, d = _exact(r, k)
            total_mism += m
            max_diff = max(max_diff, d)
        assert total_mism == 0, f"{total_mism} mismatched elements"

        stats = {
            "num_decodes": NUM_DECODES,
            "max_decode_len": MAX_DECODE_LEN,
            "num_decode_tokens": NUM_DECODE_TOKENS,
            "seq_lens_shape": tuple(SEQ_LENS.shape),
            "block_table_shape": tuple(BLOCK_TABLE.shape),
            "dtype": str(SEQ_LENS.dtype),
            "device": str(SEQ_LENS.device),
            "mismatch_count": total_mism,
            "max_abs_int_diff": max_diff,
            "grid": f"({NUM_DECODE_TOKENS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(SEQ_LENS, BLOCK_TABLE))
        kern_stats = _bench(lambda: kernel_impl(SEQ_LENS, BLOCK_TABLE))
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
