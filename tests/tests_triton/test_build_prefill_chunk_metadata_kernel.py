"""
Standalone QAIC validation for `_build_prefill_chunk_metadata_kernel`.

Source under test:
vllm/v1/attention/backends/mla/indexer.py
  - _build_prefill_chunk_metadata_kernel

For chunked prefill of the sparse indexer, this kernel builds, per query token,
the KV range [cu_seqlen_ks, cu_seqlen_ke) into the (compressed) KV cache, plus
a token_to_seq mapping (which request each compressed KV token belongs to).

Per batch (grid = (num_reqs,)):
  query_start/end from rebased query_start_loc
  seq_start/end   from cumulative compressed seq lens
  start_pos = uncompressed_seq_len - query_len
  for each in-slice query offset o:
    ks[out_pos] = seq_start
    ke[out_pos] = seq_start + (start_pos + 1 + o) // COMPRESS_RATIO
  token_to_seq[seq_start + j] = batch_idx  for j in [0, compressed_seq_len)

We use the simplest config: COMPRESS_RATIO=1 (no compression) and the full
query slice (query_slice_start=0, stop=total_query_len). All three outputs
(ks, ke, token_to_seq) are integer arrays validated with EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.indexer import (
    _build_prefill_chunk_metadata_kernel,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_build_prefill_chunk_metadata_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/indexer.py"
KERNEL_NAME = "_build_prefill_chunk_metadata_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
COMPRESS_RATIO = 1  # simplest config: no KV compression
QUERY_LENS = [3, 4]
NUM_REQS = len(QUERY_LENS)
SEQ_LENS_LIST = [5, 6]  # uncompressed context+query seq lens (>= query lens)

# rebased query_start_loc: [0, 3, 7]
QUERY_START_LOC = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(QUERY_LENS), 0).tolist()),
    dtype=torch.int32,
    device=DEVICE,
)
UNCOMPRESSED_SEQ_LENS = torch.tensor(SEQ_LENS_LIST, dtype=torch.int32, device=DEVICE)
# compressed seq lens == uncompressed (compress_ratio == 1)
CU_COMPRESSED = torch.tensor(
    [0] + list(torch.cumsum(UNCOMPRESSED_SEQ_LENS, 0).cpu().tolist()),
    dtype=torch.int32,
    device=DEVICE,
)
TOTAL_SEQ_LENS = int(CU_COMPRESSED[-1].item())
TOTAL_QUERY_LEN = int(QUERY_START_LOC[-1].item())
QUERY_SLICE_START = 0
QUERY_SLICE_STOP = TOTAL_QUERY_LEN


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


def pytorch_ref(query_start_loc, uncompressed_seq_lens, cu_compressed):
    query_start_loc = query_start_loc.cpu()
    uncompressed_seq_lens = uncompressed_seq_lens.cpu()
    cu_compressed = cu_compressed.cpu()

    ks = torch.zeros(QUERY_SLICE_STOP - QUERY_SLICE_START, dtype=torch.int32)
    ke = torch.zeros(QUERY_SLICE_STOP - QUERY_SLICE_START, dtype=torch.int32)
    token_to_seq = torch.zeros(TOTAL_SEQ_LENS, dtype=torch.int32)

    for b in range(NUM_REQS):
        q_start = int(query_start_loc[b].item())
        q_end = int(query_start_loc[b + 1].item())
        query_len = q_end - q_start
        seq_start = int(cu_compressed[b].item())
        seq_end = int(cu_compressed[b + 1].item())
        compressed_seq_len = seq_end - seq_start
        uncompressed_seq_len = int(uncompressed_seq_lens[b].item())
        start_pos = uncompressed_seq_len - query_len
        for o in range(query_len):
            abs_pos = q_start + o
            if QUERY_SLICE_START <= abs_pos < QUERY_SLICE_STOP:
                out_pos = abs_pos - QUERY_SLICE_START
                ks[out_pos] = seq_start
                ke[out_pos] = seq_start + (start_pos + 1 + o) // COMPRESS_RATIO
        for j in range(compressed_seq_len):
            token_to_seq[seq_start + j] = b
    return ks, ke, token_to_seq


def kernel_impl(query_start_loc, uncompressed_seq_lens, cu_compressed):
    ks = torch.zeros(
        QUERY_SLICE_STOP - QUERY_SLICE_START, dtype=torch.int32, device=DEVICE
    )
    ke = torch.zeros(
        QUERY_SLICE_STOP - QUERY_SLICE_START, dtype=torch.int32, device=DEVICE
    )
    token_to_seq = torch.zeros(TOTAL_SEQ_LENS, dtype=torch.int32, device=DEVICE)
    _build_prefill_chunk_metadata_kernel[(NUM_REQS,)](
        query_start_loc,
        uncompressed_seq_lens,
        cu_compressed,
        token_to_seq,
        ks,
        ke,
        QUERY_SLICE_START,
        QUERY_SLICE_STOP,
        BLOCK_SIZE=1024,
        COMPRESS_RATIO=COMPRESS_RATIO,
    )
    return ks, ke, token_to_seq


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
        ref = pytorch_ref(QUERY_START_LOC, UNCOMPRESSED_SEQ_LENS, CU_COMPRESSED)
        ker = kernel_impl(QUERY_START_LOC, UNCOMPRESSED_SEQ_LENS, CU_COMPRESSED)

        names = ["cu_seqlen_ks", "cu_seqlen_ke", "token_to_seq"]
        total_mism = 0
        max_diff = 0
        for n, r, k in zip(names, ref, ker):
            m, d = _exact(r, k)
            total_mism += m
            max_diff = max(max_diff, d)
        assert total_mism == 0, f"{total_mism} mismatched elements"

        stats = {
            "num_reqs": NUM_REQS,
            "compress_ratio": COMPRESS_RATIO,
            "total_seq_lens": TOTAL_SEQ_LENS,
            "total_query_len": TOTAL_QUERY_LEN,
            "dtype": str(QUERY_START_LOC.dtype),
            "device": str(QUERY_START_LOC.device),
            "mismatch_count": total_mism,
            "max_abs_int_diff": max_diff,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(
            lambda: pytorch_ref(QUERY_START_LOC, UNCOMPRESSED_SEQ_LENS, CU_COMPRESSED)
        )
        kern_stats = _bench(
            lambda: kernel_impl(QUERY_START_LOC, UNCOMPRESSED_SEQ_LENS, CU_COMPRESSED)
        )
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
