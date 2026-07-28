"""
Standalone QAIC validation for `generate_sparse_seqlen_kernel`.

Source under test:
vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
  - generate_sparse_seqlen_kernel
  - generate_sparse_seqlen_triton  (launcher)

Per query token, computes the effective sparse context length, clamped
(capped) at top-k. For each seq (grid = (num_seqs, blocks)) and each local
query offset o in [0, query_len):
  context_start = seq_len - query_len
  sparse_seqlen = context_start + o
  out[query_start + o] = min(sparse_seqlen + 1, topk_token)
Sequences with seq_len == 0 leave their (zero-initialized) region untouched.

Integer arithmetic; validated with EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    generate_sparse_seqlen_triton,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_generate_sparse_seqlen_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py"
KERNEL_NAME = "generate_sparse_seqlen_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
TOPK_TOKEN = 8
QUERY_LENS_LIST = [3, 2, 4]
NUM_SEQS = len(QUERY_LENS_LIST)
# seq_lens chosen so some sparse_seqlen+1 exceed TOPK (clamped) and some do not.
SEQ_LENS_LIST = [4, 20, 6]

QUERY_LENS = torch.tensor(QUERY_LENS_LIST, dtype=torch.int32, device=DEVICE)
SEQ_LENS = torch.tensor(SEQ_LENS_LIST, dtype=torch.int32, device=DEVICE)
CU_QUERY_LENS = torch.tensor(
    [0] + list(torch.cumsum(QUERY_LENS, 0).cpu().tolist()),
    dtype=torch.int32,
    device=DEVICE,
)
NUM_TOKENS = int(CU_QUERY_LENS[-1].item())
MAX_QUERY_LEN = max(QUERY_LENS_LIST)


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


def pytorch_ref(query_lens, seq_lens, cu_query_lens):
    query_lens = query_lens.cpu()
    seq_lens = seq_lens.cpu()
    cu_query_lens = cu_query_lens.cpu()
    out = torch.zeros(NUM_TOKENS, dtype=torch.int32)
    for seq_id in range(NUM_SEQS):
        q_start = int(cu_query_lens[seq_id].item())
        q_end = int(cu_query_lens[seq_id + 1].item())
        query_len = q_end - q_start
        seq_len = int(seq_lens[seq_id].item())
        if seq_len == 0:
            continue
        context_start = seq_len - query_len
        for o in range(query_len):
            sparse_seqlen = context_start + o
            out[q_start + o] = min(sparse_seqlen + 1, TOPK_TOKEN)
    return out


def kernel_impl(query_lens, seq_lens, cu_query_lens):
    return generate_sparse_seqlen_triton(
        query_lens,
        seq_lens,
        cu_query_lens,
        TOPK_TOKEN,
        NUM_TOKENS,
        MAX_QUERY_LEN,
    )


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
        ref = pytorch_ref(QUERY_LENS, SEQ_LENS, CU_QUERY_LENS)
        ker = kernel_impl(QUERY_LENS, SEQ_LENS, CU_QUERY_LENS)
        mism, maxdiff = _exact(ref, ker)
        assert mism == 0, f"{mism} mismatched elements"

        stats = {
            "num_seqs": NUM_SEQS,
            "topk_token": TOPK_TOKEN,
            "query_lens": QUERY_LENS_LIST,
            "seq_lens": SEQ_LENS_LIST,
            "num_tokens": NUM_TOKENS,
            "dtype": "torch.int32",
            "device": DEVICE,
            "mismatch_count": mism,
            "max_abs_int_diff": maxdiff,
            "grid": f"({NUM_SEQS}, cdiv({MAX_QUERY_LEN},64))",
        }
        pt_stats = _bench(lambda: pytorch_ref(QUERY_LENS, SEQ_LENS, CU_QUERY_LENS))
        kern_stats = _bench(lambda: kernel_impl(QUERY_LENS, SEQ_LENS, CU_QUERY_LENS))
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
