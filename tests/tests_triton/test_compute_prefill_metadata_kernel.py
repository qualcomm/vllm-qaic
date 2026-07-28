"""
Standalone QAIC validation for `_compute_prefill_metadata_kernel`.

Source under test:
vllm/v1/attention/backends/mla/sparse_swa.py
  - _compute_prefill_metadata_kernel

Per-prefill-request sliding-window "gather length" metadata. This kernel has no
standalone launcher; the grid/args are replicated from its launch site in
DeepseekSparseSWAMetadataBuilder._build_deepseek_v4_metadata (grid=(1,),
BLOCK_SIZE=next_power_of_2(num_prefills)).

For each prefill request p (offset over [0, num_prefills)):
  seq_len   = seq_lens[num_decodes + p]
  query_len = query_start_loc[num_decodes+p+1] - query_start_loc[num_decodes+p]
  prefix_len = seq_len - query_len
  gather_len = query_len + min(prefix_len, window_size - 1)

Integer arithmetic; validated with EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.triton_utils import triton
from vllm.v1.attention.backends.mla.sparse_swa import (
    _compute_prefill_metadata_kernel,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_prefill_metadata_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/sparse_swa.py"
KERNEL_NAME = "_compute_prefill_metadata_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
WINDOW_SIZE = 8
NUM_DECODES = 1  # decodes precede prefills in the reordered batch
PREFILL_QUERY_LENS = [4, 3]
NUM_PREFILLS = len(PREFILL_QUERY_LENS)
# full query_start_loc includes the leading decode(s); decode query_len == 1.
DECODE_QUERY_LENS = [1] * NUM_DECODES
ALL_QUERY_LENS = DECODE_QUERY_LENS + PREFILL_QUERY_LENS

QUERY_START_LOC = torch.tensor(
    [0] + list(torch.cumsum(torch.tensor(ALL_QUERY_LENS), 0).tolist()),
    dtype=torch.int32,
    device=DEVICE,
)
# seq_lens for all requests: decodes then prefills. Prefills have prefix
# context so seq_len > query_len; pick some > window and some < window.
DECODE_SEQ_LENS = [12] * NUM_DECODES
PREFILL_SEQ_LENS = [20, 5]  # prefixes: 20-4=16 (> window-1), 5-3=2 (< window-1)
SEQ_LENS = torch.tensor(
    DECODE_SEQ_LENS + PREFILL_SEQ_LENS, dtype=torch.int32, device=DEVICE
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


def pytorch_ref(seq_lens, query_start_loc):
    seq_lens = seq_lens.cpu()
    query_start_loc = query_start_loc.cpu()
    out = torch.zeros(NUM_PREFILLS, dtype=torch.int32)
    for p in range(NUM_PREFILLS):
        seq_len = int(seq_lens[NUM_DECODES + p].item())
        qsl_start = int(query_start_loc[NUM_DECODES + p].item())
        qsl_end = int(query_start_loc[NUM_DECODES + p + 1].item())
        query_len = qsl_end - qsl_start
        prefix_len = seq_len - query_len
        out[p] = query_len + min(prefix_len, WINDOW_SIZE - 1)
    return out


def kernel_impl(seq_lens, query_start_loc):
    out = torch.zeros(NUM_PREFILLS, dtype=torch.int32, device=DEVICE)
    _compute_prefill_metadata_kernel[(1,)](
        out,
        seq_lens,
        query_start_loc,
        NUM_PREFILLS,
        NUM_DECODES,
        WINDOW_SIZE,
        BLOCK_SIZE=triton.next_power_of_2(NUM_PREFILLS),
    )
    return out


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
        ref = pytorch_ref(SEQ_LENS, QUERY_START_LOC)
        ker = kernel_impl(SEQ_LENS, QUERY_START_LOC)
        mism, maxdiff = _exact(ref, ker)
        assert mism == 0, f"{mism} mismatched elements"

        stats = {
            "num_prefills": NUM_PREFILLS,
            "num_decodes": NUM_DECODES,
            "window_size": WINDOW_SIZE,
            "prefill_query_lens": PREFILL_QUERY_LENS,
            "prefill_seq_lens": PREFILL_SEQ_LENS,
            "dtype": "torch.int32",
            "device": DEVICE,
            "mismatch_count": mism,
            "max_abs_int_diff": maxdiff,
            "grid": "(1,)",
        }
        pt_stats = _bench(lambda: pytorch_ref(SEQ_LENS, QUERY_START_LOC))
        kern_stats = _bench(lambda: kernel_impl(SEQ_LENS, QUERY_START_LOC))
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
