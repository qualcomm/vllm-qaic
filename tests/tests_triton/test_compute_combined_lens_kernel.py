"""
Standalone QAIC validation for `_compute_combined_lens_kernel`.

Source under test:
vllm/models/deepseek_v4/amd/rocm.py
  - _compute_combined_lens_kernel
  (launched inside combine_topk_swa_indices_ragged; driven directly here)

Computes the combined (topk_len + swa_len) active length per query token, purely
from each token's absolute position:

    topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
    swa_len  = min(pos + 1, WINDOW_SIZE)
    combined_lens[token] = topk_len + swa_len

Integer metadata kernel -> EXACT-equality comparison. Grid = (num_reqs,
num_workers); we launch the kernel directly with the same argument list as the
source wrapper.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.amd.rocm import _compute_combined_lens_kernel

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_combined_lens_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/amd/rocm.py"

DEVICE = "qaic"
torch.manual_seed(42)

COMPRESS_RATIO = 4
WINDOW_SIZE = 8
TOP_K = 8
NUM_REQS = 3
NUM_WORKERS = 128

QUERY_START_LOC = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device=DEVICE)
NUM_TOKENS = int(QUERY_START_LOC[-1].item())
SEQ_LENS = torch.tensor([6, 20, 40], dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(query_start_loc, seq_lens):
    qsl = query_start_loc.cpu()
    seq_lens = seq_lens.cpu()
    lens = torch.zeros(NUM_TOKENS, dtype=torch.int32)
    base = int(qsl[0].item())
    for b in range(NUM_REQS):
        qs = int(qsl[b].item()) - base
        qe = int(qsl[b + 1].item()) - base
        ql = qe - qs
        sl = int(seq_lens[b].item())
        start_pos = sl - ql
        for tok in range(qs, qe):
            pos = start_pos + (tok - qs)
            topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
            swa_len = min(pos + 1, WINDOW_SIZE)
            lens[tok] = topk_len + swa_len
    return lens


def kernel_impl(query_start_loc, seq_lens):
    combined_lens = torch.empty(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
    _compute_combined_lens_kernel[(NUM_REQS, NUM_WORKERS)](
        combined_lens,
        query_start_loc,
        seq_lens,
        TOP_K=TOP_K,
        COMPRESS_RATIO=COMPRESS_RATIO,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    return combined_lens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(QUERY_START_LOC, SEQ_LENS)
        out = kernel_impl(QUERY_START_LOC, SEQ_LENS).cpu()
        mismatch = int((out != ref).sum().item())
        assert mismatch == 0, f"combined_lens mismatch={mismatch}"

        stats = {
            "input_shape": tuple(QUERY_START_LOC.shape),
            "output_shape": tuple(out.shape),
            "in_dtype": str(QUERY_START_LOC.dtype),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "mismatch": mismatch,
            "ref": ref.tolist(),
        }
        pt_stats = _bench(lambda: pytorch_ref(QUERY_START_LOC, SEQ_LENS))
        kern_stats = _bench(lambda: kernel_impl(QUERY_START_LOC, SEQ_LENS))
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _compute_combined_lens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- query_start_loc: {QUERY_START_LOC.cpu().tolist()}\n",
                f"- seq_lens: {SEQ_LENS.cpu().tolist()}\n",
                f"- TOP_K={TOP_K}, WINDOW={WINDOW_SIZE}, COMPRESS_RATIO={COMPRESS_RATIO}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- combined_lens shape: {stats['output_shape']} dtype {stats['out_dtype']}\n",
                f"- combined_lens: {stats['ref']}\n",
                f"- mismatches: {stats['mismatch']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
                    "Timing:\n",
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
                ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
