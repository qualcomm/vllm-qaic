"""
Standalone QAIC validation for `_combine_topk_swa_indices_kernel` (ROCm version).

Source under test:
vllm/models/deepseek_v4/amd/rocm.py
  - _combine_topk_swa_indices_kernel
  - launcher: combine_topk_swa_indices(...)

The kernel combines the compressed sparse-attention top-k indices with the
sliding-window (SWA) window indices into a single padded per-token index array
(`combined_indices`, filled with -1 elsewhere) and writes the per-token active
length (`combined_lens = topk_len + swa_len`).

Per query token at absolute position `pos`:
  topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
  swa_len  = min(pos + 1, WINDOW_SIZE)
  combined[:, 0:topk_len]            = valid(topk_idx) ? topk_idx + M*batch : -1
  combined[:, topk_len:topk_len+swa] = M*batch + N + off + pos - swa_len + 1 - gather_start
where a top-k index is valid iff 0 <= idx < N.

Integer index metadata kernel -> EXACT-equality comparison on both outputs.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.amd.rocm import combine_topk_swa_indices

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_combine_topk_swa_indices_kernel_rocm.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/amd/rocm.py"

DEVICE = "qaic"
torch.manual_seed(42)

# Config (tiny, faithful).
COMPRESS_RATIO = 4
WINDOW_SIZE = 8
TOP_K = 8
TOPK_WIDTH = 8
NUM_REQS = 2
M = 64
N = 32
_ALIGN = 128
COMBINED_TOPK = (TOP_K + WINDOW_SIZE + _ALIGN - 1) // _ALIGN * _ALIGN

QUERY_START_LOC = torch.tensor([0, 3, 6], dtype=torch.int32, device=DEVICE)
NUM_TOKENS = int(QUERY_START_LOC[-1].item())
SEQ_LENS = torch.tensor([10, 12], dtype=torch.int32, device=DEVICE)
GATHER_LENS = torch.tensor([8, 8], dtype=torch.int32, device=DEVICE)
# Mix of valid (0..N-1), out-of-range (>=N), and -1 sentinels.
TOPK_INDICES = torch.randint(
    -1, N + 4, (NUM_TOKENS, TOPK_WIDTH), dtype=torch.int32, device=DEVICE
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


def pytorch_ref(topk_indices, query_start_loc, seq_lens, gather_lens):
    topk_indices = topk_indices.cpu()
    qsl = query_start_loc.cpu()
    seq_lens = seq_lens.cpu()
    gather_lens = gather_lens.cpu()

    combined = torch.full(
        (NUM_TOKENS, COMBINED_TOPK), -1, dtype=torch.int32
    )
    lens = torch.zeros(NUM_TOKENS, dtype=torch.int32)

    base = int(qsl[0].item())
    for b in range(NUM_REQS):
        qs = int(qsl[b].item()) - base
        qe = int(qsl[b + 1].item()) - base
        ql = qe - qs
        sl = int(seq_lens[b].item())
        gl = int(gather_lens[b].item())
        start_pos = sl - ql
        gather_start = sl - gl
        for tok in range(qs, qe):
            pos = start_pos + (tok - qs)
            topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
            swa_len = min(pos + 1, WINDOW_SIZE)
            for off in range(topk_len):
                idx = int(topk_indices[tok, off].item()) if off < TOPK_WIDTH else -1
                valid = (idx >= 0) and (idx < N)
                combined[tok, off] = (idx + M * b) if valid else -1
            for so in range(swa_len):
                combined[tok, topk_len + so] = (
                    M * b + N + so + pos - swa_len + 1 - gather_start
                )
            lens[tok] = topk_len + swa_len
    return combined, lens


def kernel_impl(topk_indices, query_start_loc, seq_lens, gather_lens):
    return combine_topk_swa_indices(
        topk_indices,
        query_start_loc,
        seq_lens,
        gather_lens,
        WINDOW_SIZE,
        COMPRESS_RATIO,
        TOP_K,
        M,
        N,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_idx, ref_lens = pytorch_ref(
            TOPK_INDICES, QUERY_START_LOC, SEQ_LENS, GATHER_LENS
        )
        k_idx, k_lens = kernel_impl(
            TOPK_INDICES, QUERY_START_LOC, SEQ_LENS, GATHER_LENS
        )
        k_idx = k_idx.cpu()
        k_lens = k_lens.cpu()

        idx_mismatch = int((k_idx != ref_idx).sum().item())
        lens_mismatch = int((k_lens != ref_lens).sum().item())
        assert idx_mismatch == 0, f"combined_indices mismatches={idx_mismatch}"
        assert lens_mismatch == 0, f"combined_lens mismatches={lens_mismatch}"

        stats = {
            "input_shape": tuple(TOPK_INDICES.shape),
            "combined_indices_shape": tuple(k_idx.shape),
            "combined_lens_shape": tuple(k_lens.shape),
            "in_dtype": str(TOPK_INDICES.dtype),
            "out_dtype": str(k_idx.dtype),
            "device": DEVICE,
            "idx_mismatch": idx_mismatch,
            "lens_mismatch": lens_mismatch,
        }
        pt_stats = _bench(
            lambda: pytorch_ref(
                TOPK_INDICES, QUERY_START_LOC, SEQ_LENS, GATHER_LENS
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                TOPK_INDICES, QUERY_START_LOC, SEQ_LENS, GATHER_LENS
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
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _combine_topk_swa_indices_kernel (ROCm)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- topk_indices shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- num_reqs: {NUM_REQS}, TOP_K={TOP_K}, WINDOW={WINDOW_SIZE}, "
                f"COMPRESS_RATIO={COMPRESS_RATIO}, M={M}, N={N}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- combined_indices shape: {stats['combined_indices_shape']} "
                f"dtype {stats['out_dtype']}\n",
                f"- combined_lens shape: {stats['combined_lens_shape']}\n",
                f"- combined_indices mismatches: {stats['idx_mismatch']}\n",
                f"- combined_lens mismatches: {stats['lens_mismatch']}\n",
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
