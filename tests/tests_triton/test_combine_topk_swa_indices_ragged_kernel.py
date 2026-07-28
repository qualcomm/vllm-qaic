"""
Standalone QAIC validation for `_combine_topk_swa_indices_ragged_kernel`.

Source under test:
vllm/models/deepseek_v4/amd/rocm.py
  - _combine_topk_swa_indices_ragged_kernel
  - launcher: combine_topk_swa_indices_ragged(...)
    (also runs _compute_combined_lens_kernel + _build_indptr_from_lengths)

Ragged (variable-length, indptr-addressed) version of combining the compressed
top-k indices with the SWA window indices. Per query token at absolute position
`pos`, with out_start = combined_indptr[token]:

    topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)
    swa_len  = min(pos + 1, WINDOW_SIZE)
    ragged[out_start + off]            = topk_idx[off] + M*batch     (off < topk_len)
    ragged[out_start + topk_len + so]  = M*batch + N + so + pos - swa_len + 1
                                         - gather_start              (so < swa_len)

NOTE: unlike the dense ROCm combine kernel, the ragged kernel does NOT validate
top-k indices against [0, N); it packs them verbatim (offset-bounded by
topk_width). We therefore use non-negative in-range indices and topk_width==TOP_K
so every top-k slot is written.

Integer index kernel -> EXACT-equality on the packed ragged region, indptr, and
combined_lens.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.amd.rocm import combine_topk_swa_indices_ragged

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_combine_topk_swa_indices_ragged_kernel.txt"
)
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/amd/rocm.py"

DEVICE = "qaic"
torch.manual_seed(42)

COMPRESS_RATIO = 4
WINDOW_SIZE = 8
TOP_K = 8
TOPK_WIDTH = TOP_K
NUM_REQS = 2
M = 64
N = 32

QUERY_START_LOC = torch.tensor([0, 3, 6], dtype=torch.int32, device=DEVICE)
NUM_TOKENS = int(QUERY_START_LOC[-1].item())
SEQ_LENS = torch.tensor([10, 12], dtype=torch.int32, device=DEVICE)
GATHER_LENS = torch.tensor([8, 8], dtype=torch.int32, device=DEVICE)
TOPK_INDICES = torch.randint(
    0, N, (NUM_TOKENS, TOPK_WIDTH), dtype=torch.int32, device=DEVICE
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

    lens = torch.zeros(NUM_TOKENS, dtype=torch.int32)
    base = int(qsl[0].item())
    metas = []
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
            lens[tok] = topk_len + swa_len
            metas.append((tok, b, pos, topk_len, swa_len, gather_start))

    indptr = torch.zeros(NUM_TOKENS + 1, dtype=torch.int32)
    torch.cumsum(lens, dim=0, out=indptr[1:])
    total = int(indptr[-1].item())
    ragged = torch.empty(total, dtype=torch.int32)
    for tok, b, pos, topk_len, swa_len, gather_start in metas:
        out_start = int(indptr[tok].item())
        for off in range(topk_len):
            ragged[out_start + off] = int(topk_indices[tok, off].item()) + M * b
        for so in range(swa_len):
            ragged[out_start + topk_len + so] = (
                M * b + N + so + pos - swa_len + 1 - gather_start
            )
    return ragged, indptr, lens


def kernel_impl(topk_indices, query_start_loc, seq_lens, gather_lens):
    ragged, indptr, lens = combine_topk_swa_indices_ragged(
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
    total = int(indptr[-1].item())
    return ragged[:total], indptr, lens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ragged, ref_indptr, ref_lens = pytorch_ref(
            TOPK_INDICES, QUERY_START_LOC, SEQ_LENS, GATHER_LENS
        )
        k_ragged, k_indptr, k_lens = kernel_impl(
            TOPK_INDICES, QUERY_START_LOC, SEQ_LENS, GATHER_LENS
        )
        k_ragged = k_ragged.cpu()
        k_indptr = k_indptr.cpu()
        k_lens = k_lens.cpu()

        lens_mm = int((k_lens != ref_lens).sum().item())
        indptr_mm = int((k_indptr != ref_indptr).sum().item())
        ragged_mm = int((k_ragged != ref_ragged).sum().item())
        assert lens_mm == 0, f"combined_lens mismatch={lens_mm}"
        assert indptr_mm == 0, f"indptr mismatch={indptr_mm}"
        assert ragged_mm == 0, f"ragged mismatch={ragged_mm}"

        stats = {
            "input_shape": tuple(TOPK_INDICES.shape),
            "ragged_shape": tuple(k_ragged.shape),
            "indptr_shape": tuple(k_indptr.shape),
            "in_dtype": str(TOPK_INDICES.dtype),
            "out_dtype": str(k_ragged.dtype),
            "device": DEVICE,
            "lens_mm": lens_mm,
            "indptr_mm": indptr_mm,
            "ragged_mm": ragged_mm,
            "indptr": k_indptr.tolist(),
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
            "Kernel: _combine_topk_swa_indices_ragged_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- topk_indices shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- TOP_K={TOP_K}, WINDOW={WINDOW_SIZE}, COMPRESS_RATIO={COMPRESS_RATIO}, M={M}, N={N}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- combined_ragged shape: {stats['ragged_shape']} dtype {stats['out_dtype']}\n",
                f"- indptr={stats['indptr']}\n",
                f"- combined_lens mismatches: {stats['lens_mm']}\n",
                f"- indptr mismatches: {stats['indptr_mm']}\n",
                f"- ragged mismatches: {stats['ragged_mm']}\n",
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
