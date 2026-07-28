"""
Standalone QAIC validation for `_build_flashinfer_mixed_sparse_indices_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/cache_utils.py
  - _build_flashinfer_mixed_sparse_indices_kernel
  - launcher: build_flashinfer_mixed_sparse_indices(...)

Builds the FlashInfer DeepSeek-V4 sparse-index matrix for decode-first batches.
Output `sparse_indices` has shape [num_tokens, window_size + padded_topk]:
the first `window_size` columns are SWA slot ids, the remaining columns are
compressed / top-k slot ids. `sparse_topk_lens` is the active length per token.

  * Decode tokens (token_idx < NUM_DECODE_TOKENS):
      SWA cols       = decode_swa_indices[token]         (verbatim)
      compressed cols= decode_compressed_indices[token]  (verbatim; are_local=False)
      len            = window_size + decode_compressed_topk_lens[token]
  * Prefill tokens (token_idx >= NUM_DECODE_TOKENS), prefill_idx = token - NUM_DECODE:
      req = token_to_req_indices[token]; pos derived from query_start_loc/seq_lens
      swa_len  = min(pos+1, window_size); swa_start = pos - swa_len + 1
      SWA cols = swa_block_table[req, (swa_start+off)//swa_bs]*swa_bs
                 + (swa_start+off)%swa_bs   (off < swa_len else -1)
      topk_len = min((pos+1)//compress_ratio, topk)
      compressed cols = compressed_block_table[req, local//c_bs]*c_bs + local%c_bs
                        (off < topk_len & local >= 0 else -1)
      len = window_size + topk_len

CONFIG / simplifications documented here:
  - decode_compressed_indices_are_local = False (no per-decode block-table
    translation; decode compressed indices are copied verbatim). This is the
    common decode-first path; the local-translation branch is not exercised.
  - decode_compressed_topk_lens supplied (HAS_DECODE_COMPRESSED_LENS=True).
  - decode_compressed_topk == topk and prefill_topk_stride == topk, so
    padded_topk == topk (already a multiple of 4) and every column is written.
  - Batch: 2 single-token decode reqs + 1 prefill req of 3 tokens; global token
    indices 0,1 are decode, 2,3,4 are prefill.

Integer index kernel -> EXACT-equality on BOTH outputs.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    build_flashinfer_mixed_sparse_indices,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_build_flashinfer_mixed_sparse_indices_kernel.txt"
)
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/cache_utils.py"

DEVICE = "qaic"
torch.manual_seed(42)

WINDOW_SIZE = 4
COMPRESS_RATIO = 4
TOP_K = 8
SWA_BLOCK_SIZE = 4
COMPRESSED_BLOCK_SIZE = 4
MAX_BLOCKS = 4
NUM_REQS = 3

NUM_DECODE_TOKENS = 2
NUM_PREFILL_TOKENS = 3
NUM_TOKENS = NUM_DECODE_TOKENS + NUM_PREFILL_TOKENS
PADDED_TOP_K = ((max(TOP_K, TOP_K) + 3) // 4) * 4  # == 8

# req0: token0 (decode), req1: token1 (decode), req2: tokens 2,3,4 (prefill)
QUERY_START_LOC = torch.tensor([0, 1, 2, 5], dtype=torch.int32, device=DEVICE)
TOKEN_TO_REQ = torch.tensor([0, 1, 2, 2, 2], dtype=torch.int32, device=DEVICE)
SEQ_LENS = torch.tensor([6, 9, 5], dtype=torch.int32, device=DEVICE)

DECODE_SWA_INDICES = torch.randint(
    0, MAX_BLOCKS * SWA_BLOCK_SIZE, (NUM_DECODE_TOKENS, WINDOW_SIZE),
    dtype=torch.int32, device=DEVICE,
)
DECODE_COMPRESSED_INDICES = torch.randint(
    0, MAX_BLOCKS * COMPRESSED_BLOCK_SIZE, (NUM_DECODE_TOKENS, TOP_K),
    dtype=torch.int32, device=DEVICE,
)
DECODE_COMPRESSED_TOPK_LENS = torch.tensor(
    [5, 3], dtype=torch.int32, device=DEVICE
)
PREFILL_TOPK_INDICES = torch.randint(
    0, MAX_BLOCKS * COMPRESSED_BLOCK_SIZE, (NUM_PREFILL_TOKENS, TOP_K),
    dtype=torch.int32, device=DEVICE,
)
SWA_BLOCK_TABLE = torch.arange(
    NUM_REQS * MAX_BLOCKS, dtype=torch.int32, device=DEVICE
).reshape(NUM_REQS, MAX_BLOCKS)
COMPRESSED_BLOCK_TABLE = (
    SWA_BLOCK_TABLE + 100
).to(torch.int32)


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


def pytorch_ref():
    swa = DECODE_SWA_INDICES.cpu()
    dci = DECODE_COMPRESSED_INDICES.cpu()
    dcl = DECODE_COMPRESSED_TOPK_LENS.cpu()
    ptk = PREFILL_TOPK_INDICES.cpu()
    qsl = QUERY_START_LOC.cpu()
    seq_lens = SEQ_LENS.cpu()
    t2r = TOKEN_TO_REQ.cpu()
    sbt = SWA_BLOCK_TABLE.cpu()
    cbt = COMPRESSED_BLOCK_TABLE.cpu()

    ncols = WINDOW_SIZE + PADDED_TOP_K
    sparse = torch.full((NUM_TOKENS, ncols), -1, dtype=torch.int32)
    lens = torch.zeros(NUM_TOKENS, dtype=torch.int32)

    for token_idx in range(NUM_TOKENS):
        if token_idx < NUM_DECODE_TOKENS:
            # SWA verbatim
            for off in range(WINDOW_SIZE):
                sparse[token_idx, off] = int(swa[token_idx, off].item())
            # compressed verbatim (are_local=False)
            for off in range(PADDED_TOP_K):
                v = int(dci[token_idx, off].item()) if off < TOP_K else -1
                sparse[token_idx, WINDOW_SIZE + off] = v
            lens[token_idx] = WINDOW_SIZE + int(dcl[token_idx].item())
            continue

        prefill_idx = token_idx - NUM_DECODE_TOKENS
        req = int(t2r[token_idx].item())
        query_start = int(qsl[req].item())
        query_end = int(qsl[req + 1].item())
        query_len = query_end - query_start
        seq_len = int(seq_lens[req].item())
        start_pos = seq_len - query_len
        pos = start_pos + (token_idx - query_start)
        swa_len = min(pos + 1, WINDOW_SIZE)
        swa_start_pos = pos - swa_len + 1
        topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)

        for off in range(WINDOW_SIZE):
            if off < swa_len:
                pos_off = swa_start_pos + off
                bn = int(sbt[req, pos_off // SWA_BLOCK_SIZE].item())
                sparse[token_idx, off] = bn * SWA_BLOCK_SIZE + pos_off % SWA_BLOCK_SIZE
            else:
                sparse[token_idx, off] = -1

        for off in range(PADDED_TOP_K):
            local = int(ptk[prefill_idx, off].item()) if off < TOP_K else -1
            is_valid = local >= 0
            if (off < topk_len) and is_valid:
                bn = int(cbt[req, local // COMPRESSED_BLOCK_SIZE].item())
                slot = bn * COMPRESSED_BLOCK_SIZE + local % COMPRESSED_BLOCK_SIZE
            else:
                slot = -1
            sparse[token_idx, WINDOW_SIZE + off] = slot

        lens[token_idx] = WINDOW_SIZE + topk_len
    return sparse, lens


def kernel_impl():
    return build_flashinfer_mixed_sparse_indices(
        DECODE_SWA_INDICES,
        DECODE_COMPRESSED_INDICES,
        DECODE_COMPRESSED_TOPK_LENS,
        PREFILL_TOPK_INDICES,
        QUERY_START_LOC,
        SEQ_LENS,
        TOKEN_TO_REQ,
        SWA_BLOCK_TABLE,
        SWA_BLOCK_SIZE,
        COMPRESSED_BLOCK_TABLE,
        COMPRESSED_BLOCK_SIZE,
        WINDOW_SIZE,
        COMPRESS_RATIO,
        TOP_K,
        decode_compressed_indices_are_local=False,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_idx, ref_lens = pytorch_ref()
        k_idx, k_lens = kernel_impl()
        k_idx = k_idx.cpu()
        k_lens = k_lens.cpu()

        idx_mm = int((k_idx != ref_idx).sum().item())
        lens_mm = int((k_lens != ref_lens).sum().item())
        assert idx_mm == 0, f"sparse_indices mismatch={idx_mm}"
        assert lens_mm == 0, f"sparse_topk_lens mismatch={lens_mm}"

        stats = {
            "sparse_indices_shape": tuple(k_idx.shape),
            "sparse_lens_shape": tuple(k_lens.shape),
            "out_dtype": str(k_idx.dtype),
            "device": DEVICE,
            "idx_mm": idx_mm,
            "lens_mm": lens_mm,
            "lens": k_lens.tolist(),
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
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
            "Kernel: _build_flashinfer_mixed_sparse_indices_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- num_decode_tokens={NUM_DECODE_TOKENS}, num_prefill_tokens={NUM_PREFILL_TOKENS}\n",
                f"- window_size={WINDOW_SIZE}, topk={TOP_K}, padded_topk={PADDED_TOP_K}, "
                f"compress_ratio={COMPRESS_RATIO}\n",
                f"- decode_compressed_indices_are_local=False\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- sparse_indices shape: {stats['sparse_indices_shape']} dtype {stats['out_dtype']}\n",
                f"- sparse_topk_lens shape: {stats['sparse_lens_shape']}, lens={stats['lens']}\n",
                f"- sparse_indices mismatches: {stats['idx_mm']}\n",
                f"- sparse_topk_lens mismatches: {stats['lens_mm']}\n",
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
