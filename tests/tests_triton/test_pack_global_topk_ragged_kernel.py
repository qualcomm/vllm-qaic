"""
Standalone QAIC validation for `_pack_global_topk_ragged_kernel`.

Source under test:
vllm/models/deepseek_v4/amd/rocm.py
  - _pack_global_topk_ragged_kernel
  - launcher: compute_global_topk_ragged_indices_and_indptr(...)
    (also runs _compute_topk_lens_kernel + _build_indptr_from_lengths)

Translates local compressed top-k indices into global paged KV-cache slot ids
via a per-request block-table lookup, packing the results into a ragged
(indptr-addressed) buffer:

    topk_len   = (idx >= 0).sum(-1)                (per token; 0 if pad token)
    indptr     = cumsum(topk_len)
    for offset in [0, topk_len):
        local = topk_indices[token, offset]
        slot  = block_table[req, local // block_size] * block_size
                + local % block_size            (if local >= 0 else -1)
        global_topk_ragged[indptr[token] + offset] = slot

Integer index kernel -> EXACT-equality comparison on the packed ragged region,
the indptr, and topk_lens. Valid indices are placed contiguously at the front
of each row (matching the indexer's -1-padded output), so `topk_len` equals the
leading valid count.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.amd.rocm import (
    compute_global_topk_ragged_indices_and_indptr,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_pack_global_topk_ragged_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/amd/rocm.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_REQS = 2
NUM_TOKENS = 4
TOP_K = 8
BLOCK_SIZE = 4
MAX_BLOCKS = 6

# token -> req mapping
TOKEN_TO_REQ = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=DEVICE)
IS_VALID_TOKEN = torch.tensor([1, 1, 1, 0], dtype=torch.int32, device=DEVICE)
# block table: physical block numbers per (req, logical block)
BLOCK_TABLE = torch.arange(
    NUM_REQS * MAX_BLOCKS, dtype=torch.int32, device=DEVICE
).reshape(NUM_REQS, MAX_BLOCKS)


def _build_topk_row(n_valid, width):
    # valid local indices packed at front, -1 padding after.
    vals = torch.randint(0, MAX_BLOCKS * BLOCK_SIZE, (n_valid,), dtype=torch.int32)
    row = torch.full((width,), -1, dtype=torch.int32)
    row[:n_valid] = vals
    return row


TOPK_INDICES = torch.stack(
    [
        _build_topk_row(3, TOP_K),
        _build_topk_row(5, TOP_K),
        _build_topk_row(2, TOP_K),
        _build_topk_row(4, TOP_K),  # this token is a pad token (is_valid=0)
    ]
).to(DEVICE)


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


def pytorch_ref(topk_indices, token_to_req, block_table, block_size, is_valid_token):
    topk_indices = topk_indices.cpu()
    token_to_req = token_to_req.cpu()
    block_table = block_table.cpu()
    is_valid_token = is_valid_token.cpu()
    num_tokens, topk = topk_indices.shape

    topk_lens = torch.zeros(num_tokens, dtype=torch.int32)
    for t in range(num_tokens):
        cnt = int((topk_indices[t] >= 0).sum().item())
        topk_lens[t] = cnt if int(is_valid_token[t].item()) else 0

    indptr = torch.zeros(num_tokens + 1, dtype=torch.int32)
    torch.cumsum(topk_lens, dim=0, out=indptr[1:])

    total = int(indptr[-1].item())
    ragged = torch.empty(total, dtype=torch.int32)
    for t in range(num_tokens):
        out_start = int(indptr[t].item())
        out_len = int((indptr[t + 1] - indptr[t]).item())
        req = int(token_to_req[t].item())
        for off in range(out_len):
            local = int(topk_indices[t, off].item())
            if local >= 0:
                bn = int(block_table[req, local // block_size].item())
                slot = bn * block_size + local % block_size
            else:
                slot = -1
            ragged[out_start + off] = slot
    return ragged, indptr, topk_lens


def kernel_impl(topk_indices, token_to_req, block_table, block_size, is_valid_token):
    ragged, indptr, lens = compute_global_topk_ragged_indices_and_indptr(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
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
            TOPK_INDICES, TOKEN_TO_REQ, BLOCK_TABLE, BLOCK_SIZE, IS_VALID_TOKEN
        )
        k_ragged, k_indptr, k_lens = kernel_impl(
            TOPK_INDICES, TOKEN_TO_REQ, BLOCK_TABLE, BLOCK_SIZE, IS_VALID_TOKEN
        )
        k_ragged = k_ragged.cpu()
        k_indptr = k_indptr.cpu()
        k_lens = k_lens.cpu()

        lens_mm = int((k_lens != ref_lens).sum().item())
        indptr_mm = int((k_indptr != ref_indptr).sum().item())
        ragged_mm = int((k_ragged != ref_ragged).sum().item())
        assert lens_mm == 0, f"topk_lens mismatch={lens_mm}"
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
                TOPK_INDICES, TOKEN_TO_REQ, BLOCK_TABLE, BLOCK_SIZE, IS_VALID_TOKEN
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                TOPK_INDICES, TOKEN_TO_REQ, BLOCK_TABLE, BLOCK_SIZE, IS_VALID_TOKEN
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
            "Kernel: _pack_global_topk_ragged_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- topk_indices shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- block_size={BLOCK_SIZE}, num_reqs={NUM_REQS}, block_table {tuple(BLOCK_TABLE.shape)}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- global_topk_ragged shape: {stats['ragged_shape']} dtype {stats['out_dtype']}\n",
                f"- indptr shape: {stats['indptr_shape']}, indptr={stats['indptr']}\n",
                f"- topk_lens mismatches: {stats['lens_mm']}\n",
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
