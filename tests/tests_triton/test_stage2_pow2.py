"""
Standalone QAIC validation for `_stage2_pow2`.

Source under test:
vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py
  - _stage2_pow2  (@triton.jit, defined inside _patch_make_bitmatrix_metadata)

`_stage2_pow2` is the power-of-2-safe replacement for triton_kernels'
`make_bitmatrix_metadata` stage-2 kernel. Given the per-token nonzero column
(expert) indices `NonzeroIndx` it builds the two index arrays that describe a
*stable sort of the nonzero entries by column (expert)*:
  - RowSortedIndx[orig_pos] = destination slot of that entry after the sort
  - ColSortedIndx[dest]     = original flat position feeding that slot
It works by packing (col << 16 | local_pos) keys, `tl.sort`-ing them to group by
column, running a keyed segmented `tl.associative_scan` (`_keyed_add`) to get
each entry's rank within its column, then adding the column base offset
(`ColOffs`) and the per-partial-block base (`ColPartialSum`). The "pow2" trick
uses BLOCK_SIZE_PADDED (next power of 2) for `tl.arange` while striding by the
true BLOCK_SIZE so non-power-of-2 top_k compiles.

RECONSTRUCTION / ASSUMPTIONS (documented):
  * `_stage2_pow2` is a nested closure and `_keyed_add` lives in the external
    `triton_kernels` package (not vendored here), so both are reconstructed in
    this file: the kernel body is copied verbatim from source, and `_keyed_add`
    is the canonical keyed-segmented-add combiner (key = high 16 bits, value =
    low 16 bits; add values iff keys match).
  * We use the SMALLEST FAITHFUL layout: a SINGLE partial block that covers all
    tokens (BLOCK_PER_TOK = n_tokens). With one block the per-block base
    `ColPartialSum` is all zeros and `ColOffs` is the exclusive prefix sum of
    per-column counts, so the produced permutation is exactly a stable
    argsort-by-column. All entries are valid (no -1 padding).

Reference: pure PyTorch stable argsort of the flattened expert ids by column;
its permutation and inverse are compared to ColSortedIndx / RowSortedIndx
exactly.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import tl, triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_stage2_pow2.txt")
KERNEL_FILE_PATH = (
    "vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
)

DEVICE = "qaic"
NUM_EXPERTS = 4  # n_cols
N_TOKENS = 8
TOKS_PER_ROW = 2  # top-k
BLOCK_PER_TOK = N_TOKENS  # single partial block covering all tokens
N_INDX = N_TOKENS * TOKS_PER_ROW  # 16
BLOCK_SIZE = BLOCK_PER_TOK * TOKS_PER_ROW  # 16
BLOCK_SIZE_PADDED = 1 << (max(BLOCK_SIZE, 1) - 1).bit_length()  # next pow2 (=16)

torch.manual_seed(42)
# Flattened per-token expert (column) ids, all valid, values in [0, NUM_EXPERTS).
NONZERO_INDX = torch.randint(
    0, NUM_EXPERTS, (N_INDX,), dtype=torch.int32, device=DEVICE
)


# --- Reconstructed keyed segmented-add combiner (from triton_kernels) ---------
@triton.jit
def _keyed_add(x, y):
    # High 16 bits are the sort key (column); low 16 bits are the running value.
    kv_mask = 0xFFFF0000
    x_key = x & kv_mask
    y_key = y & kv_mask
    y_val = y & 0x0000FFFF
    x_val = x & 0x0000FFFF
    same = x_key == y_key
    val = tl.where(same, (x_val + y_val) & 0xFFFF, y_val)
    return y_key | val


# --- Reconstructed kernel body (copied verbatim from source) ------------------
@triton.jit
def _stage2_pow2(
    ColSortedIndx,
    RowSortedIndx,
    NonzeroIndx,
    n_tokens,
    ColPartialSum,
    stride_pm,
    stride_pn,
    ColOffs,
    TOKS_PER_ROW: tl.constexpr,
    BLOCK_PER_TOK: tl.constexpr,
    BLOCK_SIZE_PADDED: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
    tl.static_assert(BLOCK_SIZE_PADDED <= 32768)
    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    nonzero_indx_size = n_tokens * TOKS_PER_ROW
    pid_m = tl.program_id(0)
    offs_local = tl.arange(0, BLOCK_SIZE_PADDED)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < nonzero_indx_size
    col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)
    kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    col_indx = kv_pairs >> 16
    offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xFFFF)
    mask = col_indx != 0xFFFF
    x = kv_pairs & 0xFFFF0000 | 0x00000001
    cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xFFFF
    row_sorted_indx = tl.load(
        ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask
    )
    row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
    row_sorted_indx += exclusive_run_lengths
    tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
    tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)


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


def _col_offs(nonzero_indx):
    counts = torch.bincount(nonzero_indx.cpu().to(torch.int64), minlength=NUM_EXPERTS)
    return (torch.cumsum(counts, 0) - counts).to(torch.int32)


def pytorch_ref(nonzero_indx):
    """Pure PyTorch stable argsort-by-column -> (col_sorted, row_sorted)."""
    flat = nonzero_indx.cpu().to(torch.int64)
    # Stable sort of positions by column id (ties keep original position order).
    order = torch.argsort(flat, stable=True).to(torch.int32)  # dest -> orig_pos
    col_sorted = order
    # row_sorted is the inverse permutation: orig_pos -> dest slot.
    row_sorted = torch.empty(N_INDX, dtype=torch.int32)
    row_sorted[order.to(torch.int64)] = torch.arange(N_INDX, dtype=torch.int32)
    return col_sorted, row_sorted


def kernel_impl(nonzero_indx):
    col_offs = _col_offs(nonzero_indx).to(nonzero_indx.device)
    # Single partial block -> per-block base offsets are all zero.
    col_partial_sum = torch.zeros(
        1, NUM_EXPERTS, dtype=torch.int32, device=nonzero_indx.device
    )
    col_sorted = torch.full(
        (N_INDX,), -1, dtype=torch.int32, device=nonzero_indx.device
    )
    row_sorted = torch.full(
        (N_INDX,), -1, dtype=torch.int32, device=nonzero_indx.device
    )
    grid = (triton.cdiv(N_TOKENS, BLOCK_PER_TOK),)  # = (1,)
    _stage2_pow2[grid](
        col_sorted,
        row_sorted,
        nonzero_indx,
        N_TOKENS,
        col_partial_sum,
        col_partial_sum.stride(0),
        col_partial_sum.stride(1),
        col_offs,
        TOKS_PER_ROW=TOKS_PER_ROW,
        BLOCK_PER_TOK=BLOCK_PER_TOK,
        BLOCK_SIZE_PADDED=BLOCK_SIZE_PADDED,
    )
    return col_sorted, row_sorted


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_col, ref_row = pytorch_ref(NONZERO_INDX)
        k_col, k_row = kernel_impl(NONZERO_INDX)

        k_col_c, k_row_c = k_col.cpu(), k_row.cpu()

        row_ok = bool(torch.equal(k_row_c, ref_row))
        col_ok = bool(torch.equal(k_col_c, ref_col))
        assert row_ok, "RowSortedIndx mismatch"
        assert col_ok, "ColSortedIndx mismatch"
        stats = {
            "input_shape": tuple(NONZERO_INDX.shape),
            "col_sorted_shape": tuple(k_col.shape),
            "row_sorted_shape": tuple(k_row.shape),
            "dtype": str(NONZERO_INDX.dtype),
            "device": str(NONZERO_INDX.device),
            "row_exact_match": row_ok,
            "col_exact_match": col_ok,
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
        }

        pt_stats = _bench(lambda: pytorch_ref(NONZERO_INDX))
        kern_stats = _bench(lambda: kernel_impl(NONZERO_INDX))
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
            "Kernel: _stage2_pow2 (reconstructed; nested closure in source)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- nonzero_indx shape: {stats['input_shape']}\n",
                f"- n_tokens={N_TOKENS}, toks_per_row={TOKS_PER_ROW}, "
                f"n_experts={NUM_EXPERTS}\n",
                f"- block_size={BLOCK_SIZE}, block_size_padded={BLOCK_SIZE_PADDED}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- col_sorted shape: {stats['col_sorted_shape']}\n",
                f"- row_sorted shape: {stats['row_sorted_shape']}\n",
                f"- row_exact_match: {stats['row_exact_match']}\n",
                f"- col_exact_match: {stats['col_exact_match']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
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
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
