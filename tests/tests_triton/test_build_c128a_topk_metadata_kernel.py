"""
Standalone QAIC validation for `_build_c128a_topk_metadata_kernel`.

Source under test:
vllm/models/deepseek_v4/sparse_mla.py
  - _build_c128a_topk_metadata_kernel
  - build_c128a_topk_metadata  (launcher)

Builds C128A (compress-ratio) sparse top-k attention metadata in a single
kernel over all tokens (grid = (num_tokens,)). Both branches are validated.

Common:
  position       = positions[token_idx]
  num_compressed = min((position + 1) // compress_ratio, max_compressed_tokens)

Decode branch (token_idx < num_decode_tokens):
  is_valid_token = slot_mapping[token_idx] >= 0
  req_idx        = token_to_req_indices[token_idx]
  for offset in [0, max_compressed_tokens):
    is_valid  = offset < num_compressed
    block_num = block_table[req_idx, offset // block_size]
    slot      = block_num*block_size + offset % block_size  (offset<num_compressed)
    global_decode[token_idx, offset] = slot if is_valid else -1
  decode_lens[token_idx] = num_compressed if is_valid_token else 0

Prefill branch (token_idx >= num_decode_tokens):
  prefill_local[pfx, offset] = offset if offset < num_compressed else -1

NOTE: the kernel is generic in `compress_ratio`; for tractable tiny shapes we
use compress_ratio=2 (the C128A math -- block-table lookup vs. local index -- is
identical regardless of the exact ratio). All outputs are integer arrays
validated with EXACT equality.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.models.deepseek_v4.sparse_mla import build_c128a_topk_metadata

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_build_c128a_topk_metadata_kernel.txt")
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/sparse_mla.py"
KERNEL_NAME = "_build_c128a_topk_metadata_kernel"
DEVICE = "qaic"

# ----- Global shared inputs -----
torch.manual_seed(42)
COMPRESS_RATIO = 2
MAX_COMPRESSED = 8  # buffer row width AND kernel loop bound
BLOCK_SIZE = 4  # kernel block_size arg (storage_block_size // compress_ratio)
NUM_DECODE_TOKENS = 2
NUM_PREFILL_TOKENS = 2
NUM_TOKENS = NUM_DECODE_TOKENS + NUM_PREFILL_TOKENS
MAX_NUM_BLOCKS = 3

# positions -> num_compressed = (pos+1)//2 = [4, 3, 5, 2]
POSITIONS = torch.tensor([7, 5, 9, 3], dtype=torch.int64, device=DEVICE)
# decode token 0 valid, decode token 1 invalid (slot_mapping < 0)
SLOT_MAPPING = torch.tensor([0, -1, 5, 6], dtype=torch.int64, device=DEVICE)
TOKEN_TO_REQ = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
# block_table for the decode requests only (num_decodes == num_decode_tokens here)
BLOCK_TABLE = (
    torch.arange(
        NUM_DECODE_TOKENS * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE
    ).reshape(NUM_DECODE_TOKENS, MAX_NUM_BLOCKS)
    + 1
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


def pytorch_ref(positions, slot_mapping, token_to_req, block_table):
    positions = positions.cpu()
    slot_mapping = slot_mapping.cpu()
    token_to_req = token_to_req.cpu()
    block_table = block_table.cpu()

    global_decode = torch.zeros(NUM_DECODE_TOKENS, MAX_COMPRESSED, dtype=torch.int32)
    decode_lens = torch.zeros(NUM_DECODE_TOKENS, dtype=torch.int32)
    prefill_local = torch.zeros(NUM_PREFILL_TOKENS, MAX_COMPRESSED, dtype=torch.int32)

    for token_idx in range(NUM_TOKENS):
        position = int(positions[token_idx].item())
        num_compressed = min((position + 1) // COMPRESS_RATIO, MAX_COMPRESSED)
        if token_idx < NUM_DECODE_TOKENS:
            is_valid_token = int(slot_mapping[token_idx].item()) >= 0
            req_idx = int(token_to_req[token_idx].item())
            count = 0
            for offset in range(MAX_COMPRESSED):
                is_valid = offset < num_compressed
                if is_valid:
                    block_num = int(block_table[req_idx, offset // BLOCK_SIZE].item())
                    slot = block_num * BLOCK_SIZE + offset % BLOCK_SIZE
                    global_decode[token_idx, offset] = slot
                    count += 1
                else:
                    global_decode[token_idx, offset] = -1
            decode_lens[token_idx] = count if is_valid_token else 0
        else:
            pfx = token_idx - NUM_DECODE_TOKENS
            for offset in range(MAX_COMPRESSED):
                prefill_local[pfx, offset] = offset if offset < num_compressed else -1
    return global_decode, decode_lens, prefill_local


def kernel_impl(positions, slot_mapping, token_to_req, block_table):
    global_decode_buffer = torch.zeros(
        NUM_DECODE_TOKENS, MAX_COMPRESSED, dtype=torch.int32, device=DEVICE
    )
    decode_lens_buffer = torch.zeros(
        NUM_DECODE_TOKENS, dtype=torch.int32, device=DEVICE
    )
    prefill_buffer = torch.zeros(
        NUM_PREFILL_TOKENS, MAX_COMPRESSED, dtype=torch.int32, device=DEVICE
    )
    global_decode, decode_lens, prefill_local = build_c128a_topk_metadata(
        positions,
        COMPRESS_RATIO,
        NUM_DECODE_TOKENS,
        token_to_req,
        block_table,
        BLOCK_SIZE,
        slot_mapping,
        global_decode_buffer,
        decode_lens_buffer,
        prefill_buffer,
        max_compressed_tokens=MAX_COMPRESSED,
    )
    return global_decode, decode_lens, prefill_local


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
        ref = pytorch_ref(POSITIONS, SLOT_MAPPING, TOKEN_TO_REQ, BLOCK_TABLE)
        ker = kernel_impl(POSITIONS, SLOT_MAPPING, TOKEN_TO_REQ, BLOCK_TABLE)
        names = ["global_decode", "decode_lens", "prefill_local"]
        total_mism = 0
        max_diff = 0
        for n, r, k in zip(names, ref, ker):
            m, d = _exact(r, k)
            total_mism += m
            max_diff = max(max_diff, d)
        assert total_mism == 0, f"{total_mism} mismatched elements"

        stats = {
            "num_tokens": NUM_TOKENS,
            "num_decode_tokens": NUM_DECODE_TOKENS,
            "num_prefill_tokens": NUM_PREFILL_TOKENS,
            "compress_ratio": COMPRESS_RATIO,
            "max_compressed": MAX_COMPRESSED,
            "block_size": BLOCK_SIZE,
            "dtype": "torch.int32",
            "device": DEVICE,
            "mismatch_count": total_mism,
            "max_abs_int_diff": max_diff,
            "grid": f"({NUM_TOKENS},)",
        }

        pt_stats = _bench(
            lambda: pytorch_ref(POSITIONS, SLOT_MAPPING, TOKEN_TO_REQ, BLOCK_TABLE)
        )
        kern_stats = _bench(
            lambda: kernel_impl(POSITIONS, SLOT_MAPPING, TOKEN_TO_REQ, BLOCK_TABLE)
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
            _timing_keys = {
                "pytorch_latency_ms",
                "kernel_latency_ms",
                "speedup_kernel_over_pytorch",
            }
            for k, v in stats.items():
                if k in _timing_keys:
                    continue
                lines.append(f"- {k}: {v}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
