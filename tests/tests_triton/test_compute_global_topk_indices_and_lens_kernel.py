"""
Standalone QAIC validation for `_compute_global_topk_indices_and_lens_kernel`.

Source under test:
vllm/models/deepseek_v4/common/ops/cache_utils.py
  - _compute_global_topk_indices_and_lens_kernel
  - launcher: compute_global_topk_indices_and_lens(...)

Fuses a block-table lookup (local compressed index -> global KV-cache slot) with
valid-entry counting to build DENSE global top-k indices plus per-token lengths:

    for each entry: is_valid = local_idx >= 0
        block_number = block_table[req, local_idx // block_size]   (if valid)
        slot = block_number * block_size + local_idx % block_size  (if valid else -1)
    global_topk_indices[token] = slots (dense, same width as input, -1 where invalid)
    topk_lens[token] = where(is_valid_token, count(is_valid), 0)

Integer index kernel -> EXACT-equality comparison on BOTH outputs.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(
    LOG_DIR, "log_compute_global_topk_indices_and_lens_kernel.txt"
)
KERNEL_FILE_PATH = "vllm/models/deepseek_v4/common/ops/cache_utils.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_REQS = 2
NUM_TOKENS = 4
TOP_K = 8
BLOCK_SIZE = 4
MAX_BLOCKS = 6

TOKEN_TO_REQ = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=DEVICE)
IS_VALID_TOKEN = torch.tensor([1, 1, 1, 0], dtype=torch.int32, device=DEVICE)
BLOCK_TABLE = torch.arange(
    NUM_REQS * MAX_BLOCKS, dtype=torch.int32, device=DEVICE
).reshape(NUM_REQS, MAX_BLOCKS)


def _build_topk_row(n_valid, width):
    vals = torch.randint(0, MAX_BLOCKS * BLOCK_SIZE, (n_valid,), dtype=torch.int32)
    row = torch.full((width,), -1, dtype=torch.int32)
    row[:n_valid] = vals
    return row


TOPK_INDICES = torch.stack(
    [
        _build_topk_row(3, TOP_K),
        _build_topk_row(5, TOP_K),
        _build_topk_row(2, TOP_K),
        _build_topk_row(4, TOP_K),
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

    global_idx = torch.full((num_tokens, topk), -1, dtype=torch.int32)
    lens = torch.zeros(num_tokens, dtype=torch.int32)
    for t in range(num_tokens):
        req = int(token_to_req[t].item())
        cnt = 0
        for off in range(topk):
            local = int(topk_indices[t, off].item())
            if local >= 0:
                bn = int(block_table[req, local // block_size].item())
                global_idx[t, off] = bn * block_size + local % block_size
                cnt += 1
            else:
                global_idx[t, off] = -1
        lens[t] = cnt if int(is_valid_token[t].item()) else 0
    return global_idx, lens


def kernel_impl(topk_indices, token_to_req, block_table, block_size, is_valid_token):
    return compute_global_topk_indices_and_lens(
        topk_indices, token_to_req, block_table, block_size, is_valid_token
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_idx, ref_lens = pytorch_ref(
            TOPK_INDICES, TOKEN_TO_REQ, BLOCK_TABLE, BLOCK_SIZE, IS_VALID_TOKEN
        )
        k_idx, k_lens = kernel_impl(
            TOPK_INDICES, TOKEN_TO_REQ, BLOCK_TABLE, BLOCK_SIZE, IS_VALID_TOKEN
        )
        k_idx = k_idx.cpu()
        k_lens = k_lens.cpu()

        idx_mm = int((k_idx != ref_idx).sum().item())
        lens_mm = int((k_lens != ref_lens).sum().item())
        assert idx_mm == 0, f"global_topk_indices mismatch={idx_mm}"
        assert lens_mm == 0, f"topk_lens mismatch={lens_mm}"

        stats = {
            "input_shape": tuple(TOPK_INDICES.shape),
            "global_idx_shape": tuple(k_idx.shape),
            "lens_shape": tuple(k_lens.shape),
            "in_dtype": str(TOPK_INDICES.dtype),
            "out_dtype": str(k_idx.dtype),
            "device": DEVICE,
            "idx_mm": idx_mm,
            "lens_mm": lens_mm,
            "lens": k_lens.tolist(),
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
            "Kernel: _compute_global_topk_indices_and_lens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- topk_indices shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- block_size={BLOCK_SIZE}, num_reqs={NUM_REQS}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- global_topk_indices shape: {stats['global_idx_shape']} dtype {stats['out_dtype']}\n",
                f"- topk_lens shape: {stats['lens_shape']}, topk_lens={stats['lens']}\n",
                f"- global_topk_indices mismatches: {stats['idx_mm']}\n",
                f"- topk_lens mismatches: {stats['lens_mm']}\n",
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
