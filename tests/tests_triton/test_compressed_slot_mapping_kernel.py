"""
Standalone QAIC validation for `_compressed_slot_mapping_kernel`.

Source under test:
vllm/v1/attention/backends/mla/compressor_utils.py
  - _compressed_slot_mapping_kernel
  - launcher: get_compressed_slot_mapping(num_tokens, query_start_loc,
        seq_lens, block_table, block_size, compress_ratio, out=None)

Computes the paged-KV slot index for each query token AFTER compression. For
each request b, with query positions pos = (seq_len[b] - query_len) .. seq_len[b]-1:
    is_valid          = (pos + 1) % compress_ratio == 0
    pos_after_compress = pos // compress_ratio
    block_id          = pos_after_compress // block_size
    slot              = block_table[b, block_id] * block_size
                        + pos_after_compress % block_size
    slot_mapping[q]   = slot if is_valid else -1   (PAD_ID)
Positions that are not compression boundaries are left at the -1 pre-fill.

Config tested: 3 requests, block_size=16, compress_ratio=4.
INTEGER/index output -> compared EXACTLY with torch.equal.
Reference: pure PyTorch replicating the per-token index math above.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compressed_slot_mapping_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/backends/mla/compressor_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.backends.mla.compressor_utils import (  # noqa: E402
    get_compressed_slot_mapping,
)

torch.manual_seed(42)

BLOCK_SIZE = 16
COMPRESS_RATIO = 4
MAX_NUM_BLOCKS = 8

# ---- Global shared inputs (used by BOTH implementations) ----
# 3 requests with query lengths 6, 4, 5 -> num_tokens = 15
QUERY_LENS = [6, 4, 5]
SEQ_LENS_LIST = [20, 9, 13]
NUM_REQS = len(QUERY_LENS)
NUM_TOKENS = sum(QUERY_LENS)

_qsl = [0]
for q in QUERY_LENS:
    _qsl.append(_qsl[-1] + q)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)
SEQ_LENS = torch.tensor(SEQ_LENS_LIST, dtype=torch.int32, device=DEVICE)
# Distinct physical block numbers per request.
BLOCK_TABLE = (
    torch.arange(NUM_REQS * MAX_NUM_BLOCKS, dtype=torch.int32, device=DEVICE) + 3
).reshape(NUM_REQS, MAX_NUM_BLOCKS)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
    """Device-synced wall-clock benchmark. Returns latency stats (ms)."""
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
    arr = np.array(times)
    return {
        "avg_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def pytorch_ref(query_start_loc, seq_lens, block_table):
    """Pure PyTorch replica of the compressed slot-mapping index math."""
    qsl = query_start_loc.cpu()
    sl = seq_lens.cpu()
    bt = block_table.cpu()
    slot_mapping = torch.full((NUM_TOKENS,), -1, dtype=torch.int64)
    for b in range(NUM_REQS):
        q_start = int(qsl[b].item())
        q_end = int(qsl[b + 1].item())
        query_len = q_end - q_start
        start_pos = int(sl[b].item()) - query_len
        for i in range(query_len):
            pos = start_pos + i
            if (pos + 1) % COMPRESS_RATIO != 0:
                continue
            pos_after = pos // COMPRESS_RATIO
            block_id = pos_after // BLOCK_SIZE
            slot = (
                int(bt[b, block_id].item()) * BLOCK_SIZE
                + pos_after % BLOCK_SIZE
            )
            slot_mapping[q_start + i] = slot
    return slot_mapping


def kernel_impl(query_start_loc, seq_lens, block_table):
    """Kernel launch only."""
    return get_compressed_slot_mapping(
        NUM_TOKENS,
        query_start_loc,
        seq_lens,
        block_table,
        BLOCK_SIZE,
        COMPRESS_RATIO,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(QUERY_START_LOC, SEQ_LENS, BLOCK_TABLE)
        kernel_out = kernel_impl(QUERY_START_LOC, SEQ_LENS, BLOCK_TABLE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), (
            f"index mismatch\nref={ref_cpu.tolist()}\nkern={kernel_cpu.tolist()}"
        )

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(BLOCK_TABLE.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(BLOCK_TABLE.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(BLOCK_TABLE.device),
            "max_abs_diff": int(diff.max().item()),
            "mean_abs_diff": float(diff.float().mean().item()),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(QUERY_START_LOC, SEQ_LENS, BLOCK_TABLE)
        )
        kern_stats = _bench(
            lambda: kernel_impl(QUERY_START_LOC, SEQ_LENS, BLOCK_TABLE)
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
            "Kernel: _compressed_slot_mapping_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- block_table shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(
                f"- block_size={BLOCK_SIZE}, compress_ratio={COMPRESS_RATIO}, "
                f"num_tokens={NUM_TOKENS}\n"
            )
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output (integer slot indices, EXACT compare):\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
