"""
Standalone QAIC validation for `_pack_dense_prefix_to_ragged_kernel`.

Source under test:
vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
  - _pack_dense_prefix_to_ragged_kernel
  - launcher: build_ragged_indices_from_dense(indices, lengths, num_rows)

Packs a dense fixed-width top-k index matrix (with per-row valid lengths) into a
compact ragged (CSR-style) index array:

    lengths = clamp(lengths, 0, width)
    indptr  = cumsum(lengths)                       (int32, len rows+1)
    for offset in [0, lengths[row]):
        v = indices[row, offset]
        v = v if (0 <= v < num_rows) else -1        (num_rows filter, if >= 0)
        out[indptr[row] + offset] = v

Integer index kernel -> EXACT-equality on the packed ragged array and indptr.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_pack_dense_prefix_to_ragged_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_ROWS = 4
WIDTH = 8
NUM_KV = 20  # num_rows filter bound
LENGTHS = torch.tensor([3, 5, 0, 8], dtype=torch.int32, device=DEVICE)
# Mix of in-range, out-of-range (>= NUM_KV) and -1 values.
INDICES = torch.randint(
    -1, NUM_KV + 5, (NUM_ROWS, WIDTH), dtype=torch.int32, device=DEVICE
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


def pytorch_ref(indices, lengths, num_rows):
    indices = indices.cpu()
    lengths = lengths.cpu().clamp(min=0, max=WIDTH)
    indptr = torch.zeros(NUM_ROWS + 1, dtype=torch.int32)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    total = int(indptr[-1].item())
    out = torch.empty(total, dtype=torch.int32)
    for r in range(NUM_ROWS):
        start = int(indptr[r].item())
        row_len = int(lengths[r].item())
        for off in range(row_len):
            v = int(indices[r, off].item()) if off < WIDTH else -1
            if num_rows >= 0 and not (0 <= v < num_rows):
                v = -1
            out[start + off] = v
    return out, indptr


def kernel_impl(indices, lengths, num_rows):
    flat, indptr = build_ragged_indices_from_dense(indices, lengths, num_rows=num_rows)
    total = int(indptr[-1].item())
    return flat[:total], indptr


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_indptr = pytorch_ref(INDICES, LENGTHS, NUM_KV)
        k_out, k_indptr = kernel_impl(INDICES, LENGTHS, NUM_KV)
        k_out = k_out.cpu()
        k_indptr = k_indptr.cpu()

        out_mm = int((k_out != ref_out).sum().item())
        indptr_mm = int((k_indptr != ref_indptr).sum().item())
        assert indptr_mm == 0, f"indptr mismatch={indptr_mm}"
        assert out_mm == 0, f"ragged mismatch={out_mm}"

        stats = {
            "input_shape": tuple(INDICES.shape),
            "ragged_shape": tuple(k_out.shape),
            "indptr_shape": tuple(k_indptr.shape),
            "in_dtype": str(INDICES.dtype),
            "out_dtype": str(k_out.dtype),
            "device": DEVICE,
            "out_mm": out_mm,
            "indptr_mm": indptr_mm,
            "indptr": k_indptr.tolist(),
        }

        pt_stats = _bench(lambda: pytorch_ref(INDICES, LENGTHS, NUM_KV))
        kern_stats = _bench(lambda: kernel_impl(INDICES, LENGTHS, NUM_KV))
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
            "Kernel: _pack_dense_prefix_to_ragged_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- indices shape: {stats['input_shape']} dtype {stats['in_dtype']}\n",
                f"- lengths: {LENGTHS.cpu().tolist()}, num_rows(filter)={NUM_KV}, width={WIDTH}\n",
                f"- device: {stats['device']}\n\n",
                "Output (EXACT-equality comparison):\n",
                f"- ragged shape: {stats['ragged_shape']} dtype {stats['out_dtype']}\n",
                f"- indptr shape: {stats['indptr_shape']}, indptr={stats['indptr']}\n",
                f"- ragged mismatches: {stats['out_mm']}\n",
                f"- indptr mismatches: {stats['indptr_mm']}\n",
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
