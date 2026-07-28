"""
Standalone QAIC validation for `kv_cache_gather_kernel`.

Source under test:
vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py
  - kv_cache_gather_kernel  (@triton.jit)
  - gather_kv_caches        (launcher; MLA path exercised here)

The kernel gathers rows out of a set of per-layer KV-cache tensors into a
packed destination tensor. `kv_caches_ptrs` is an int64 tensor holding the raw
`data_ptr()` of each per-layer cache; the kernel casts each address back to a
typed pointer and, for grid cell (layer_idx, token_pos), copies
`hidden_size` elements in BLOCK_SIZE chunks.

MLA layout (is_mla=True):
  * per-layer cache:  [total_token_in_kvcache, hidden_size]
  * dst tensor:       [num_layers, num_tokens_in_block, hidden_size]
  * dst[l, t, :] = kv_caches[l][token_indices[t], :]

We build the pointer array from module-scope per-layer caches (kept alive so
data_ptr() stays valid) and call the repo launcher directly. Comparison is
EXACT (float copy, no arithmetic) via torch.equal.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_kv_cache_gather_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/"
    "gather_scatter_helper.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.gather_scatter_helper import (  # noqa: E402,E501
    gather_kv_caches,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_LAYERS = 2
TOTAL_TOKEN = 8
NUM_TOKENS_IN_BLOCK = 3
HIDDEN = 256  # spans 2 BLOCK_SIZE (128) chunks in the kernel
TOKEN_INDICES = [0, 3, 5]

# Per-layer caches, kept alive at module scope so data_ptr() stays valid.
KV_CACHES = [
    torch.randn(TOTAL_TOKEN, HIDDEN, dtype=torch.float32, device=DEVICE)
    for _ in range(NUM_LAYERS)
]
KV_CACHES_PTRS = torch.tensor(
    [c.data_ptr() for c in KV_CACHES], dtype=torch.int64, device=DEVICE
)


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


def pytorch_ref(kv_caches, token_indices):
    """Pure PyTorch gather (no kernel calls)."""
    idx = torch.tensor(token_indices, dtype=torch.long)
    rows = [c.cpu()[idx] for c in kv_caches]  # each [num_tokens, hidden]
    return torch.stack(rows, dim=0)  # [num_layers, num_tokens, hidden]


def kernel_impl(kv_caches_ptrs, token_indices):
    dst = torch.empty(
        NUM_LAYERS, NUM_TOKENS_IN_BLOCK, HIDDEN,
        dtype=torch.float32, device=DEVICE,
    )
    gather_kv_caches(
        kv_caches_ptrs, TOTAL_TOKEN, dst, token_indices, is_mla=True
    )
    return dst


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(KV_CACHES, TOKEN_INDICES)
        kernel_out = kernel_impl(KV_CACHES_PTRS, TOKEN_INDICES)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), "gathered rows mismatch"

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": (NUM_LAYERS, TOTAL_TOKEN, HIDDEN),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(KV_CACHES[0].dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(kernel_out.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(KV_CACHES, TOKEN_INDICES))
        kern_stats = _bench(lambda: kernel_impl(KV_CACHES_PTRS, TOKEN_INDICES))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
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
            "Kernel: kv_cache_gather_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- per-layer cache shape: {stats['input_shape']}\n")
            lines.append(f"- token_indices: {TOKEN_INDICES}, is_mla=True\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']} (exact copy)\n")
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
