"""
Standalone QAIC validation for `kv_cache_scatter_kernel`.

Source under test:
vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/gather_scatter_helper.py
  - kv_cache_scatter_kernel  (@triton.jit)
  - scatter_kv_caches        (launcher; MLA path exercised here)

Inverse of the gather kernel: scatters rows from a packed source tensor into a
set of per-layer KV-cache tensors. `kv_caches_ptrs` is an int64 tensor of
per-layer `data_ptr()` addresses; for grid cell (layer_idx, token_pos) the
kernel copies `hidden_size` elements from the source into the target cache row
selected by token_indices[token_pos], in BLOCK_SIZE chunks.

MLA layout (is_mla=True):
  * source tensor:   [num_layers, num_tokens_in_block, hidden_size]
  * per-layer cache: [total_token_in_kvcache, hidden_size]
  * kv_caches[l][token_indices[t], :] = source[l, t, :]

Caches are mutated in place. We pre-fill caches with a sentinel, scatter, and
compare the resulting cache tensors against a pure-PyTorch scatter. Comparison
is EXACT (pure copy) via torch.equal.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_kv_cache_scatter_kernel.txt")
KERNEL_FILE_PATH = (
    "vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/utils/"
    "gather_scatter_helper.py"
)
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.utils.gather_scatter_helper import (  # noqa: E402,E501
    scatter_kv_caches,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_LAYERS = 2
TOTAL_TOKEN = 8
NUM_TOKENS_IN_BLOCK = 3
HIDDEN = 256  # spans 2 BLOCK_SIZE (128) chunks in the kernel
TOKEN_INDICES = [0, 3, 5]

# Source rows to scatter: [num_layers, num_tokens_in_block, hidden].
SRC = torch.randn(
    NUM_LAYERS, NUM_TOKENS_IN_BLOCK, HIDDEN, dtype=torch.float32, device=DEVICE
)

# Per-layer destination caches, kept alive at module scope. Cloned fresh in
# kernel_impl each call so the in-place scatter is reproducible.
_INIT_CACHES = [
    torch.full((TOTAL_TOKEN, HIDDEN), -1.0, dtype=torch.float32, device=DEVICE)
    for _ in range(NUM_LAYERS)
]


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


def pytorch_ref(src, token_indices):
    """Pure PyTorch scatter into sentinel-filled caches (no kernel calls)."""
    idx = torch.tensor(token_indices, dtype=torch.long)
    caches = [
        torch.full((TOTAL_TOKEN, HIDDEN), -1.0, dtype=torch.float32)
        for _ in range(NUM_LAYERS)
    ]
    src_cpu = src.cpu()
    for l in range(NUM_LAYERS):
        caches[l][idx] = src_cpu[l]
    return torch.stack(caches, dim=0)  # [num_layers, total_token, hidden]


def kernel_impl(src, token_indices):
    # Fresh sentinel caches each call so scatter is reproducible; build the
    # pointer array from these live tensors.
    caches = [c.clone() for c in _INIT_CACHES]
    ptrs = torch.tensor(
        [c.data_ptr() for c in caches], dtype=torch.int64, device=DEVICE
    )
    scatter_kv_caches(ptrs, TOTAL_TOKEN, src, token_indices, is_mla=True)
    return torch.stack(caches, dim=0)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(SRC, TOKEN_INDICES)
        kernel_out = kernel_impl(SRC, TOKEN_INDICES)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        assert torch.equal(kernel_cpu, ref_cpu), "scattered caches mismatch"

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(SRC.shape),
            "output_shape": tuple(kernel_out.shape),
            "in_dtype": str(SRC.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(SRC.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(SRC, TOKEN_INDICES))
        kern_stats = _bench(lambda: kernel_impl(SRC, TOKEN_INDICES))
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
            "Kernel: kv_cache_scatter_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- src shape: {stats['input_shape']}\n")
            lines.append(f"- token_indices: {TOKEN_INDICES}, is_mla=True\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- caches shape: {stats['output_shape']}\n")
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
