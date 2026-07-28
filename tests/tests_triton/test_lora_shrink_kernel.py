"""
Standalone QAIC validation for `_lora_shrink_kernel`.

Source under test:
vllm/lora/ops/triton_ops/lora_shrink_op.py
  - _lora_shrink_kernel  (Punica-style grid dispatcher -> do_shrink_kernel)

The dispatcher selects the active (lora, slice, token-block) from the sorted
routing metadata and calls `do_shrink_kernel`, computing per token
  out[token] = scaling * (x[token] @ A[lora_of_token].T)
where A (LoRA shrink weight) has shape [num_loras, rank, hidden].

We drive it through the file's launcher `lora_shrink` and build the routing
metadata with the repo's own `LoRAKernelMeta` helper (sort + unique +
cumsum), rather than hand-rolling the sorted-token bookkeeping.

Reference: pure PyTorch per-token scaling*(x @ A.T). FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
from vllm.lora.ops.triton_ops.lora_shrink_op import lora_shrink

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_lora_shrink_kernel.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/lora_shrink_op.py"
DEVICE = "qaic"
DTYPE = torch.float16

torch.manual_seed(42)

# Global shared inputs
NUM_TOKENS = 8
HIDDEN = 16
RANK = 8
NUM_LORAS = 2
SCALING = 0.5

X = torch.randn(NUM_TOKENS, HIDDEN, dtype=DTYPE, device=DEVICE)
LORA_A = torch.randn(NUM_LORAS, RANK, HIDDEN, dtype=DTYPE, device=DEVICE)
# Assign each token to a LoRA id in [0, NUM_LORAS).
TOKEN_LORA_MAPPING = torch.tensor(
    [0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32, device=DEVICE
)


def _build_meta():
    meta = LoRAKernelMeta.make(NUM_LORAS, NUM_TOKENS, device=DEVICE)
    meta.prepare_tensors(TOKEN_LORA_MAPPING)
    return meta.meta_args(NUM_TOKENS, specialize_active_lora=False)


def pytorch_ref(x, lora_a, mapping):
    x32 = x.float()
    out = torch.zeros(NUM_TOKENS, RANK, dtype=torch.float32, device=x.device)
    for t in range(NUM_TOKENS):
        lid = int(mapping[t].item())
        if lid < 0:
            continue
        out[t] = SCALING * (x32[t] @ lora_a[lid].float().t())
    return out


def kernel_impl(x, lora_a):
    # output_tensor shape [num_slices, num_tokens, rank]
    out = torch.zeros(1, NUM_TOKENS, RANK, dtype=torch.float32, device=x.device)
    (
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
    ) = _build_meta()
    lora_shrink(
        x,
        [lora_a],
        out,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        SCALING,
    )
    return out[0]


def _log(text):
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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X, LORA_A, TOKEN_LORA_MAPPING)
        kernel_out = kernel_impl(X, LORA_A)
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(X.shape), tuple(LORA_A.shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(X.dtype),
            "device": str(X.device),
            "scaling": SCALING,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, LORA_A, TOKEN_LORA_MAPPING))
        kern_stats = _bench(lambda: kernel_impl(X, LORA_A))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n", error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _lora_shrink_kernel (launcher lora_shrink)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                             f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                             f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                             f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                             f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                             f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n")
                lines.append(f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n")
        else:
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
