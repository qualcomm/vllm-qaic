"""
Standalone QAIC validation for `_lora_expand_kernel`.

Source under test:
vllm/lora/ops/triton_ops/lora_expand_op.py
  - _lora_expand_kernel  (Punica-style grid dispatcher -> do_expand_kernel)

The dispatcher selects the active (lora, slice, token-block) from the sorted
routing metadata and calls `do_expand_kernel`, computing per token
  out[token] = shrink_out[token] @ B[lora_of_token].T
where B (LoRA expand weight) has shape [num_loras, hidden, rank] and the
input is the shrink output of shape [num_slices, num_tokens, rank].

We drive it through the file's launcher `lora_expand` and build routing
metadata with the repo's `LoRAKernelMeta` helper (single slice here).

Reference: pure PyTorch per-token shrink_out @ B.T. FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.lora_expand_op import lora_expand
from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_lora_expand_kernel.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/lora_expand_op.py"
DEVICE = "qaic"
DTYPE = torch.float16

torch.manual_seed(42)

# Global shared inputs
NUM_TOKENS = 8
RANK = 8
HIDDEN = 16
NUM_LORAS = 2
NUM_SLICES = 1

# shrink output: [num_slices, num_tokens, rank]
SHRINK_OUT = torch.randn(NUM_SLICES, NUM_TOKENS, RANK, dtype=DTYPE, device=DEVICE)
# LoRA B weight: [num_loras, hidden, rank]
LORA_B = torch.randn(NUM_LORAS, HIDDEN, RANK, dtype=DTYPE, device=DEVICE)
TOKEN_LORA_MAPPING = torch.tensor(
    [0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.int32, device=DEVICE
)


def _build_meta():
    meta = LoRAKernelMeta.make(NUM_LORAS, NUM_TOKENS, device=DEVICE)
    meta.prepare_tensors(TOKEN_LORA_MAPPING)
    return meta.meta_args(NUM_TOKENS, specialize_active_lora=False)


def pytorch_ref(shrink_out, lora_b, mapping):
    inp = shrink_out[0].float()  # [num_tokens, rank]
    out = torch.zeros(NUM_TOKENS, HIDDEN, dtype=torch.float32, device=inp.device)
    for t in range(NUM_TOKENS):
        lid = int(mapping[t].item())
        if lid < 0:
            continue
        out[t] = inp[t] @ lora_b[lid].float().t()
    return out


def kernel_impl(shrink_out, lora_b):
    out = torch.zeros(NUM_TOKENS, HIDDEN * NUM_SLICES, dtype=DTYPE, device=shrink_out.device)
    (
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
    ) = _build_meta()
    lora_expand(
        shrink_out,
        [lora_b],
        out,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        offset_start=0,
        add_inputs=False,
    )
    return out


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
        ref_out = pytorch_ref(SHRINK_OUT, LORA_B, TOKEN_LORA_MAPPING)
        kernel_out = kernel_impl(SHRINK_OUT, LORA_B).float()
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(SHRINK_OUT.shape), tuple(LORA_B.shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(SHRINK_OUT.dtype),
            "device": str(SHRINK_OUT.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(SHRINK_OUT, LORA_B, TOKEN_LORA_MAPPING))
        kern_stats = _bench(lambda: kernel_impl(SHRINK_OUT, LORA_B))
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
            "Kernel: _lora_expand_kernel (launcher lora_expand)\n",
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
