"""
Standalone QAIC validation for `moe_fused_mul_sum_kernel`.

Source under test:
vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py
  - moe_fused_mul_sum_kernel  (@triton.jit)

This is the final MoE combine step: each token's top-k expert outputs are
multiplied by their routing weights and summed across the top-k dimension to
produce the combined per-token hidden state.
    inputs:       [num_tokens, top_k, hidden]
    topk_weights: [num_tokens, top_k]
    output:       [num_tokens, hidden]
    output[t, :] = sum_k inputs[t, k, :] * topk_weights[t, k]

Launcher: the repo's own `moe_fused_mul_sum` (no expert_map -> the simple
non-EP path). It tiles over (hidden, tokens) and accumulates in fp32.

Reference: pure PyTorch weighted sum over the top-k axis.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_moe_fused_mul_sum_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/moe_fused_mul_sum.py"

DEVICE = "qaic"
NUM_TOKENS = 8
TOP_K = 2
HIDDEN = 16

torch.manual_seed(42)
INPUTS = torch.randn(
    NUM_TOKENS, TOP_K, HIDDEN, dtype=torch.float32, device=DEVICE
).contiguous()
TOPK_WEIGHTS = torch.rand(
    NUM_TOKENS, TOP_K, dtype=torch.float32, device=DEVICE
).contiguous()


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


def pytorch_ref(inputs, topk_weights):
    """Pure PyTorch weighted sum over top-k:
    out[t,:] = sum_k inputs[t,k,:] * topk_weights[t,k]."""
    inputs = inputs.cpu().to(torch.float32)
    topk_weights = topk_weights.cpu().to(torch.float32)
    return (inputs * topk_weights[:, :, None]).sum(dim=1)


def kernel_impl(inputs, topk_weights):
    return moe_fused_mul_sum(inputs, topk_weights)


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(INPUTS, TOPK_WEIGHTS)
        kernel_out = kernel_impl(INPUTS, TOPK_WEIGHTS)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().max().clamp_min(1e-12)
        stats = {
            "input_shape": tuple(INPUTS.shape),
            "weights_shape": tuple(TOPK_WEIGHTS.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(INPUTS.dtype),
            "device": str(INPUTS.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff.max() / denom).item(),
        }

        pt_stats = _bench(lambda: pytorch_ref(INPUTS, TOPK_WEIGHTS))
        kern_stats = _bench(lambda: kernel_impl(INPUTS, TOPK_WEIGHTS))
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
            "Kernel: moe_fused_mul_sum_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- inputs shape: {stats['input_shape']}\n",
                f"- topk_weights shape: {stats['weights_shape']}\n",
                f"- num_tokens={NUM_TOKENS}, top_k={TOP_K}, hidden={HIDDEN}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- rel_error: {stats['rel_error']}\n",
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
