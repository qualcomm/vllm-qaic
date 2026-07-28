"""
Standalone QAIC validation for `merge_attn_states_kernel`.

Source under test:
vllm/v1/attention/ops/triton_merge_attn_states.py
  - merge_attn_states_kernel  (Combines prefix-context and suffix partial
    attention outputs/LSEs into a single numerically-stable merged output,
    implementing section 2.2 of https://www.arxiv.org/pdf/2501.01005.)

Per (token, head):
    max_lse   = max(prefix_lse, suffix_lse)
    p_se      = exp(prefix_lse - max_lse)
    s_se      = exp(suffix_lse - max_lse)
    out_se    = p_se + s_se
    p_scale   = p_se / out_se
    s_scale   = s_se / out_se
    out       = prefix_output * p_scale + suffix_output * s_scale
    out_lse   = log(out_se) + max_lse   (if output_lse requested)

We use finite, well-separated lse values and `prefill_tokens_with_context=None`
(defaults internally to num_tokens, i.e. every token uses the "with context"
merge path), so the early-return "no context" / -inf special-casing branch
is never exercised.

Reference: pure PyTorch reimplementation of the above (no triton/vllm calls).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_merge_attn_states_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_merge_attn_states.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_merge_attn_states import (  # noqa: E402
    merge_attn_states,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
NUM_TOKENS = 8
NUM_HEADS = 2
HEAD_SIZE = 16

torch.manual_seed(42)

PREFIX_OUTPUT = torch.randn(
    NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
)
SUFFIX_OUTPUT = torch.randn(
    NUM_TOKENS, NUM_HEADS, HEAD_SIZE, dtype=torch.float32, device=DEVICE
)
# Well-separated, finite lse values (shape [num_heads, num_tokens]) to avoid
# the zero-length-sequence -inf/inf special-casing.
PREFIX_LSE = (torch.rand(NUM_HEADS, NUM_TOKENS, dtype=torch.float32) * 4.0 - 2.0).to(
    DEVICE
)
SUFFIX_LSE = (torch.rand(NUM_HEADS, NUM_TOKENS, dtype=torch.float32) * 4.0 - 2.0).to(
    DEVICE
)


def pytorch_ref(prefix_output, prefix_lse, suffix_output, suffix_lse):
    """Pure PyTorch reference implementation.

    Requirements (Claude.md):
      - Pure PyTorch only.
      - No custom kernel calls.
      - No Triton kernel calls.
      - No vLLM kernel calls.
      - No QAIC custom operator calls.
    """
    prefix_output = prefix_output.float()
    suffix_output = suffix_output.float()
    prefix_lse = prefix_lse.float()
    suffix_lse = suffix_lse.float()

    max_lse = torch.maximum(prefix_lse, suffix_lse)  # [num_heads, num_tokens]
    p_se = torch.exp(prefix_lse - max_lse)
    s_se = torch.exp(suffix_lse - max_lse)
    out_se = p_se + s_se
    p_scale = p_se / out_se
    s_scale = s_se / out_se

    # [num_heads, num_tokens] -> [num_tokens, num_heads, 1] for broadcasting.
    p_scale_b = p_scale.transpose(0, 1).unsqueeze(-1)
    s_scale_b = s_scale.transpose(0, 1).unsqueeze(-1)

    out = prefix_output * p_scale_b + suffix_output * s_scale_b
    out_lse = torch.log(out_se) + max_lse  # [num_heads, num_tokens]

    return out, out_lse


def kernel_impl(prefix_output, prefix_lse, suffix_output, suffix_lse):
    """Kernel wrapper: launch only.

    Requirements (Claude.md):
      - Kernel launch only.
      - Minimal setup logic.
      - No reference implementation logic.
      - No correctness-check logic.
      - No validation logic.
    """
    output = torch.empty_like(prefix_output)
    output_lse = torch.empty_like(prefix_lse)
    merge_attn_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output_lse=output_lse,
        prefill_tokens_with_context=None,
        output_scale=None,
    )
    return output, output_lse


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out, ref_lse = pytorch_ref(
            PREFIX_OUTPUT, PREFIX_LSE, SUFFIX_OUTPUT, SUFFIX_LSE
        )
        kernel_out, kernel_lse = kernel_impl(
            PREFIX_OUTPUT, PREFIX_LSE, SUFFIX_OUTPUT, SUFFIX_LSE
        )

        ref_out_cpu = ref_out.cpu()
        ref_lse_cpu = ref_lse.cpu()
        kernel_out_cpu = kernel_out.cpu()
        kernel_lse_cpu = kernel_lse.cpu()

        torch.testing.assert_close(kernel_out_cpu, ref_out_cpu, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(kernel_lse_cpu, ref_lse_cpu, rtol=1e-3, atol=1e-3)

        diff_out = (kernel_out_cpu - ref_out_cpu).abs()
        diff_lse = (kernel_lse_cpu - ref_lse_cpu).abs()
        max_abs_diff = max(diff_out.max().item(), diff_lse.max().item())
        mean_abs_diff = (diff_out.mean().item() + diff_lse.mean().item()) / 2.0
        rel_err_out = (diff_out.max() / (ref_out_cpu.abs().max() + 1e-8)).item()
        rel_err_lse = (diff_lse.max() / (ref_lse_cpu.abs().max() + 1e-8)).item()

        stats = {
            "prefix_output_shape": tuple(PREFIX_OUTPUT.shape),
            "suffix_output_shape": tuple(SUFFIX_OUTPUT.shape),
            "prefix_lse_shape": tuple(PREFIX_LSE.shape),
            "suffix_lse_shape": tuple(SUFFIX_LSE.shape),
            "output_shape": tuple(kernel_out.shape),
            "output_lse_shape": tuple(kernel_lse.shape),
            "dtype": str(PREFIX_OUTPUT.dtype),
            "device": str(PREFIX_OUTPUT.device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "rel_err_out": rel_err_out,
            "rel_err_lse": rel_err_lse,
        }

        pt_stats = _bench(lambda: pytorch_ref(
            PREFIX_OUTPUT, PREFIX_LSE, SUFFIX_OUTPUT, SUFFIX_LSE))
        kern_stats = _bench(lambda: kernel_impl(
            PREFIX_OUTPUT, PREFIX_LSE, SUFFIX_OUTPUT, SUFFIX_LSE))
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
            "Kernel: merge_attn_states_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- prefix_output shape: {stats['prefix_output_shape']}\n")
            lines.append(f"- suffix_output shape: {stats['suffix_output_shape']}\n")
            lines.append(f"- prefix_lse shape: {stats['prefix_lse_shape']}\n")
            lines.append(f"- suffix_lse shape: {stats['suffix_lse_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- output_lse shape: {stats['output_lse_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err (output): {stats['rel_err_out']}\n")
            lines.append(f"- rel_err (output_lse): {stats['rel_err_lse']}\n")
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
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
