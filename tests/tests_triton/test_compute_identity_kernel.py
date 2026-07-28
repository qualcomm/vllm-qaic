"""
Standalone QAIC validation for `compute_identity_kernel`.

Source under test:
vllm/model_executor/layers/fused_moe/fused_moe.py
  - compute_identity_kernel  (@triton.jit)
  - zero_experts_compute_triton  (launcher)

`compute_identity_kernel` implements the "identity" (zero-compute) expert
shortcut in MoE: for each token it accumulates a weighted sum of the token's
OWN hidden state over its top-k expert slots, using the per-slot routing
scale. In the launcher, only the slots that were routed to an identity expert
(expert index >= num_experts) keep their scale; slots routed to normal experts
have their scale zeroed. So:

    output[t, :] = sum_i identity_scale[t, i] * hidden_states[t, :]

Launcher: the repo's own `zero_experts_compute_triton` with
zero_expert_type="identity".

Reference: replicate the launcher's scale-masking, then compute the weighted
sum of each token's hidden state over its top-k slots.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_identity_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/fused_moe.py"

# Global inputs (tiny). hidden_dim must be a multiple of BLOCK_SIZE (256) for
# the launcher's grid, so use 256.
NUM_TOKENS = 8
HIDDEN_DIM = 256
TOP_K = 2
NUM_EXPERTS = 4  # experts with index >= NUM_EXPERTS are "identity" experts
DEVICE = "qaic"

_IS_CHILD = os.environ.get("CIK_CHILD") == "1"

if _IS_CHILD or __name__ != "__main__":
    import torch

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm")
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        zero_experts_compute_triton,
    )

    torch.manual_seed(42)
    HIDDEN_STATES = torch.randn(
        NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32, device=DEVICE
    )
    # Expert indices in [0, 2*NUM_EXPERTS): some normal (< NUM_EXPERTS),
    # some identity (>= NUM_EXPERTS).
    EXPERT_INDICES = torch.randint(
        0, 2 * NUM_EXPERTS, (NUM_TOKENS, TOP_K), dtype=torch.int64, device=DEVICE
    )
    EXPERT_SCALES = torch.rand(
        NUM_TOKENS, TOP_K, dtype=torch.float32, device=DEVICE
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


def pytorch_ref(hidden_states, expert_indices, expert_scales):
    """Pure PyTorch identity-expert weighted sum.

    Mirrors zero_experts_compute_triton's masking: for the "identity" type
    only slots whose expert index >= num_experts contribute (their scale is
    kept; normal-expert slots are zeroed).
    """
    h = hidden_states.cpu()
    idx = expert_indices.cpu()
    scales = expert_scales.cpu().clone()

    # identity mask: slot routed to an identity expert (idx >= num_experts)
    zero_expert_scales = scales.clone()
    normal_mask = idx < NUM_EXPERTS
    zero_expert_scales[normal_mask] = 0.0  # keep only identity slots

    out = torch.zeros(NUM_TOKENS, HIDDEN_DIM, dtype=torch.float32)
    for t in range(NUM_TOKENS):
        acc = torch.zeros(HIDDEN_DIM, dtype=torch.float32)
        for i in range(TOP_K):
            acc += h[t] * float(zero_expert_scales[t, i].item())
        out[t] = acc
    return out


def kernel_impl(hidden_states, expert_indices, expert_scales):
    # The launcher mutates its inputs in place; pass clones.
    return zero_experts_compute_triton(
        expert_indices.clone(),
        expert_scales.clone(),
        NUM_EXPERTS,
        "identity",
        hidden_states.clone(),
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(HIDDEN_STATES, EXPERT_INDICES, EXPERT_SCALES)
        kernel_out = kernel_impl(HIDDEN_STATES, EXPERT_INDICES, EXPERT_SCALES)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(HIDDEN_STATES.shape),
            "indices_shape": tuple(EXPERT_INDICES.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(HIDDEN_STATES.dtype),
            "device": str(HIDDEN_STATES.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff.max() / (ref_cpu.abs().max() + 1e-12)).item(),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(HIDDEN_STATES, EXPERT_INDICES, EXPERT_SCALES)
        )
        kern_stats = _bench(
            lambda: kernel_impl(HIDDEN_STATES, EXPERT_INDICES, EXPERT_SCALES)
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
            "Kernel: compute_identity_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- hidden_states shape: {stats['input_shape']}\n")
            lines.append(f"- expert_indices shape: {stats['indices_shape']}\n")
            lines.append(f"- num_experts={NUM_EXPERTS}, top_k={TOP_K}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_error: {stats['rel_error']}\n")
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


def _run_with_crash_guard():
    import subprocess

    if os.environ.get("CIK_CHILD") == "1":
        sys.exit(0 if main() == "SUCCESS" else 1)

    env = dict(os.environ, CIK_CHILD="1")
    proc = subprocess.run([sys.executable, __file__], env=env)
    if proc.returncode < 0:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log(
            f"{timestamp}\n"
            "Kernel: compute_identity_kernel\n"
            f"Kernel file: {KERNEL_FILE_PATH}\n"
            f"Device target: QAIC (device='{DEVICE}')\n"
            "Status: FAILURE\n\n"
            "Error:\n"
            f"Child killed by signal (exit {proc.returncode}) during "
            "Triton->Hexagon compile/execution.\n"
            "\n------------------------------------\n\n"
        )
    sys.exit(proc.returncode if proc.returncode >= 0 else 1)


if __name__ == "__main__":
    _run_with_crash_guard()
