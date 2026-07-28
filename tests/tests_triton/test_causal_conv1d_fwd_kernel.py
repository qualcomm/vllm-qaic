"""
Standalone QAIC validation for `_causal_conv1d_fwd_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/causal_conv1d.py
  - _causal_conv1d_fwd_kernel  (launched via causal_conv1d_fn)

Causal depthwise 1D convolution over a (continuous-batched / varlen) sequence
laid out as x: (dim, cu_seqlen). For each channel and output position t:
    out[c, t] = bias[c] + sum_{k=0}^{width-1} w[c, k] * x[c, t - (width-1) + k]
with left zero-padding for t - (width-1) + k < 0 (i.e. no initial state), then
an optional SiLU activation out = out * sigmoid(out). The kernel also manages
conv-state cache writes (prefix-cache-aware, chunked), but for the simplest
config below (single sequence, no initial state) those writes do not affect the
returned output.

Simplest config: single sequence, has_initial_state=False, with bias + SiLU,
dim=16, seqlen=32, width=4. conv_states cache present (required by launcher);
cache_indices points to a non-null cache line.

Reference: pure PyTorch left-padded causal depthwise conv1d + bias + SiLU.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_causal_conv1d_fwd.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"

DEVICE = "qaic"
DIM = 16
SEQLEN = 32
WIDTH = 4
STATE_LEN = WIDTH - 1
NUM_CACHE_LINES = 2
ACTIVATION = "silu"

torch.manual_seed(42)
# x: (dim, cu_seqlen). One sequence of length SEQLEN.
X = torch.randn(DIM, SEQLEN, dtype=torch.float32, device=DEVICE)
WEIGHT = torch.randn(DIM, WIDTH, dtype=torch.float32, device=DEVICE)
BIAS = torch.randn(DIM, dtype=torch.float32, device=DEVICE)
# conv_states cache (num_cache_lines, dim, state_len). Line 0 is the null block.
CONV_STATES = torch.zeros(
    NUM_CACHE_LINES, DIM, STATE_LEN, dtype=torch.float32, device=DEVICE
)
QUERY_START_LOC = torch.tensor([0, SEQLEN], dtype=torch.int32, device=DEVICE)
# Point sequence 0 at cache line 1 (line 0 == null_block_id).
CACHE_INDICES = torch.tensor([1], dtype=torch.int32, device=DEVICE)
HAS_INITIAL_STATE = torch.tensor([False], dtype=torch.bool, device=DEVICE)


def pytorch_ref(x, weight, bias):
    """Pure PyTorch left-padded causal depthwise conv1d + bias + SiLU.

    out[c, t] = bias[c] + sum_k w[c, k] * x_padded[c, t + k],
    where x_padded has (width-1) zeros on the left.
    """
    x = x.cpu().to(torch.float32)
    weight = weight.cpu().to(torch.float32)
    bias = bias.cpu().to(torch.float32)
    pad = WIDTH - 1
    x_pad = torch.zeros(DIM, SEQLEN + pad, dtype=torch.float32)
    x_pad[:, pad:] = x
    out = torch.zeros(DIM, SEQLEN, dtype=torch.float32)
    for t in range(SEQLEN):
        window = x_pad[:, t : t + WIDTH]  # (dim, width)
        out[:, t] = (window * weight).sum(dim=1) + bias
    # SiLU
    out = out * torch.sigmoid(out)
    return out


def kernel_impl(x, weight, bias):
    conv_states = CONV_STATES.clone()
    out = causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        QUERY_START_LOC,
        cache_indices=CACHE_INDICES,
        has_initial_state=HAS_INITIAL_STATE,
        activation=ACTIVATION,
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
        ref_out = pytorch_ref(X, WEIGHT, BIAS)
        kernel_out = kernel_impl(X, WEIGHT, BIAS)
        ref_cpu = ref_out
        ker_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "x_shape": tuple(X.shape),
            "weight_shape": tuple(WEIGHT.shape),
            "out_shape": tuple(kernel_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(X.device),
            "activation": ACTIVATION,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, WEIGHT, BIAS))
        kern_stats = _bench(lambda: kernel_impl(X, WEIGHT, BIAS))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: _causal_conv1d_fwd_kernel\n",
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
