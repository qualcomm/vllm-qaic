"""
Standalone QAIC validation for `_causal_conv1d_update_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/causal_conv1d.py
  - _causal_conv1d_update_kernel  (launched via causal_conv1d_update)

Single-step (decode) causal depthwise conv1d update. For each sequence in the
batch it reads that sequence's rolling conv-state cache window
[s_0, s_1, ..., s_{state_len-1}] (state_len = width - 1), forms the full width
window [s_0, ..., s_{state_len-1}, x_new] with the new token appended on the
right, and computes:
    out[c] = bias[c] + sum_{k=0}^{width-1} w[c, k] * window[c, k]
followed by optional SiLU. It then rolls the conv-state left and writes the new
token, so the cache becomes [s_1, ..., s_{state_len-1}, x_new] for the next
step. cache line selection is via conv_state_indices (line 0 is the null block).

Simplest config: batch=2, dim=16, width=4, single token per sequence, with
bias + SiLU. We validate the returned per-step output AND the rolled conv_state
cache lines that were updated.

Reference: pure PyTorch roll + append + weighted sum + bias + SiLU.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_causal_conv1d_update.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/causal_conv1d.py"

DEVICE = "qaic"
BATCH = 2
DIM = 16
WIDTH = 4
STATE_LEN = WIDTH - 1
NUM_CACHE_LINES = BATCH + 1  # +1 so line 0 stays the null block
ACTIVATION = "silu"

torch.manual_seed(42)
# x: (batch, dim) single-token decode step.
X = torch.randn(BATCH, DIM, dtype=torch.float32, device=DEVICE)
WEIGHT = torch.randn(DIM, WIDTH, dtype=torch.float32, device=DEVICE)
BIAS = torch.randn(DIM, dtype=torch.float32, device=DEVICE)
# conv_state cache: (num_cache_lines, dim, state_len). Line 0 == null block.
CONV_STATE = torch.randn(
    NUM_CACHE_LINES, DIM, STATE_LEN, dtype=torch.float32, device=DEVICE
)
# Map batch item b -> cache line b + 1 (avoid null line 0).
CONV_STATE_INDICES = torch.tensor(
    [1, 2], dtype=torch.int32, device=DEVICE
)


def pytorch_ref(x, conv_state, weight, bias, conv_state_indices):
    """Pure PyTorch single-step conv update.

    Returns (out, new_conv_state) where new_conv_state is the full updated
    cache tensor (only the referenced lines change).
    """
    x = x.cpu().to(torch.float32)
    conv_state = conv_state.cpu().to(torch.float32).clone()
    weight = weight.cpu().to(torch.float32)
    bias = bias.cpu().to(torch.float32)
    idx = conv_state_indices.cpu().tolist()

    out = torch.zeros(BATCH, DIM, dtype=torch.float32)
    for b in range(BATCH):
        line = idx[b]
        state = conv_state[line]  # (dim, state_len)
        # full window: [state_0, ..., state_{state_len-1}, x_new]
        window = torch.cat([state, x[b][:, None]], dim=1)  # (dim, width)
        out[b] = (window * weight).sum(dim=1) + bias
        # roll left and append new token
        conv_state[line] = window[:, 1:]
    # SiLU activation
    out = out * torch.sigmoid(out)
    return out, conv_state


def kernel_impl(x, conv_state, weight, bias, conv_state_indices):
    cs = conv_state.clone()
    out = causal_conv1d_update(
        x,
        cs,
        weight,
        bias=bias,
        activation=ACTIVATION,
        conv_state_indices=conv_state_indices,
    )
    return out, cs


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
        ref_out, ref_state = pytorch_ref(
            X, CONV_STATE, WEIGHT, BIAS, CONV_STATE_INDICES
        )
        ker_out, ker_state = kernel_impl(
            X, CONV_STATE, WEIGHT, BIAS, CONV_STATE_INDICES
        )
        ro, ko = ref_out, ker_out.cpu().to(torch.float32)
        rs, ks = ref_state, ker_state.cpu().to(torch.float32)
        torch.testing.assert_close(ko, ro, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ks, rs, rtol=1e-3, atol=1e-3)
        odiff = (ko - ro).abs()
        sdiff = (ks - rs).abs()
        stats = {
            "x_shape": tuple(X.shape),
            "conv_state_shape": tuple(CONV_STATE.shape),
            "out_shape": tuple(ker_out.shape),
            "in_dtype": str(X.dtype),
            "out_dtype": str(ker_out.dtype),
            "device": str(X.device),
            "activation": ACTIVATION,
            "out_max_abs_diff": odiff.max().item(),
            "out_mean_abs_diff": odiff.mean().item(),
            "state_max_abs_diff": sdiff.max().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(X, CONV_STATE, WEIGHT, BIAS, CONV_STATE_INDICES))
        kern_stats = _bench(lambda: kernel_impl(X, CONV_STATE, WEIGHT, BIAS, CONV_STATE_INDICES))
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
            "Kernel: _causal_conv1d_update_kernel\n",
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
