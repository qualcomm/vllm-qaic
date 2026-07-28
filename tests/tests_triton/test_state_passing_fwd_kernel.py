"""
Standalone QAIC validation for `_state_passing_fwd_kernel`.

Source under test:
vllm/model_executor/layers/mamba/ops/ssd_state_passing.py
  - _state_passing_fwd_kernel  (launched via _state_passing_fwd)

Propagates the per-chunk SSM states across chunks with a sequential
chunk-level decay recurrence, producing the running inter-chunk state at each
chunk boundary. Per source (HAS_INITSTATES=False), with the running state
initialised to zero:
    for each chunk c in the sequence:
        dA_last = dA_cumsum[h, c, chunk_size - 1]
        state = exp(dA_last) * state + chunk_state[c]
        out[c] = state
So out[c] = out[c-1] * exp(dA_last[c]) + chunk_state[c], out[-1] = 0. The decay
factor is applied to the *incoming* running state at each chunk.

Shapes: states (nchunks, nheads, dim); dA_cumsum (nheads, nchunks, chunk_size);
last_chunk_indices (batch,). Output: (nchunks, nheads, dim).

Small: batch=1, nchunks=3, nheads=2, dim=8, chunk_size=16.

Reference: pure PyTorch sequential recurrence over chunks (no initial state).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_state_passing_fwd.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/mamba/ops/ssd_state_passing.py"

DEVICE = "qaic"
NCHUNKS = 3
NHEADS = 2
DIM = 8
CHUNK_SIZE = 16
BATCH = 1

torch.manual_seed(42)
STATES = torch.randn(NCHUNKS, NHEADS, DIM, dtype=torch.float32, device=DEVICE)
# dA_cumsum: only the last column (chunk_size-1) is read per chunk. Use
# negative values so exp(dA_last) is a stable decay in (0, 1].
DA_CUMSUM = -torch.rand(
    NHEADS, NCHUNKS, CHUNK_SIZE, dtype=torch.float32, device=DEVICE
)
# Single sequence spanning all chunks -> last chunk index = NCHUNKS - 1.
LAST_CHUNK_INDICES = torch.tensor(
    [NCHUNKS - 1], dtype=torch.int32, device=DEVICE
)


def pytorch_ref(states, dA_cumsum, last_chunk_indices):
    """Pure PyTorch sequential inter-chunk state passing (no init state)."""
    states = states.cpu().to(torch.float32)
    dA_cumsum = dA_cumsum.cpu().to(torch.float32)
    lci = last_chunk_indices.cpu().tolist()

    out = torch.zeros(NCHUNKS, NHEADS, DIM, dtype=torch.float32)
    for b in range(BATCH):
        chunk_end = lci[b] + 1
        chunk_start = (lci[b - 1] + 1) if b > 0 else 0
        for h in range(NHEADS):
            running = torch.zeros(DIM, dtype=torch.float32)
            for c in range(chunk_start, chunk_end):
                dA_last = dA_cumsum[h, c, CHUNK_SIZE - 1]
                running = torch.exp(dA_last) * running + states[c, h]
                out[c, h] = running
    return out


def kernel_impl(states, dA_cumsum, last_chunk_indices):
    return _state_passing_fwd(
        states,
        dA_cumsum,
        last_chunk_indices,
        initial_states=None,
        out_dtype=torch.float32,
    )


def _log(text):
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
        ref_out = pytorch_ref(STATES, DA_CUMSUM, LAST_CHUNK_INDICES)
        kernel_out = kernel_impl(STATES, DA_CUMSUM, LAST_CHUNK_INDICES)
        ref_cpu = ref_out
        ker_cpu = kernel_out.cpu().to(torch.float32)
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "states_shape": tuple(STATES.shape),
            "dA_cumsum_shape": tuple(DA_CUMSUM.shape),
            "out_shape": tuple(kernel_out.shape),
            "in_dtype": str(STATES.dtype),
            "out_dtype": str(kernel_out.dtype),
            "device": str(STATES.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(STATES, DA_CUMSUM, LAST_CHUNK_INDICES))
        kern_stats = _bench(lambda: kernel_impl(STATES, DA_CUMSUM, LAST_CHUNK_INDICES))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
        status = "SUCCESS"
        print("SUCCESS", stats)
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _state_passing_fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
        ]
        if status == "SUCCESS":
            for k, v in stats.items():
                lines.append(f"- {k}: {v}\n")
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
            lines.append("Error:\n" + error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
