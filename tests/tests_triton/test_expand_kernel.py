"""
Standalone QAIC validation for `expand_kernel`.

Source under test:
vllm/v1/sample/rejection_sampler.py
  - expand_kernel  (launched via `expand_batch_to_tokens`)

Expands a per-request scalar tensor `x` of shape [batch_size] into a per-token
tensor of shape [num_tokens], broadcasting each request's scalar across the
slice [cu_num_tokens[req-1], cu_num_tokens[req]). Any occurrence of the value
`replace_from` is replaced by `replace_to` before broadcasting (used e.g. to
turn the greedy temperature 0 into 1). `cu_num_tokens` is an INCLUSIVE
cumulative sum of length batch_size.

We validate against a pure PyTorch reference. Integer-exact comparison.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_expand_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/sample/rejection_sampler.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.sample.rejection_sampler import expand_batch_to_tokens

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
NUM_REQS = 4
NUM_TOKENS_PER_REQ = [2, 3, 1, 2]
_cu = []
_acc = 0
for _n in NUM_TOKENS_PER_REQ:
    _acc += _n
    _cu.append(_acc)
CU_NUM_TOKENS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
NUM_TOKENS = _cu[-1]

REPLACE_FROM = 0
REPLACE_TO = 1
# Per-request scalar; include a 0 to exercise the replace path.
X = torch.tensor([5, 0, 7, 3], dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(x, cu_num_tokens, num_tokens, replace_from, replace_to):
    x = x.cpu()
    cu = cu_num_tokens.cpu()
    out = x.new_empty(num_tokens)
    for r in range(NUM_REQS):
        start = 0 if r == 0 else int(cu[r - 1].item())
        end = int(cu[r].item())
        val = int(x[r].item())
        if val == replace_from:
            val = replace_to
        out[start:end] = val
    return out


def kernel_impl(x, cu_num_tokens, num_tokens, replace_from, replace_to):
    return expand_batch_to_tokens(
        x, cu_num_tokens, num_tokens, replace_from=replace_from, replace_to=replace_to
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(X, CU_NUM_TOKENS, NUM_TOKENS, REPLACE_FROM, REPLACE_TO)
        k_out = kernel_impl(
            X, CU_NUM_TOKENS, NUM_TOKENS, REPLACE_FROM, REPLACE_TO
        ).cpu()
        mism = int((k_out != ref_out).sum().item())
        assert mism == 0, f"expanded tensor mismatch count={mism}"
        stats = {
            "input_shape": tuple(X.shape),
            "output_shape": tuple(k_out.shape),
            "dtype": str(k_out.dtype),
            "device": str(X.device),
            "mismatch": mism,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(
            X, CU_NUM_TOKENS, NUM_TOKENS, REPLACE_FROM, REPLACE_TO
        ))
        kern_stats = _bench(lambda: kernel_impl(
            X, CU_NUM_TOKENS, NUM_TOKENS, REPLACE_FROM, REPLACE_TO
        ))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: expand_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"num_tokens_per_req: {NUM_TOKENS_PER_REQ}\n")
            lines.append(f"input shape: {stats['input_shape']}\n")
            lines.append(f"output shape: {stats['output_shape']}\n")
            lines.append(f"dtype: {stats['dtype']}  device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch: {stats['mismatch']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
