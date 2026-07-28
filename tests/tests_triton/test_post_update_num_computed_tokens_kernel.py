"""
Standalone QAIC validation for `_post_update_num_computed_tokens_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _post_update_num_computed_tokens_kernel  (launched via
    `post_update_num_computed_tokens`)

For each batch row it computes query_len = query_start_loc[b+1] -
query_start_loc[b] and increments the persistent
num_computed_tokens[req_state_idx] by query_len.

Integer-exact validation of the mutated num_computed_tokens buffer.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_post_update_num_computed_tokens_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.gpu.input_batch import post_update_num_computed_tokens

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
MAX_NUM_REQS = 6
NUM_REQS = 4

IDX_MAPPING = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)
QUERY_LENS = [4, 3, 5, 2]
_qsl = [0]
for q in QUERY_LENS:
    _qsl.append(_qsl[-1] + q)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)
NUM_COMPUTED_TOKENS = torch.tensor(
    [3, 7, 5, 11, 0, 0], dtype=torch.int32, device=DEVICE
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


def pytorch_ref(idx_mapping, num_computed_tokens, query_start_loc):
    idx_mapping = idx_mapping.cpu()
    nct = num_computed_tokens.cpu().clone()
    qsl = query_start_loc.cpu()
    num_reqs = idx_mapping.shape[0]
    for b in range(num_reqs):
        rs = int(idx_mapping[b].item())
        query_len = int(qsl[b + 1].item()) - int(qsl[b].item())
        nct[rs] = int(nct[rs].item()) + query_len
    return nct


def kernel_impl(idx_mapping, num_computed_tokens, query_start_loc):
    num_computed_tokens = num_computed_tokens.clone()
    post_update_num_computed_tokens(idx_mapping, num_computed_tokens, query_start_loc)
    return num_computed_tokens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(IDX_MAPPING, NUM_COMPUTED_TOKENS, QUERY_START_LOC)
        ker = kernel_impl(IDX_MAPPING, NUM_COMPUTED_TOKENS, QUERY_START_LOC)
        ker = ker.cpu()
        mism = int((ker != ref).sum().item())
        assert mism == 0, f"num_computed_tokens mismatch count={mism}"
        stats = {
            "shape": tuple(NUM_COMPUTED_TOKENS.shape),
            "dtype": str(NUM_COMPUTED_TOKENS.dtype),
            "device": str(NUM_COMPUTED_TOKENS.device),
            "mismatch": mism,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(IDX_MAPPING, NUM_COMPUTED_TOKENS, QUERY_START_LOC))
        kern_stats = _bench(lambda: kernel_impl(IDX_MAPPING, NUM_COMPUTED_TOKENS, QUERY_START_LOC))
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
            "Kernel: _post_update_num_computed_tokens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"shape: {stats['shape']} dtype {stats['dtype']}\n")
            lines.append(f"device: {stats['device']}\n")
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
