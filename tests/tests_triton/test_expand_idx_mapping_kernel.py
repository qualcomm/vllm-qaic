"""
Standalone QAIC validation for `_expand_idx_mapping_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _expand_idx_mapping_kernel  (launched via `expand_idx_mapping`)

Expands the per-request (batch-level) idx_mapping into a per-logit
(token-level) mapping for multi-token / spec-decode batches. For each request
req_idx it fills the slice [cu_num_logits[req_idx], cu_num_logits[req_idx+1])
of:
  - expanded_idx_mapping with the request's req_state_idx (broadcast), and
  - expanded_local_pos with 0,1,2,... (local position within the request).

Integer-exact validation of both output buffers.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_expand_idx_mapping_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.gpu.input_batch import expand_idx_mapping

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
NUM_REQS = 4
IDX_MAPPING = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)
# Per-request number of logits (>=1 each; spec decode gives >1).
NUM_LOGITS_PER_REQ = [1, 3, 2, 4]
MAX_EXPAND_LEN = max(NUM_LOGITS_PER_REQ)
_cu = [0]
for n in NUM_LOGITS_PER_REQ:
    _cu.append(_cu[-1] + n)
CU_NUM_LOGITS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
TOTAL_NUM_LOGITS = _cu[-1]


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(idx_mapping, total_num_logits, cu_num_logits):
    idx_mapping = idx_mapping.cpu()
    cu = cu_num_logits.cpu()
    exp_idx = torch.empty(total_num_logits, dtype=idx_mapping.dtype)
    exp_pos = torch.empty(total_num_logits, dtype=torch.int32)
    num_reqs = idx_mapping.shape[0]
    for r in range(num_reqs):
        start = int(cu[r].item())
        end = int(cu[r + 1].item())
        n = end - start
        exp_idx[start:end] = int(idx_mapping[r].item())
        exp_pos[start:end] = torch.arange(n, dtype=torch.int32)
    return exp_idx, exp_pos


def kernel_impl(idx_mapping, total_num_logits, cu_num_logits, max_expand_len):
    exp_idx, exp_pos = expand_idx_mapping(
        idx_mapping, total_num_logits, cu_num_logits, max_expand_len
    )
    return exp_idx, exp_pos


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
        ref_idx, ref_pos = pytorch_ref(IDX_MAPPING, TOTAL_NUM_LOGITS, CU_NUM_LOGITS)
        k_idx, k_pos = kernel_impl(
            IDX_MAPPING, TOTAL_NUM_LOGITS, CU_NUM_LOGITS, MAX_EXPAND_LEN
        )
        k_idx = k_idx.cpu()
        k_pos = k_pos.cpu()
        mism_idx = int((k_idx != ref_idx).sum().item())
        mism_pos = int((k_pos != ref_pos).sum().item())
        assert mism_idx == 0, f"expanded_idx_mapping mismatch count={mism_idx}"
        assert mism_pos == 0, f"expanded_local_pos mismatch count={mism_pos}"
        stats = {
            "total_num_logits": TOTAL_NUM_LOGITS,
            "idx_dtype": str(k_idx.dtype),
            "pos_dtype": str(k_pos.dtype),
            "device": str(IDX_MAPPING.device),
            "mismatch_idx": mism_idx,
            "mismatch_pos": mism_pos,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(
            lambda: pytorch_ref(IDX_MAPPING, TOTAL_NUM_LOGITS, CU_NUM_LOGITS)
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                IDX_MAPPING, TOTAL_NUM_LOGITS, CU_NUM_LOGITS, MAX_EXPAND_LEN
            )
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
            "Kernel: _expand_idx_mapping_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"total_num_logits: {stats['total_num_logits']}\n")
            lines.append(f"idx dtype: {stats['idx_dtype']}  pos dtype: {stats['pos_dtype']}\n")
            lines.append(f"device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch(idx): {stats['mismatch_idx']}\n")
            lines.append(f"mismatch(pos): {stats['mismatch_pos']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
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


if __name__ == "__main__":
    main()
