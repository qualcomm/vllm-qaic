"""
Standalone QAIC validation for `_fill_logprob_token_ids_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/logprob.py
  - _fill_logprob_token_ids_kernel (build per-token logprob token-id matrix +
    validity mask)

Per batch row:
  - column 0 is always the sampled token id and is always valid.
  - columns 1.. are either the per-request custom logprob token ids (when the
    request specified them, num_custom > 0) or the top-k indices otherwise.
  - the validity mask marks which of those extra columns are populated.

Integer/index kernel -> exact equality on token ids and on the validity mask.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu.sample.logprob import _fill_logprob_token_ids_kernel

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fill_logprob_token_ids_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/logprob.py"
KERNEL_NAME = "_fill_logprob_token_ids_kernel"

DEVICE = "qaic"
BATCH_SIZE = 4
MAX_NUM_REQS = 4
NUM_TOPK = 3
MAX_PER_REQ = 4
MAX_LOGPROB_TOKEN_IDS = 8
NUM_COLS = max(NUM_TOPK, MAX_PER_REQ)
PADDED_COLS = triton.next_power_of_2(NUM_COLS)
VOCAB_SIZE = 128

torch.manual_seed(42)
SAMPLED = torch.randint(
    0, VOCAB_SIZE, (BATCH_SIZE,), dtype=torch.int64, device=DEVICE
)
TOPK = torch.randint(
    0, VOCAB_SIZE, (BATCH_SIZE, NUM_TOPK), dtype=torch.int32, device=DEVICE
)
EXPANDED_IDX_MAPPING = torch.arange(BATCH_SIZE, dtype=torch.int32, device=DEVICE)
# Requests 1 and 3 specify custom logprob token ids; 0 and 2 use topk.
NUM_PER_REQ = torch.tensor([0, 2, 0, 4], dtype=torch.int32, device=DEVICE)
PER_REQ_TOKEN_IDS = torch.randint(
    0, VOCAB_SIZE, (MAX_NUM_REQS, MAX_LOGPROB_TOKEN_IDS), dtype=torch.int32, device=DEVICE
)


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
        f"Kernel: {KERNEL_NAME}\n",
        f"Kernel file: {KERNEL_FILE_PATH}\n",
        f"Device target: QAIC (device='{DEVICE}')\n",
        f"Status: {status}\n\n",
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
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


def pytorch_ref():
    sampled = SAMPLED.cpu()
    topk = TOPK.cpu()
    mapping = EXPANDED_IDX_MAPPING.cpu()
    num_per_req = NUM_PER_REQ.cpu()
    per_req = PER_REQ_TOKEN_IDS.cpu()

    out = torch.zeros(BATCH_SIZE, 1 + NUM_COLS, dtype=torch.int64)
    valid = torch.zeros(BATCH_SIZE, 1 + NUM_COLS, dtype=torch.bool)
    for b in range(BATCH_SIZE):
        out[b, 0] = sampled[b]
        valid[b, 0] = True
        req = int(mapping[b].item())
        num_custom = int(num_per_req[req].item())
        for col in range(NUM_COLS):
            if num_custom > 0:
                if col < num_custom:
                    out[b, 1 + col] = int(per_req[req, col].item())
                    valid[b, 1 + col] = True
            else:
                if col < NUM_TOPK:
                    out[b, 1 + col] = int(topk[b, col].item())
                    valid[b, 1 + col] = True
    return out, valid


def kernel_impl():
    out = torch.zeros(BATCH_SIZE, 1 + NUM_COLS, dtype=torch.int64, device=DEVICE)
    valid = torch.zeros_like(out, dtype=torch.bool)
    _fill_logprob_token_ids_kernel[(BATCH_SIZE,)](
        out,
        out.stride(0),
        valid,
        valid.stride(0),
        SAMPLED,
        TOPK,
        TOPK.stride(0),
        EXPANDED_IDX_MAPPING,
        NUM_PER_REQ,
        PER_REQ_TOKEN_IDS,
        PER_REQ_TOKEN_IDS.stride(0),
        NUM_TOPK=NUM_TOPK,
        PADDED_COLS=PADDED_COLS,
    )
    return out, valid


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
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ids, ref_valid = pytorch_ref()
        out_ids, out_valid = kernel_impl()
        ids_cpu = out_ids.cpu().to(torch.int64)
        valid_cpu = out_valid.cpu().to(torch.bool)

        # Only compare token ids at valid positions (invalid entries are
        # left at their zero-init value in both ref and kernel).
        ids_mismatch = int(((ids_cpu != ref_ids) & ref_valid).sum().item())
        valid_mismatch = int((valid_cpu != ref_valid).sum().item())
        assert valid_mismatch == 0, f"validity mask mismatch count={valid_mismatch}"
        assert ids_mismatch == 0, f"token id mismatch count={ids_mismatch}"

        stats = {
            "output_shape": tuple(out_ids.shape),
            "output_dtype": f"{out_ids.dtype}/{out_valid.dtype}",
            "device": str(out_ids.device),
            "num_topk": NUM_TOPK,
            "num_cols": NUM_COLS,
            "num_per_req": NUM_PER_REQ.cpu().tolist(),
            "ids_mismatch_count": ids_mismatch,
            "valid_mismatch_count": valid_mismatch,
            "max_abs_diff": 0,
            "comparison": "exact integer equality (ids at valid positions + mask)",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
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
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
