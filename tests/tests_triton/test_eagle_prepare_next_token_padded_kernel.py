"""
Standalone QAIC validation for `eagle_prepare_next_token_padded_kernel`.

Source under test:
vllm/v1/spec_decode/utils.py
  - eagle_prepare_next_token_padded_kernel  (Eagle prepare_next_token_ids_padded)

Grid is (num_reqs,). For each request:
  - If discarded: next_token = backup_next_token, valid_count = 0.
  - Else: over the row sampled_token_ids[req, :num_sampled_tokens_per_req],
    a token is "valid" if it is != -1 and < vocab_size. valid_count = number of
    valid tokens. If valid_count > 0, next_token = the LAST valid token in the
    row; otherwise next_token = backup_next_token.

Config tested: 5 requests, num_sampled_tokens_per_req=4 (BLOCK_SIZE_TOKENS=4),
covering: trailing rejects (-1), a discarded request, all-rejected fallback to
backup, and out-of-vocab entries. No python launcher exists, so we launch the
@triton.jit kernel directly. Integer outputs -> EXACT (torch.equal) comparison.
Reference: pure PyTorch replication of the valid-count / last-valid selection.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_eagle_prepare_next_token_padded_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/spec_decode/utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.spec_decode.utils import (  # noqa: E402
    eagle_prepare_next_token_padded_kernel,
    next_power_of_2,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 5
NUM_SAMPLED_PER_REQ = 4
VOCAB_SIZE = 1000
BLOCK_SIZE_TOKENS = next_power_of_2(NUM_SAMPLED_PER_REQ)

# Rows: last valid token, trailing rejects, discarded, all rejected, oov entry.
SAMPLED_TOKEN_IDS = torch.tensor(
    [
        [10, 20, 30, 40],   # all valid -> last = 40
        [11, 22, -1, -1],   # trailing rejects -> last valid = 22
        [50, 60, 70, 80],   # discarded -> backup used
        [-1, -1, -1, -1],   # all rejected -> backup used
        [15, 5000, 25, -1],  # 5000 >= vocab -> invalid; last valid = 25
    ],
    dtype=torch.int32,
    device=DEVICE,
)
DISCARD_REQUEST_MASK = torch.tensor(
    [False, False, True, False, False], dtype=torch.bool, device=DEVICE
)
BACKUP_NEXT_TOKEN_IDS = torch.tensor(
    [900, 901, 902, 903, 904], dtype=torch.int32, device=DEVICE
)


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


def pytorch_ref():
    sampled = SAMPLED_TOKEN_IDS.cpu()
    discard = DISCARD_REQUEST_MASK.cpu().tolist()
    backup = BACKUP_NEXT_TOKEN_IDS.cpu().tolist()

    next_token_ids = torch.zeros(NUM_REQS, dtype=torch.int32)
    valid_count = torch.zeros(NUM_REQS, dtype=torch.int32)
    for i in range(NUM_REQS):
        if discard[i]:
            next_token_ids[i] = backup[i]
            valid_count[i] = 0
            continue
        row = sampled[i, :NUM_SAMPLED_PER_REQ].tolist()
        valid_positions = [
            j for j, t in enumerate(row) if t != -1 and t < VOCAB_SIZE
        ]
        valid_count[i] = len(valid_positions)
        if valid_positions:
            next_token_ids[i] = row[valid_positions[-1]]
        else:
            next_token_ids[i] = backup[i]
    return next_token_ids, valid_count


def kernel_impl():
    next_token_ids = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    valid_count = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    eagle_prepare_next_token_padded_kernel[(NUM_REQS,)](
        SAMPLED_TOKEN_IDS,
        DISCARD_REQUEST_MASK,
        BACKUP_NEXT_TOKEN_IDS,
        next_token_ids,
        valid_count,
        VOCAB_SIZE,
        NUM_SAMPLED_PER_REQ,
        NUM_REQS,
        SAMPLED_TOKEN_IDS.stride(0),
        BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
    )
    return next_token_ids, valid_count


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        kern = kernel_impl()
        names = ["next_token_ids", "valid_sampled_tokens_count"]
        for name, r, k in zip(names, ref, kern):
            kc = k.cpu().to(r.dtype)
            assert torch.equal(kc, r), (
                f"{name} mismatch: ref={r.tolist()} kern={kc.tolist()}"
            )

        stats = {
            "input_shape": tuple(SAMPLED_TOKEN_IDS.shape),
            "output_shape": tuple(kern[0].shape),
            "in_dtype": str(SAMPLED_TOKEN_IDS.dtype),
            "out_dtype": str(kern[0].dtype),
            "device": str(SAMPLED_TOKEN_IDS.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0,
        }
        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
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
            "Kernel: eagle_prepare_next_token_padded_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape: {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append("- relative_error: 0.0 (exact integer match)\n")
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
    sys.exit(0 if main() == "SUCCESS" else 1)
