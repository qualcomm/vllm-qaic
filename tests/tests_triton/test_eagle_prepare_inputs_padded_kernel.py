"""
Standalone QAIC validation for `eagle_prepare_inputs_padded_kernel`.

Source under test:
vllm/v1/spec_decode/utils.py
  - eagle_prepare_inputs_padded_kernel  (Eagle prepare_input_padded)

Grid is (num_reqs,). For each request:
  num_draft_tokens   = cu_num_draft_tokens[i] - cu_num_draft_tokens[i-1]
                       (== cu_num_draft_tokens[0] for i == 0; inclusive cumsum)
  num_rejected       = num_draft_tokens + 1 - valid_sampled_tokens_count
                       (0 if num_draft_tokens == 0)
  q_last_tok_idx     = query_start_loc_gpu[i + 1] - 1
  index_to_sample    = q_last_tok_idx - num_rejected
Outputs token_indices_to_sample and num_rejected_tokens.

Config tested: 4 requests with a mix of draft counts (including 0) and reject
counts. No python launcher exists, so we launch the @triton.jit kernel directly.
Integer outputs -> EXACT (torch.equal) comparison.
Reference: pure PyTorch replication of the arithmetic.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_eagle_prepare_inputs_padded_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/spec_decode/utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.spec_decode.utils import (  # noqa: E402
    eagle_prepare_inputs_padded_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 4
# Inclusive cumulative sum of per-request draft tokens: [3, 0, 2, 4] -> [3,3,5,9]
NUM_DRAFT_PER_REQ = [3, 0, 2, 4]
_cu = []
_acc = 0
for d in NUM_DRAFT_PER_REQ:
    _acc += d
    _cu.append(_acc)
CU_NUM_DRAFT_TOKENS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
# valid sampled tokens count (1 + accepted). Kept <= num_draft+1 where relevant.
VALID_SAMPLED_TOKENS_COUNT = torch.tensor(
    [2, 1, 3, 1], dtype=torch.int32, device=DEVICE
)
# query_start_loc has num_reqs + 1 entries (padded query lengths per request).
QUERY_START_LOC = torch.tensor([0, 4, 8, 11, 16], dtype=torch.int32, device=DEVICE)


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
    cu = CU_NUM_DRAFT_TOKENS.cpu().tolist()
    valid = VALID_SAMPLED_TOKENS_COUNT.cpu().tolist()
    qsl = QUERY_START_LOC.cpu().tolist()

    idx_to_sample = torch.zeros(NUM_REQS, dtype=torch.int32)
    num_rejected = torch.zeros(NUM_REQS, dtype=torch.int32)
    for i in range(NUM_REQS):
        num_draft = cu[i] if i == 0 else cu[i] - cu[i - 1]
        nr = num_draft + 1 - valid[i]
        nr = nr if num_draft > 0 else 0
        q_last = qsl[i + 1] - 1
        idx_to_sample[i] = q_last - nr
        num_rejected[i] = nr
    return idx_to_sample, num_rejected


def kernel_impl():
    idx_to_sample = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    num_rejected = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    eagle_prepare_inputs_padded_kernel[(NUM_REQS,)](
        CU_NUM_DRAFT_TOKENS,
        VALID_SAMPLED_TOKENS_COUNT,
        QUERY_START_LOC,
        idx_to_sample,
        num_rejected,
        NUM_REQS,
    )
    return idx_to_sample, num_rejected


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        kern = kernel_impl()
        names = ["token_indices_to_sample", "num_rejected_tokens"]
        for name, r, k in zip(names, ref, kern):
            kc = k.cpu()
            assert torch.equal(kc, r), (
                f"{name} mismatch: ref={r.tolist()} kern={kc.tolist()}"
            )

        stats = {
            "input_shape": tuple(CU_NUM_DRAFT_TOKENS.shape),
            "output_shape": tuple(kern[0].shape),
            "in_dtype": str(CU_NUM_DRAFT_TOKENS.dtype),
            "out_dtype": str(kern[0].dtype),
            "device": str(CU_NUM_DRAFT_TOKENS.device),
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
            "Kernel: eagle_prepare_inputs_padded_kernel\n",
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
