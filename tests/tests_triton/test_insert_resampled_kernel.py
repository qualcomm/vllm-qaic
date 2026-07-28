"""
Standalone QAIC validation for `_insert_resampled_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - _insert_resampled_kernel  (grid = (num_reqs,)).

For each request it:
  1. reads num_sampled[req] and start = cu_num_logits[req];
  2. increments num_sampled[req] by 1;
  3. resample_token_idx = start + num_sampled; is_bonus = (== end - 1);
  4. if temp == 0 and not is_bonus: return (target argmax already in place);
     else: pick the block with the max resampled_local_max, read the matching
     resampled_local_argmax entry, and store it into sampled[req, num_sampled].

This kernel contains NO RNG. Config tested: NON-GREEDY requests (temperature = 1)
so every request inserts. We compare the final `sampled` matrix and the mutated
`num_sampled` counter against a pure-PyTorch reference EXACTLY (integer equal).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_insert_resampled_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    _insert_resampled_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
NUM_SPEC_STEPS = 3
MAX_NUM_REQS = 4
NUM_LOGITS_PER_REQ = [4, 3, 4]
NUM_SAMPLED_INIT = [1, 2, 3]  # current sampled count per request (< spec+1)

_cu = [0]
for _n in NUM_LOGITS_PER_REQ:
    _cu.append(_cu[-1] + _n)
CU_NUM_LOGITS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
NUM_LOGITS = _cu[-1]

VOCAB = 200
RESAMPLE_BLOCK_SIZE = 64
RESAMPLE_NUM_BLOCKS = triton.cdiv(VOCAB, RESAMPLE_BLOCK_SIZE)
PADDED_RESAMPLE_NUM_BLOCKS = triton.next_power_of_2(RESAMPLE_NUM_BLOCKS)

RESAMPLED_LOCAL_MAX = torch.randn(
    NUM_REQS, RESAMPLE_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
)
RESAMPLED_LOCAL_ARGMAX = torch.randint(
    0, VOCAB, (NUM_REQS, RESAMPLE_NUM_BLOCKS), dtype=torch.int64, device=DEVICE
)
NUM_SAMPLED_INIT_T = torch.tensor(NUM_SAMPLED_INIT, dtype=torch.int32, device=DEVICE)
# Non-greedy: every request inserts regardless of bonus/non-bonus.
TEMPERATURE = torch.ones(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)
# expanded_idx_mapping: token index -> request state index (identity-ish).
EXPANDED_IDX_MAPPING = torch.zeros(NUM_LOGITS, dtype=torch.int32, device=DEVICE)
for _r in range(NUM_REQS):
    EXPANDED_IDX_MAPPING[_cu[_r]:_cu[_r + 1]] = _r
SENTINEL = -1


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def _bench(fn, warmup=3, iters=10):
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
    rmax = RESAMPLED_LOCAL_MAX.cpu().float()
    rarg = RESAMPLED_LOCAL_ARGMAX.cpu()
    cu = CU_NUM_LOGITS.cpu()
    num_sampled = NUM_SAMPLED_INIT_T.cpu().clone()

    sampled = torch.full(
        (NUM_REQS, NUM_SPEC_STEPS + 1), SENTINEL, dtype=torch.int64
    )
    for r in range(NUM_REQS):
        ns = int(num_sampled[r].item())
        start = int(cu[r].item())
        end = int(cu[r + 1].item())
        resample_token_idx = start + ns
        is_bonus = resample_token_idx == end - 1
        num_sampled[r] = ns + 1
        # temp == 1 (non-greedy) so we always insert.
        best_block = int(torch.argmax(rmax[r, :RESAMPLE_NUM_BLOCKS]).item())
        sampled[r, ns] = rarg[r, best_block]
        _ = is_bonus
    return sampled, num_sampled


def kernel_impl():
    sampled = torch.full(
        (NUM_REQS, NUM_SPEC_STEPS + 1), SENTINEL, dtype=torch.int64, device=DEVICE
    )
    num_sampled = NUM_SAMPLED_INIT_T.clone()
    _insert_resampled_kernel[(NUM_REQS,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        RESAMPLED_LOCAL_ARGMAX,
        RESAMPLED_LOCAL_ARGMAX.stride(0),
        RESAMPLED_LOCAL_MAX,
        RESAMPLED_LOCAL_MAX.stride(0),
        RESAMPLE_NUM_BLOCKS,
        CU_NUM_LOGITS,
        EXPANDED_IDX_MAPPING,
        TEMPERATURE,
        PADDED_RESAMPLE_NUM_BLOCKS=PADDED_RESAMPLE_NUM_BLOCKS,
    )
    return sampled, num_sampled


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        r_sampled, r_ns = pytorch_ref()
        k_sampled, k_ns = kernel_impl()

        k_sampled_c = k_sampled.cpu()
        k_ns_c = k_ns.cpu()
        assert torch.equal(k_sampled_c, r_sampled), "sampled mismatch"
        assert torch.equal(k_ns_c, r_ns), "num_sampled mismatch"

        stats = {
            "input_shape": tuple(RESAMPLED_LOCAL_ARGMAX.shape),
            "output_shape": tuple(k_sampled.shape),
            "in_dtype": str(RESAMPLED_LOCAL_ARGMAX.dtype),
            "out_dtype": str(k_sampled.dtype),
            "device": str(RESAMPLED_LOCAL_ARGMAX.device),
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
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
            "Kernel: _insert_resampled_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape (resampled_local_argmax): {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape (sampled): {stats['output_shape']}\n")
            lines.append(f"- out dtype: {stats['out_dtype']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
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
