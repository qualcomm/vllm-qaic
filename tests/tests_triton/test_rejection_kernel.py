"""
Standalone QAIC validation for `_rejection_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - _rejection_kernel  (grid = (num_reqs,), num_warps=1).

The kernel walks each request's draft tokens and accepts them up to the first
rejection, storing the accepted/replacement tokens into `sampled` and the number
of accepted steps into `rejected_steps`.

RNG NOTE: the general (random) path uses tl_rand64 / log(u), but the GREEDY,
NON-SYNTHETIC path (temperature == 0, SYNTHETIC_MODE=False) is fully
DETERMINISTIC: a draft token is accepted iff it equals the target argmax, and on
rejection the target argmax is stored. No random draw is consumed on this path.
We therefore validate the greedy path exactly.

Config tested: temperature == 0 for all requests, HAS_DRAFT_LOGITS=False,
SYNTHETIC_MODE=False. The precomputed target block argmax/max (normally produced
by _compute_block_stats_kernel) are built in pure PyTorch and passed in. We
compare the `sampled` matrix (over written entries) and `rejected_steps` counter
against a pure-PyTorch reference EXACTLY.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_rejection_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    _rejection_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
NUM_SPEC_STEPS = 3
MAX_NUM_REQS = 4
# num_logits per request = num draft tokens + 1 bonus.
NUM_LOGITS_PER_REQ = [4, 3, 4]
_cu = [0]
for _n in NUM_LOGITS_PER_REQ:
    _cu.append(_cu[-1] + _n)
CU_NUM_LOGITS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
NUM_LOGITS = _cu[-1]

VOCAB = 200
VOCAB_BLOCK_SIZE = 64
VOCAB_NUM_BLOCKS = triton.cdiv(VOCAB, VOCAB_BLOCK_SIZE)
PADDED_VOCAB_NUM_BLOCKS = triton.next_power_of_2(VOCAB_NUM_BLOCKS)
SENTINEL = -1

TARGET_LOGITS = torch.randn(NUM_LOGITS, VOCAB, dtype=torch.float32, device=DEVICE) * 3.0

# Precompute the target block-level argmax + max exactly as
# _compute_block_stats_kernel would in the greedy branch.
_target_local_argmax = torch.empty(
    NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.int64, device=DEVICE
)
_target_local_max = torch.full(
    (NUM_LOGITS, VOCAB_NUM_BLOCKS), float("-inf"), dtype=torch.float32, device=DEVICE
)
for _li in range(NUM_LOGITS):
    for _b in range(VOCAB_NUM_BLOCKS):
        _lo = _b * VOCAB_BLOCK_SIZE
        _hi = min(_lo + VOCAB_BLOCK_SIZE, VOCAB)
        _seg = TARGET_LOGITS[_li, _lo:_hi]
        _val, _idx = _seg.max(dim=0)
        _target_local_argmax[_li, _b] = _lo + _idx
        _target_local_max[_li, _b] = _val
TARGET_LOCAL_ARGMAX = _target_local_argmax
TARGET_LOCAL_MAX = _target_local_max

# Make ~half the draft tokens match the target argmax so both accept and reject
# branches are exercised.
_global_argmax = TARGET_LOGITS.argmax(dim=-1)  # [NUM_LOGITS]
DRAFT_SAMPLED = torch.randint(0, VOCAB, (NUM_LOGITS,), dtype=torch.int64, device=DEVICE)
# draft_sampled is read at logit_idx+1, so seed matches accordingly.
for _r in range(NUM_REQS):
    _s = _cu[_r]
    _e = _cu[_r + 1]
    for _i in range(_e - _s - 1):
        if (_s + _i) % 2 == 0:  # deterministically force some matches
            DRAFT_SAMPLED[_s + _i + 1] = _global_argmax[_s + _i]

# Unused-on-greedy-path tensors (still required by the signature).
TARGET_LOCAL_SUMEXP = torch.zeros(
    NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
)
DRAFT_LOGITS = TARGET_LOGITS.new_empty(1, 1, 1)
DRAFT_LOCAL_MAX = torch.zeros(
    NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
)
DRAFT_LOCAL_SUMEXP = torch.zeros(
    NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
)
IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
TEMPERATURE = torch.zeros(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)  # greedy
SEED = torch.zeros(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
POS = torch.arange(NUM_LOGITS, dtype=torch.int64, device=DEVICE)


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
    tgt = TARGET_LOGITS.cpu().float()
    draft_sampled = DRAFT_SAMPLED.cpu()
    cu = CU_NUM_LOGITS.cpu()

    sampled = torch.full((NUM_REQS, NUM_SPEC_STEPS + 1), SENTINEL, dtype=torch.int64)
    rejected_steps = torch.zeros(NUM_REQS, dtype=torch.int32)
    for r in range(NUM_REQS):
        start = int(cu[r].item())
        end = int(cu[r + 1].item())
        num_tokens = end - start
        accepted = True
        rej = 0
        for i in range(num_tokens - 1):
            if accepted:
                logit_idx = start + i
                draft = int(draft_sampled[logit_idx + 1].item())
                target_argmax = int(tgt[logit_idx].argmax().item())
                accepted = accepted and (target_argmax == draft)
                sampled[r, i] = draft if accepted else target_argmax
                rej += int(accepted)
        rejected_steps[r] = rej
    return sampled, rejected_steps


def kernel_impl():
    sampled = torch.full(
        (NUM_REQS, NUM_SPEC_STEPS + 1), SENTINEL, dtype=torch.int64, device=DEVICE
    )
    rejected_steps = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
    target_rejected_lse = torch.zeros(NUM_REQS, dtype=torch.float32, device=DEVICE)
    draft_rejected_lse = torch.zeros(NUM_REQS, dtype=torch.float32, device=DEVICE)
    _rejection_kernel[(NUM_REQS,)](
        sampled,
        sampled.stride(0),
        rejected_steps,
        target_rejected_lse,
        draft_rejected_lse,
        TARGET_LOGITS,
        TARGET_LOGITS.stride(0),
        TARGET_LOCAL_ARGMAX,
        TARGET_LOCAL_ARGMAX.stride(0),
        TARGET_LOCAL_MAX,
        TARGET_LOCAL_MAX.stride(0),
        TARGET_LOCAL_SUMEXP,
        TARGET_LOCAL_SUMEXP.stride(0),
        DRAFT_SAMPLED,
        DRAFT_LOGITS,
        DRAFT_LOGITS.stride(0),
        DRAFT_LOGITS.stride(1),
        DRAFT_LOCAL_MAX,
        DRAFT_LOCAL_MAX.stride(0),
        DRAFT_LOCAL_SUMEXP,
        DRAFT_LOCAL_SUMEXP.stride(0),
        CU_NUM_LOGITS,
        IDX_MAPPING,
        TEMPERATURE,
        SEED,
        POS,
        None,  # synthetic_conditional_rates
        VOCAB_NUM_BLOCKS,
        PADDED_VOCAB_NUM_BLOCKS=PADDED_VOCAB_NUM_BLOCKS,
        HAS_DRAFT_LOGITS=False,
        SYNTHETIC_MODE=False,
        num_warps=1,
    )
    return sampled, rejected_steps


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        r_sampled, r_rej = pytorch_ref()
        k_sampled, k_rej = kernel_impl()

        k_sampled_c = k_sampled.cpu()
        k_rej_c = k_rej.cpu()
        assert torch.equal(k_sampled_c, r_sampled), "sampled mismatch"
        assert torch.equal(k_rej_c, r_rej), "rejected_steps mismatch"

        stats = {
            "input_shape": tuple(TARGET_LOGITS.shape),
            "output_shape": tuple(k_sampled.shape),
            "in_dtype": str(TARGET_LOGITS.dtype),
            "out_dtype": str(k_sampled.dtype),
            "device": str(TARGET_LOGITS.device),
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
            "Kernel: _rejection_kernel (greedy, non-synthetic path)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
            "RNG note: greedy path (temperature==0, SYNTHETIC_MODE=False) is "
            "deterministic (accept iff draft==target argmax); no random draw "
            "is consumed, so exact compare is valid.\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape (target_logits): {stats['input_shape']}\n")
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
