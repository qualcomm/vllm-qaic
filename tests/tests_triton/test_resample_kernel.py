"""
Standalone QAIC validation for `_resample_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - _resample_kernel  (grid = (num_reqs, resample_num_blocks)).

The kernel resamples the rejected/bonus token for each request. It builds a
residual distribution and calls `gumbel_block_argmax` to pick a per-block
argmax + max, which are later folded by _insert_resampled_kernel.

RNG NOTE: the general path perturbs logits with Gumbel noise (tl_rand64), which
is infeasible to reproduce exactly in pure PyTorch. However `gumbel_block_argmax`
adds noise ONLY when temperature != 0. We therefore validate the deterministic
GREEDY + BONUS-token path: temperature == 0 with the resample position being the
bonus token (resample_token_idx == end - 1). On this path:
  - is_bonus is True, so residual_logits = target_logits (no draft mixing);
  - gumbel_block_argmax sees temp == 0 -> NO noise -> plain per-block argmax/max.
This makes the output a deterministic per-block argmax over the target logits,
which we reproduce exactly in pure PyTorch. HAS_DRAFT_LOGITS=False, USE_FP64=False.

We compare resampled_local_argmax (exact int) and resampled_local_max
(assert_close float).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_resample_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    _resample_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
MAX_NUM_REQS = 4
NUM_LOGITS_PER_REQ = [3, 4, 3]
_cu = [0]
for _n in NUM_LOGITS_PER_REQ:
    _cu.append(_cu[-1] + _n)
CU_NUM_LOGITS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
NUM_LOGITS = _cu[-1]

VOCAB = 200
RESAMPLE_BLOCK_SIZE = 64
RESAMPLE_NUM_BLOCKS = triton.cdiv(VOCAB, RESAMPLE_BLOCK_SIZE)

TARGET_LOGITS = torch.randn(NUM_LOGITS, VOCAB, dtype=torch.float32, device=DEVICE) * 3.0

# rejected_step = n - 1 => resample_token_idx = start + (n-1) = end - 1 = bonus.
REJECTED_STEP = torch.tensor(
    [n - 1 for n in NUM_LOGITS_PER_REQ], dtype=torch.int32, device=DEVICE
)
# expanded_idx_mapping: token index -> request state index.
EXPANDED_IDX_MAPPING = torch.zeros(NUM_LOGITS, dtype=torch.int32, device=DEVICE)
for _r in range(NUM_REQS):
    EXPANDED_IDX_MAPPING[_cu[_r]:_cu[_r + 1]] = _r

TEMPERATURE = torch.zeros(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)  # greedy
SEED = torch.zeros(MAX_NUM_REQS, dtype=torch.int64, device=DEVICE)
POS = torch.arange(NUM_LOGITS, dtype=torch.int64, device=DEVICE)
DRAFT_SAMPLED = torch.randint(0, VOCAB, (NUM_LOGITS,), dtype=torch.int64, device=DEVICE)

# Unused-on-this-path tensors (still required by the signature).
TARGET_REJECTED_LSE = torch.zeros(NUM_REQS, dtype=torch.float32, device=DEVICE)
DRAFT_REJECTED_LSE = torch.zeros(NUM_REQS, dtype=torch.float32, device=DEVICE)
DRAFT_LOGITS = TARGET_LOGITS.new_empty(1, 1, 1)


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
    cu = CU_NUM_LOGITS.cpu()
    rej = REJECTED_STEP.cpu()

    r_argmax = torch.zeros(NUM_REQS, RESAMPLE_NUM_BLOCKS, dtype=torch.int64)
    r_max = torch.full(
        (NUM_REQS, RESAMPLE_NUM_BLOCKS), float("-inf"), dtype=torch.float32
    )
    for r in range(NUM_REQS):
        start = int(cu[r].item())
        resample_token_idx = start + int(rej[r].item())
        row = tgt[resample_token_idx]
        for b in range(RESAMPLE_NUM_BLOCKS):
            lo = b * RESAMPLE_BLOCK_SIZE
            hi = min(lo + RESAMPLE_BLOCK_SIZE, VOCAB)
            seg = row[lo:hi]
            val, idx = seg.max(dim=0)
            r_argmax[r, b] = lo + int(idx.item())
            r_max[r, b] = val
    return r_argmax, r_max


def kernel_impl():
    r_argmax = torch.empty(
        NUM_REQS, RESAMPLE_NUM_BLOCKS, dtype=torch.int64, device=DEVICE
    )
    r_max = torch.empty(
        NUM_REQS, RESAMPLE_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
    )
    _resample_kernel[(NUM_REQS, RESAMPLE_NUM_BLOCKS)](
        r_argmax,
        r_argmax.stride(0),
        r_max,
        r_max.stride(0),
        TARGET_LOGITS,
        TARGET_LOGITS.stride(0),
        TARGET_REJECTED_LSE,
        DRAFT_LOGITS,
        DRAFT_LOGITS.stride(0),
        DRAFT_LOGITS.stride(1),
        DRAFT_REJECTED_LSE,
        REJECTED_STEP,
        CU_NUM_LOGITS,
        EXPANDED_IDX_MAPPING,
        DRAFT_SAMPLED,
        TEMPERATURE,
        SEED,
        POS,
        VOCAB,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=False,
        USE_FP64=False,
    )
    return r_argmax, r_max


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        r_arg, r_mx = pytorch_ref()
        k_arg, k_mx = kernel_impl()

        k_arg_c = k_arg.cpu()
        k_mx_c = k_mx.cpu()
        assert torch.equal(k_arg_c, r_arg), "resampled_local_argmax mismatch"
        torch.testing.assert_close(k_mx_c.float(), r_mx.float(), rtol=1e-3, atol=1e-3)

        diff = (k_mx_c.float() - r_mx.float()).abs()
        stats = {
            "input_shape": tuple(TARGET_LOGITS.shape),
            "output_shape": tuple(k_arg.shape),
            "in_dtype": str(TARGET_LOGITS.dtype),
            "out_dtype": str(k_mx.dtype),
            "device": str(TARGET_LOGITS.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
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
            "Kernel: _resample_kernel (greedy + bonus, no-noise path)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
            "RNG note: gumbel_block_argmax adds noise only when temperature != 0. "
            "Validated the greedy (temp==0) bonus-token path where the residual is "
            "the raw target logits and the result is a deterministic per-block "
            "argmax/max -> exact/assert_close compare, no philox needed.\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape (target_logits): {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape (resampled_local_argmax): {stats['output_shape']}\n")
            lines.append(f"- out dtype (resampled_local_max): {stats['out_dtype']}\n")
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
