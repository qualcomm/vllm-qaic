"""
Standalone QAIC validation for `_compute_block_stats_kernel`.

Source under test:
vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py
  - _compute_block_stats_kernel  (grid = (num_logits, vocab_num_blocks)).

Per (logit_idx, block_idx) the kernel first reads the draft step for the logit
(expanded_local_pos). If it is a bonus token (step >= num_speculative_steps) it
returns. Otherwise it reads temperature for the request:
  - temp == 0.0 (greedy): stores target block argmax + block max.
  - temp != 0.0 (non-greedy): stores target block max + block sumexp via
    _compute_block_max_and_sumexp, and (if HAS_DRAFT_LOGITS) the same for the
    draft logits at [req_state_idx, draft_step_idx].

Config tested: NON-GREEDY path (all temperatures = 1.0) with HAS_DRAFT_LOGITS=True
and all logits non-bonus. This exercises the log-sum-exp partial reductions for
both target and draft. Fully deterministic (NO RNG). We compare the produced
target_local_max / target_local_sumexp / draft_local_max / draft_local_sumexp
against a pure-PyTorch reference (float assert_close).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_compute_block_stats_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (  # noqa: E402
    _compute_block_stats_kernel,
)

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_LOGITS = 4
MAX_NUM_REQS = 2
NUM_SPEC_STEPS = 2
VOCAB = 200
BLOCK_SIZE = 64
VOCAB_NUM_BLOCKS = triton.cdiv(VOCAB, BLOCK_SIZE)

TARGET_LOGITS = torch.randn(NUM_LOGITS, VOCAB, dtype=torch.float32, device=DEVICE) * 3.0
DRAFT_LOGITS = (
    torch.randn(
        MAX_NUM_REQS, NUM_SPEC_STEPS, VOCAB, dtype=torch.float32, device=DEVICE
    )
    * 3.0
)
# logit -> request state index, and logit -> draft step (all non-bonus: < NUM_SPEC_STEPS)
EXPANDED_IDX_MAPPING = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=DEVICE)
EXPANDED_LOCAL_POS = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
# Non-greedy path for all requests.
TEMPERATURE = torch.ones(MAX_NUM_REQS, dtype=torch.float32, device=DEVICE)


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


def _block_max_sumexp(row):
    """Per-block (max, sumexp) partials matching _compute_block_max_and_sumexp."""
    bmax = torch.empty(VOCAB_NUM_BLOCKS, dtype=torch.float32)
    bsum = torch.empty(VOCAB_NUM_BLOCKS, dtype=torch.float32)
    for b in range(VOCAB_NUM_BLOCKS):
        lo = b * BLOCK_SIZE
        hi = min(lo + BLOCK_SIZE, VOCAB)
        seg = row[lo:hi]
        m = seg.max()
        bmax[b] = m
        bsum[b] = torch.exp(seg - m).sum()
    return bmax, bsum


def pytorch_ref():
    tgt = TARGET_LOGITS.cpu().float()
    drf = DRAFT_LOGITS.cpu().float()
    eim = EXPANDED_IDX_MAPPING.cpu()
    elp = EXPANDED_LOCAL_POS.cpu()

    t_max = torch.zeros(NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32)
    t_sum = torch.zeros(NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32)
    d_max = torch.zeros(NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32)
    d_sum = torch.zeros(NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32)
    for li in range(NUM_LOGITS):
        bm, bs = _block_max_sumexp(tgt[li])
        t_max[li] = bm
        t_sum[li] = bs
        req = int(eim[li].item())
        step = int(elp[li].item())
        dbm, dbs = _block_max_sumexp(drf[req, step])
        d_max[li] = dbm
        d_sum[li] = dbs
    return t_max, t_sum, d_max, d_sum


def kernel_impl():
    target_local_argmax = torch.empty(
        NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.int64, device=DEVICE
    )
    target_local_max = torch.zeros(
        NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
    )
    target_local_sumexp = torch.zeros(
        NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
    )
    draft_local_max = torch.zeros(
        NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
    )
    draft_local_sumexp = torch.zeros(
        NUM_LOGITS, VOCAB_NUM_BLOCKS, dtype=torch.float32, device=DEVICE
    )
    _compute_block_stats_kernel[(NUM_LOGITS, VOCAB_NUM_BLOCKS)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        TARGET_LOGITS,
        TARGET_LOGITS.stride(0),
        DRAFT_LOGITS,
        DRAFT_LOGITS.stride(0),
        DRAFT_LOGITS.stride(1),
        EXPANDED_IDX_MAPPING,
        EXPANDED_LOCAL_POS,
        TEMPERATURE,
        VOCAB,
        NUM_SPEC_STEPS,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_DRAFT_LOGITS=True,
    )
    return target_local_max, target_local_sumexp, draft_local_max, draft_local_sumexp


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        r_tmax, r_tsum, r_dmax, r_dsum = pytorch_ref()
        k_tmax, k_tsum, k_dmax, k_dsum = kernel_impl()

        k = [t.cpu().float() for t in (k_tmax, k_tsum, k_dmax, k_dsum)]
        r = [t.cpu().float() for t in (r_tmax, r_tsum, r_dmax, r_dsum)]
        for kt, rt, name in zip(
            k, r, ["target_max", "target_sumexp", "draft_max", "draft_sumexp"]
        ):
            torch.testing.assert_close(kt, rt, rtol=1e-3, atol=1e-3, msg=f"{name} mismatch")

        diff = torch.cat([(kt - rt).abs().reshape(-1) for kt, rt in zip(k, r)])
        stats = {
            "input_shape": tuple(TARGET_LOGITS.shape),
            "output_shape": tuple(k_tmax.shape),
            "in_dtype": str(TARGET_LOGITS.dtype),
            "out_dtype": str(k_tmax.dtype),
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
            "Kernel: _compute_block_stats_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- input shape (target_logits): {stats['input_shape']}\n")
            lines.append(f"- in dtype: {stats['in_dtype']}\n")
            lines.append(f"- device: {stats['device']}\n\n")
            lines.append("Output:\n")
            lines.append(f"- output shape (per-block stats): {stats['output_shape']}\n")
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
