"""
Standalone QAIC validation for `sample_recovered_tokens_kernel`.

Source under test:
vllm/v1/sample/rejection_sampler.py
  - sample_recovered_tokens_kernel  (launched via `sample_recovered_tokens`)

For a rejected draft token, this kernel samples the "recovered" replacement
token from the residual distribution using the exponential / Gumbel-max trick.
For each (req, pos) it computes, over the vocabulary:
  prob = max(target_prob - draft_prob, 0)         (NO_DRAFT_PROBS=False)
  score = prob * inv_q          where inv_q = 1 / Exponential(1) noise
  recovered_id = argmax_v(score)
Multiplying the (un-normalized) residual probability by 1/Exp(1) and taking the
argmax is equivalent to Gumbel-max sampling from the residual distribution
(no explicit normalization is needed because argmax is scale-invariant).

RNG choice: the exponential noise `q` is generated OUTSIDE the kernel (the
launcher calls q.exponential_()); the kernel only receives inv_q = 1/q as a
tensor and is fully DETERMINISTIC given it. We therefore generate a fixed
`inv_q` (seed=42) and feed the SAME tensor to both the kernel and the PyTorch
reference, comparing recovered ids EXACTLY. No philox reproduction is needed
because no RNG happens inside the kernel. We use the standard residual path
(NO_DRAFT_PROBS=False, USE_FP64_GUMBEL=False) with BLOCK_SIZE == vocab so the
whole vocabulary is reduced in a single tile (matching tl.max tie-breaking).

Integer/index exact validation of the [num_tokens] recovered ids.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_sample_recovered_tokens_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/sample/rejection_sampler.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.sample.rejection_sampler import sample_recovered_tokens_kernel

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
NUM_REQS = 3
NUM_DRAFT_PER_REQ = [2, 3, 2]
MAX_SPEC_LEN = max(NUM_DRAFT_PER_REQ)
VOCAB = 16
BLOCK_SIZE = VOCAB  # single-tile reduction over the vocabulary
_cu = []
_acc = 0
for _n in NUM_DRAFT_PER_REQ:
    _acc += _n
    _cu.append(_acc)
CU_NUM_DRAFT_TOKENS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
TOTAL_TOKENS = _cu[-1]

DRAFT_TOKEN_IDS = torch.randint(
    0, VOCAB, (TOTAL_TOKENS,), dtype=torch.int32, device=DEVICE
)
DRAFT_PROBS = torch.rand(TOTAL_TOKENS, VOCAB, dtype=torch.float32, device=DEVICE)
DRAFT_PROBS = (DRAFT_PROBS / DRAFT_PROBS.sum(dim=-1, keepdim=True)).contiguous()
TARGET_PROBS = torch.rand(TOTAL_TOKENS, VOCAB, dtype=torch.float32, device=DEVICE)
TARGET_PROBS = (TARGET_PROBS / TARGET_PROBS.sum(dim=-1, keepdim=True)).contiguous()

# Exponential noise generated OUTSIDE the kernel, fixed by seed.
_q = torch.empty(NUM_REQS, VOCAB, dtype=torch.float32, device=DEVICE)
_q.exponential_()
INV_Q = _q.reciprocal().contiguous()


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


def pytorch_ref(cu_num_draft, draft_probs, target_probs, inv_q):
    cu = cu_num_draft.cpu()
    dprobs = draft_probs.cpu()
    tprobs = target_probs.cpu()
    iq = inv_q.cpu()
    out = torch.zeros(TOTAL_TOKENS, dtype=torch.int32)
    for r in range(NUM_REQS):
        start = 0 if r == 0 else int(cu[r - 1].item())
        end = int(cu[r].item())
        n = end - start
        for pos in range(n):
            tok = start + pos
            prob = torch.clamp(tprobs[tok] - dprobs[tok], min=0.0)
            score = prob * iq[r]
            # tl.max returns the first index of the maximum value.
            out[tok] = int(torch.argmax(score).item())
    return out


def kernel_impl(cu_num_draft, draft_token_ids, draft_probs, target_probs, inv_q):
    recovered = torch.zeros(TOTAL_TOKENS, dtype=torch.int32, device=DEVICE)
    sample_recovered_tokens_kernel[(NUM_REQS, MAX_SPEC_LEN)](
        recovered,
        cu_num_draft,
        draft_token_ids,
        draft_probs,
        target_probs,
        inv_q,
        VOCAB,
        BLOCK_SIZE,
        NO_DRAFT_PROBS=False,
        USE_FP64_GUMBEL=False,
    )
    return recovered


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(
            CU_NUM_DRAFT_TOKENS, DRAFT_PROBS, TARGET_PROBS, INV_Q
        )
        k_out = kernel_impl(
            CU_NUM_DRAFT_TOKENS, DRAFT_TOKEN_IDS, DRAFT_PROBS, TARGET_PROBS, INV_Q
        ).cpu()
        # Only positions < num_draft per request are written; compare those.
        cu = CU_NUM_DRAFT_TOKENS.cpu()
        valid = torch.zeros(TOTAL_TOKENS, dtype=torch.bool)
        for r in range(NUM_REQS):
            start = 0 if r == 0 else int(cu[r - 1].item())
            end = int(cu[r].item())
            valid[start:end] = True
        mism = int((k_out[valid] != ref_out[valid]).sum().item())
        assert mism == 0, f"recovered token id mismatch count={mism}"
        stats = {
            "input_shape": tuple(TARGET_PROBS.shape),
            "output_shape": tuple(k_out.shape),
            "dtype": str(k_out.dtype),
            "device": str(TARGET_PROBS.device),
            "mismatch": mism,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS}, {MAX_SPEC_LEN})",
        }
        pt_stats = _bench(lambda: pytorch_ref(
            CU_NUM_DRAFT_TOKENS, DRAFT_PROBS, TARGET_PROBS, INV_Q
        ))
        kern_stats = _bench(lambda: kernel_impl(
            CU_NUM_DRAFT_TOKENS, DRAFT_TOKEN_IDS, DRAFT_PROBS, TARGET_PROBS, INV_Q
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
            "Kernel: sample_recovered_tokens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
            "RNG note: exponential noise (inv_q) generated OUTSIDE the kernel; "
            "identical fixed tensor fed to both paths -> exact compare, no "
            "philox needed.\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"num_draft_per_req: {NUM_DRAFT_PER_REQ}\n")
            lines.append(f"probs shape: {stats['input_shape']}\n")
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
