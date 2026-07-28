"""
Standalone QAIC validation for `rejection_greedy_sample_kernel`.

Source under test:
vllm/v1/sample/rejection_sampler.py
  - rejection_greedy_sample_kernel  (launched via `rejection_sample`)

For greedy-sampling requests, this kernel walks the draft tokens of each
request in order, comparing each draft token to the target model's argmax.
It stores the target argmax at every position up to (and including) the first
mismatch; once a mismatch occurs no further positions are written. If every
draft token matched (no rejection) the bonus token is appended right after the
last draft position. This is DETERMINISTIC (no RNG, SYNTHETIC_MODE=False).

Integer/index exact validation of the [batch_size, max_spec_len + 1] output.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_rejection_greedy_sample_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/sample/rejection_sampler.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.sample.rejection_sampler import rejection_greedy_sample_kernel

torch.manual_seed(42)

PLACEHOLDER_TOKEN_ID = -1

# ---- Global shared inputs -------------------------------------------------
NUM_REQS = 3
NUM_DRAFT_PER_REQ = [2, 3, 2]
MAX_SPEC_LEN = max(NUM_DRAFT_PER_REQ)
VOCAB = 50
_cu = []
_acc = 0
for _n in NUM_DRAFT_PER_REQ:
    _acc += _n
    _cu.append(_acc)
CU_NUM_DRAFT_TOKENS = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
TOTAL_TOKENS = _cu[-1]

# target argmax (what the target model would greedily pick) per draft position.
TARGET_ARGMAX = torch.randint(
    0, VOCAB, (TOTAL_TOKENS,), dtype=torch.int64, device=DEVICE
)
# draft tokens: start equal to target argmax (all accept), then perturb a
# couple of positions to force mismatches (early rejection paths).
DRAFT_TOKEN_IDS = TARGET_ARGMAX.clone().to(torch.int32)
# token idx layout: req0=[0,1], req1=[2,3,4], req2=[5,6]
DRAFT_TOKEN_IDS[3] = (TARGET_ARGMAX[3].item() + 1) % VOCAB  # req1 mismatch @pos1
DRAFT_TOKEN_IDS[5] = (TARGET_ARGMAX[5].item() + 1) % VOCAB  # req2 mismatch @pos0

BONUS_TOKEN_IDS = torch.randint(
    0, VOCAB, (NUM_REQS,), dtype=torch.int32, device=DEVICE
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


def pytorch_ref(cu_num_draft, draft_token_ids, target_argmax, bonus_token_ids):
    cu = cu_num_draft.cpu()
    draft = draft_token_ids.cpu()
    targ = target_argmax.cpu().to(torch.int32)
    bonus = bonus_token_ids.cpu()
    out = torch.full(
        (NUM_REQS, MAX_SPEC_LEN + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32
    )
    for r in range(NUM_REQS):
        start = 0 if r == 0 else int(cu[r - 1].item())
        end = int(cu[r].item())
        n = end - start
        rejected = False
        for pos in range(n):
            if not rejected:
                d = int(draft[start + pos].item())
                t = int(targ[start + pos].item())
                token_id = t  # greedy always stores the target argmax
                rejected = d != t
                out[r, pos] = token_id
        if not rejected:
            out[r, n] = int(bonus[r].item())
    return out


def kernel_impl(cu_num_draft, draft_token_ids, target_argmax, bonus_token_ids):
    out = torch.full(
        (NUM_REQS, MAX_SPEC_LEN + 1),
        PLACEHOLDER_TOKEN_ID,
        dtype=torch.int32,
        device=DEVICE,
    )
    rejection_greedy_sample_kernel[(NUM_REQS,)](
        out,
        cu_num_draft,
        draft_token_ids,
        target_argmax,
        bonus_token_ids,
        None,  # is_greedy_ptr -> None means all greedy
        MAX_SPEC_LEN,
        None,  # uniform_probs (synthetic only)
        None,  # synthetic_conditional_rates
        SYNTHETIC_MODE=False,
    )
    return out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(
            CU_NUM_DRAFT_TOKENS, DRAFT_TOKEN_IDS, TARGET_ARGMAX, BONUS_TOKEN_IDS
        )
        k_out = kernel_impl(
            CU_NUM_DRAFT_TOKENS, DRAFT_TOKEN_IDS, TARGET_ARGMAX, BONUS_TOKEN_IDS
        ).cpu()
        mism = int((k_out != ref_out).sum().item())
        assert mism == 0, f"output_token_ids mismatch count={mism}"
        stats = {
            "input_shape": tuple(DRAFT_TOKEN_IDS.shape),
            "output_shape": tuple(k_out.shape),
            "dtype": str(k_out.dtype),
            "device": str(DRAFT_TOKEN_IDS.device),
            "mismatch": mism,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(
            CU_NUM_DRAFT_TOKENS, DRAFT_TOKEN_IDS, TARGET_ARGMAX, BONUS_TOKEN_IDS
        ))
        kern_stats = _bench(lambda: kernel_impl(
            CU_NUM_DRAFT_TOKENS, DRAFT_TOKEN_IDS, TARGET_ARGMAX, BONUS_TOKEN_IDS
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
            "Kernel: rejection_greedy_sample_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"num_draft_per_req: {NUM_DRAFT_PER_REQ}\n")
            lines.append(f"input (draft) shape: {stats['input_shape']}\n")
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
