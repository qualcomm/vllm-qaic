"""
Standalone QAIC validation for `_post_update_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _post_update_kernel  (launched via `post_update`)

Per-request post-sampling state update. For each batch row (req_id) mapping to
a persistent req_state_idx (negative => skip):
  - record the last sampled token id into last_sampled_tokens[req_state_idx]
    and bump total_len[req_state_idx] by num_sampled (only if num_sampled>0),
  - append each sampled token id to the request's token history
    all_token_ids[req_state_idx, total_len + i],
  - increment output_bin_counts[req_state_idx, token_id] for each sampled token,
  - update num_computed_tokens[req_state_idx] by (query_len - num_rejected).

CONFIG NOTE: We use a simple non-speculative config -- num_speculative_steps=0
so sampled_tokens has shape [num_reqs, 1] and num_sampled==1 per request. This
exercises the full body (the `for i in range(num_sampled)` loop, bin counts,
and the num_computed delta) without spec-decode branching. output_bin_counts is
provided (non-None) so the bin-count path is tested. All idx_mapping entries are
non-negative.

Integer-exact validation of all 5 mutated buffers: last_sampled_tokens,
total_len, all_token_ids, output_bin_counts, num_computed_tokens.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_post_update_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.gpu.input_batch import post_update

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
MAX_NUM_REQS = 6
MAX_MODEL_LEN = 32
VOCAB = 50
NUM_REQS = 4
NUM_SAMPLED_PER_REQ = 1  # non-speculative: 1 sampled token per request.

IDX_MAPPING = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)

# Per-req_state persistent buffers (sized MAX_NUM_REQS).
NUM_COMPUTED_TOKENS = torch.tensor(
    [3, 7, 5, 11, 0, 0], dtype=torch.int32, device=DEVICE
)
LAST_SAMPLED_TOKENS = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
TOTAL_LEN = torch.tensor([3, 7, 5, 11, 0, 0], dtype=torch.int32, device=DEVICE)
ALL_TOKEN_IDS = torch.zeros(
    MAX_NUM_REQS, MAX_MODEL_LEN, dtype=torch.int32, device=DEVICE
)
OUTPUT_BIN_COUNTS = torch.zeros(MAX_NUM_REQS, VOCAB, dtype=torch.int32, device=DEVICE)

# Per-batch buffers (sized NUM_REQS / NUM_REQS+1).
SAMPLED_TOKENS = torch.randint(
    0, VOCAB, (NUM_REQS, NUM_SAMPLED_PER_REQ), dtype=torch.int32, device=DEVICE
)
NUM_SAMPLED = torch.ones(NUM_REQS, dtype=torch.int32, device=DEVICE)
NUM_REJECTED = torch.zeros(NUM_REQS, dtype=torch.int32, device=DEVICE)
QUERY_LENS = [1, 1, 1, 1]
_qsl = [0]
for q in QUERY_LENS:
    _qsl.append(_qsl[-1] + q)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(idx_mapping, num_computed_tokens, last_sampled_tokens,
                output_bin_counts, sampled_tokens, num_sampled, num_rejected,
                query_start_loc, all_token_ids, total_len):
    idx_mapping = idx_mapping.cpu()
    num_computed = num_computed_tokens.cpu().clone()
    last_sampled = last_sampled_tokens.cpu().clone()
    obc = output_bin_counts.cpu().clone()
    sampled = sampled_tokens.cpu()
    n_sampled = num_sampled.cpu()
    n_rejected = num_rejected.cpu()
    qsl = query_start_loc.cpu()
    ati = all_token_ids.cpu().clone()
    tlen = total_len.cpu().clone()
    num_reqs = idx_mapping.shape[0]
    for b in range(num_reqs):
        rs = int(idx_mapping[b].item())
        if rs < 0:
            continue
        orig_total = int(tlen[rs].item())
        ns = int(n_sampled[b].item())
        if ns > 0:
            token_id = int(sampled[b, ns - 1].item())
            last_sampled[rs] = token_id
            tlen[rs] = orig_total + ns
        for i in range(ns):
            token_id = int(sampled[b, i].item())
            ati[rs, orig_total + i] = token_id
            obc[rs, token_id] = obc[rs, token_id] + 1
        qs = int(qsl[b].item())
        qe = int(qsl[b + 1].item())
        query_len = qe - qs
        nr = int(n_rejected[b].item())
        computed_delta = query_len - nr
        if computed_delta != 0:
            num_computed[rs] = int(num_computed[rs].item()) + computed_delta
    return last_sampled, tlen, ati, obc, num_computed


def kernel_impl(idx_mapping, num_computed_tokens, last_sampled_tokens,
                output_bin_counts, sampled_tokens, num_sampled, num_rejected,
                query_start_loc, all_token_ids, total_len):
    num_computed_tokens = num_computed_tokens.clone()
    last_sampled_tokens = last_sampled_tokens.clone()
    output_bin_counts = output_bin_counts.clone()
    all_token_ids = all_token_ids.clone()
    total_len = total_len.clone()
    post_update(
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )
    return last_sampled_tokens, total_len, all_token_ids, output_bin_counts, num_computed_tokens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        refs = pytorch_ref(
            IDX_MAPPING, NUM_COMPUTED_TOKENS, LAST_SAMPLED_TOKENS,
            OUTPUT_BIN_COUNTS, SAMPLED_TOKENS, NUM_SAMPLED, NUM_REJECTED,
            QUERY_START_LOC, ALL_TOKEN_IDS, TOTAL_LEN,
        )
        kers = kernel_impl(
            IDX_MAPPING, NUM_COMPUTED_TOKENS, LAST_SAMPLED_TOKENS,
            OUTPUT_BIN_COUNTS, SAMPLED_TOKENS, NUM_SAMPLED, NUM_REJECTED,
            QUERY_START_LOC, ALL_TOKEN_IDS, TOTAL_LEN,
        )
        names = ["last_sampled_tokens", "total_len", "all_token_ids",
                 "output_bin_counts", "num_computed_tokens"]
        mismatches = {}
        for name, r, k in zip(names, refs, kers):
            m = int((k.cpu() != r).sum().item())
            mismatches[name] = m
            assert m == 0, f"{name} mismatch count={m}"

        stats = {
            "device": str(IDX_MAPPING.device),
            "num_reqs": NUM_REQS,
            "vocab": VOCAB,
            "mismatches": mismatches,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(
            IDX_MAPPING, NUM_COMPUTED_TOKENS, LAST_SAMPLED_TOKENS,
            OUTPUT_BIN_COUNTS, SAMPLED_TOKENS, NUM_SAMPLED, NUM_REJECTED,
            QUERY_START_LOC, ALL_TOKEN_IDS, TOTAL_LEN,
        ))
        kern_stats = _bench(lambda: kernel_impl(
            IDX_MAPPING, NUM_COMPUTED_TOKENS, LAST_SAMPLED_TOKENS,
            OUTPUT_BIN_COUNTS, SAMPLED_TOKENS, NUM_SAMPLED, NUM_REJECTED,
            QUERY_START_LOC, ALL_TOKEN_IDS, TOTAL_LEN,
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
            "Kernel: _post_update_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Config: non-speculative (num_sampled=1/req, output_bin_counts enabled)\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"device: {stats['device']}\n")
            lines.append(f"num_reqs: {stats['num_reqs']}  vocab: {stats['vocab']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            for name, m in stats["mismatches"].items():
                lines.append(f"mismatch({name}): {m}\n")
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
