"""
Standalone QAIC validation for `_prepare_prefill_inputs_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _prepare_prefill_inputs_kernel  (launched via `prepare_prefill_inputs`)

For every request in the batch the kernel gathers `query_len` prompt/output
token ids from the persistent per-request history buffer `all_token_ids`
(row = req_state_idx, starting at `num_computed`) into the flat batch
`input_ids` buffer at `query_start_loc[batch_idx]`. If the request still has
more prompt to process it stages the next chunk's first token into
`next_prefill_tokens[req_state_idx]`. Requests that are past prefill
(num_computed >= prefill_len) are skipped.

Integer-exact validation of both mutated buffers (input_ids and
next_prefill_tokens).
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prepare_prefill_inputs_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"

DEVICE = "qaic"

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))
from vllm.v1.worker.gpu.input_batch import prepare_prefill_inputs

torch.manual_seed(42)

# ---- Global shared inputs -------------------------------------------------
MAX_NUM_REQS = 6
MAX_MODEL_LEN = 32
NUM_REQS = 4

# batch_idx -> req_state_idx (permutation to exercise the gather).
IDX_MAPPING = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=DEVICE)
# per-request query lengths (batch order).
QUERY_LENS = [4, 3, 5, 2]
_qsl = [0]
for q in QUERY_LENS:
    _qsl.append(_qsl[-1] + q)
QUERY_START_LOC = torch.tensor(_qsl, dtype=torch.int32, device=DEVICE)
TOTAL_TOKENS = _qsl[-1]

# per-req_state persistent history and counters (size MAX_NUM_REQS).
ALL_TOKEN_IDS = torch.randint(
    0, 10000, (MAX_NUM_REQS, MAX_MODEL_LEN), dtype=torch.int32, device=DEVICE
)
PREFILL_LEN = torch.tensor([20, 20, 20, 20, 0, 0], dtype=torch.int32, device=DEVICE)
NUM_COMPUTED_TOKENS = torch.tensor(
    [0, 2, 4, 1, 0, 0], dtype=torch.int32, device=DEVICE
)

INPUT_IDS = torch.zeros(TOTAL_TOKENS, dtype=torch.int32, device=DEVICE)
NEXT_PREFILL_TOKENS = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)


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


def pytorch_ref(input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                all_token_ids, prefill_len, num_computed_tokens):
    input_ids = input_ids.cpu().clone()
    next_prefill = next_prefill_tokens.cpu().clone()
    idx_mapping = idx_mapping.cpu()
    qsl = query_start_loc.cpu()
    att = all_token_ids.cpu()
    plen = prefill_len.cpu()
    nct = num_computed_tokens.cpu()
    num_reqs = idx_mapping.shape[0]
    for b in range(num_reqs):
        rs = int(idx_mapping[b].item())
        pl = int(plen[rs].item())
        nc = int(nct[rs].item())
        if nc >= pl:
            continue
        qs = int(qsl[b].item())
        qe = int(qsl[b + 1].item())
        ql = qe - qs
        input_ids[qs:qs + ql] = att[rs, nc:nc + ql]
        next_pos = nc + ql
        if next_pos < pl:
            next_prefill[rs] = att[rs, next_pos]
    return input_ids, next_prefill


def kernel_impl(input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
                all_token_ids, prefill_len, num_computed_tokens):
    input_ids = input_ids.clone()
    next_prefill_tokens = next_prefill_tokens.clone()
    prepare_prefill_inputs(
        input_ids,
        next_prefill_tokens,
        idx_mapping,
        query_start_loc,
        all_token_ids,
        prefill_len,
        num_computed_tokens,
    )
    return input_ids, next_prefill_tokens


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ids, ref_next = pytorch_ref(
            INPUT_IDS, NEXT_PREFILL_TOKENS, IDX_MAPPING, QUERY_START_LOC,
            ALL_TOKEN_IDS, PREFILL_LEN, NUM_COMPUTED_TOKENS,
        )
        k_ids, k_next = kernel_impl(
            INPUT_IDS, NEXT_PREFILL_TOKENS, IDX_MAPPING, QUERY_START_LOC,
            ALL_TOKEN_IDS, PREFILL_LEN, NUM_COMPUTED_TOKENS,
        )
        k_ids = k_ids.cpu()
        k_next = k_next.cpu()

        mism_ids = int((k_ids != ref_ids).sum().item())
        mism_next = int((k_next != ref_next).sum().item())
        assert mism_ids == 0, f"input_ids mismatch count={mism_ids}"
        assert mism_next == 0, f"next_prefill_tokens mismatch count={mism_next}"

        stats = {
            "input_ids_shape": tuple(INPUT_IDS.shape),
            "next_prefill_shape": tuple(NEXT_PREFILL_TOKENS.shape),
            "dtype": str(INPUT_IDS.dtype),
            "device": str(INPUT_IDS.device),
            "mismatch_input_ids": mism_ids,
            "mismatch_next_prefill": mism_next,
            "max_abs_diff": 0,
            "grid": f"({NUM_REQS},)",
        }
        pt_stats = _bench(lambda: pytorch_ref(
            INPUT_IDS, NEXT_PREFILL_TOKENS, IDX_MAPPING, QUERY_START_LOC,
            ALL_TOKEN_IDS, PREFILL_LEN, NUM_COMPUTED_TOKENS,
        ))
        kern_stats = _bench(lambda: kernel_impl(
            INPUT_IDS, NEXT_PREFILL_TOKENS, IDX_MAPPING, QUERY_START_LOC,
            ALL_TOKEN_IDS, PREFILL_LEN, NUM_COMPUTED_TOKENS,
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
            "Kernel: _prepare_prefill_inputs_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append(f"input_ids shape: {stats['input_ids_shape']}\n")
            lines.append(f"next_prefill shape: {stats['next_prefill_shape']}\n")
            lines.append(f"dtype: {stats['dtype']}\n")
            lines.append(f"device: {stats['device']}\n")
            lines.append(f"grid: {stats['grid']}\n")
            lines.append(f"mismatch(input_ids): {stats['mismatch_input_ids']}\n")
            lines.append(f"mismatch(next_prefill): {stats['mismatch_next_prefill']}\n")
            lines.append(f"max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append("relative_error: 0.0 (exact integer match)\n")
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
