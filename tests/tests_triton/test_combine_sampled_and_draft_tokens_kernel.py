"""
Standalone QAIC validation for `_combine_sampled_and_draft_tokens_kernel`.

Source under test:
vllm/v1/worker/gpu/input_batch.py
  - _combine_sampled_and_draft_tokens_kernel  (via launcher
    `combine_sampled_and_draft_tokens`)

For each request (batch_idx) the kernel:
  * writes the contiguous logits indices [query_end-num_logits, query_end)
    into `logits_indices` (returned value), and
  * for decode rows (seq_len > prefill_len) writes the last sampled token id
    at position (query_end - num_logits) of `input_ids`, followed by the
    request's draft tokens filling (query_end - num_draft, query_end).
  Prefill rows (seq_len <= prefill_len) only write logits indices.

Config tested: 3 decode requests, 2 speculative draft tokens each
(num_logits = 3 per request, query_len = 3). We validate BOTH the mutated
`input_ids` and the returned `logits_indices` with EXACT integer equality.
Reference: pure-PyTorch/python replication of the kernel semantics.
"""

import datetime
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_combine_sampled_and_draft_tokens_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/input_batch.py"
DEVICE = "qaic"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.worker.gpu.input_batch import combine_sampled_and_draft_tokens  # noqa: E402

torch.manual_seed(42)

# ---- Global shared inputs (used by BOTH implementations) ----
NUM_REQS = 3
NUM_SPEC = 2  # draft tokens per request
NUM_LOGITS_PER = NUM_SPEC + 1  # 3
NUM_LOGITS = NUM_REQS * NUM_LOGITS_PER  # 9
TOTAL_TOKENS = NUM_REQS * NUM_LOGITS_PER  # query_len == num_logits == 3 each

IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
# query_start_loc: 3 tokens per request -> [0, 3, 6, 9]
QUERY_START_LOC = torch.tensor([0, 3, 6, 9], dtype=torch.int32, device=DEVICE)
# cu_num_logits: 3 logits per request -> [0, 3, 6, 9]
CU_NUM_LOGITS = torch.tensor([0, 3, 6, 9], dtype=torch.int32, device=DEVICE)
# All requests decode: seq_len > prefill_len.
SEQ_LENS = torch.tensor([10, 12, 8], dtype=torch.int32, device=DEVICE)
PREFILL_LEN = torch.tensor([5, 5, 5], dtype=torch.int32, device=DEVICE)
LAST_SAMPLED = torch.tensor([500, 501, 502], dtype=torch.int32, device=DEVICE)
DRAFT_TOKENS = torch.tensor(
    [[100, 101], [110, 111], [120, 121]], dtype=torch.int32, device=DEVICE
)
INPUT_IDS_BASE = torch.full((TOTAL_TOKENS,), -1, dtype=torch.int32, device=DEVICE)


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
    """Pure PyTorch replication. Returns (input_ids, logits_indices)."""
    input_ids = INPUT_IDS_BASE.clone().cpu()
    logits_indices = torch.empty(NUM_LOGITS, dtype=torch.int64)
    idx_mapping = IDX_MAPPING.cpu()
    qsl = QUERY_START_LOC.cpu()
    cnl = CU_NUM_LOGITS.cpu()
    seq_lens = SEQ_LENS.cpu()
    prefill_len = PREFILL_LEN.cpu()
    last_sampled = LAST_SAMPLED.cpu()
    draft = DRAFT_TOKENS.cpu()
    for b in range(NUM_REQS):
        req_state_idx = int(idx_mapping[b])
        start = int(cnl[b])
        end = int(cnl[b + 1])
        num_logits = end - start
        num_draft = num_logits - 1
        query_end = int(qsl[b + 1])
        logits_start = query_end - num_logits
        for k in range(num_logits):
            logits_indices[start + k] = logits_start + k
        if int(seq_lens[b]) <= int(prefill_len[req_state_idx]):
            continue
        input_ids[query_end - num_logits] = last_sampled[req_state_idx]
        if num_draft > 0:
            for k in range(num_draft):
                input_ids[query_end - num_draft + k] = draft[req_state_idx, k]
    return input_ids, logits_indices


def kernel_impl():
    """Kernel launch only. Returns (input_ids, logits_indices)."""
    input_ids = INPUT_IDS_BASE.clone()
    logits_indices = combine_sampled_and_draft_tokens(
        input_ids,
        IDX_MAPPING,
        LAST_SAMPLED,
        QUERY_START_LOC,
        SEQ_LENS,
        PREFILL_LEN,
        DRAFT_TOKENS,
        CU_NUM_LOGITS,
        NUM_LOGITS,
    )
    return input_ids, logits_indices


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ids, ref_li = pytorch_ref()
        kern_ids, kern_li = kernel_impl()

        ref_ids_cpu = ref_ids.cpu()
        kern_ids_cpu = kern_ids.cpu()
        ref_li_cpu = ref_li.cpu()
        kern_li_cpu = kern_li.cpu()

        assert torch.equal(kern_ids_cpu, ref_ids_cpu), (
            f"input_ids mismatch ref={ref_ids_cpu.tolist()} "
            f"k={kern_ids_cpu.tolist()}"
        )
        assert torch.equal(kern_li_cpu, ref_li_cpu), (
            f"logits_indices mismatch ref={ref_li_cpu.tolist()} "
            f"k={kern_li_cpu.tolist()}"
        )

        stats = {
            "input_shape": tuple(INPUT_IDS_BASE.shape),
            "output_shape": tuple(kern_li.shape),
            "in_dtype": str(INPUT_IDS_BASE.dtype),
            "out_dtype": str(kern_li.dtype),
            "device": str(INPUT_IDS_BASE.device),
            "max_abs_diff": 0,
            "mean_abs_diff": 0.0,
            "input_ids": kern_ids_cpu.tolist(),
            "logits_indices": kern_li_cpu.tolist(),
        }

        pt_stats = _bench(pytorch_ref)
        kern_stats = _bench(kernel_impl)
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup

        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _combine_sampled_and_draft_tokens_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input shape: {stats['input_shape']}\n",
                f"- in dtype: {stats['in_dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output shape: {stats['output_shape']}\n",
                f"- out dtype: {stats['out_dtype']}\n",
                f"- input_ids: {stats['input_ids']}\n",
                f"- logits_indices: {stats['logits_indices']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']} (exact-match comparison)\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines += [
                    "Timing:\n",
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n",
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n",
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n",
                ]
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
