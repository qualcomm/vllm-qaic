"""
Standalone QAIC validation for `_prompt_logprobs_token_ids_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/prompt_logprob.py
  - _prompt_logprobs_token_ids_kernel (gather target/next token ids from the
    request token history for prompt logprobs), via the
    `get_prompt_logprobs_token_ids` launcher.

Per batch request:
  query_start = query_start_loc[b]; query_end = query_start_loc[b+1]
  for each local position `j` in [0, query_len):
    target_pos = num_computed_tokens[req] + 1 + j   # shifted by one (next token)
    out[query_start + j] = all_token_ids[req, target_pos]

Integer/index kernel -> exact equality.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.prompt_logprob import get_prompt_logprobs_token_ids

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_prompt_logprobs_token_ids_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/prompt_logprob.py"
KERNEL_NAME = "_prompt_logprobs_token_ids_kernel"

DEVICE = "qaic"
NUM_REQS = 4
MAX_NUM_REQS = 4
MAX_LEN = 64

torch.manual_seed(42)
# Query lengths per request -> cumulative query_start_loc.
QUERY_LENS = [5, 3, 8, 4]
NUM_TOKENS = sum(QUERY_LENS)
_cu = [0]
for q in QUERY_LENS:
    _cu.append(_cu[-1] + q)
QUERY_START_LOC = torch.tensor(_cu, dtype=torch.int32, device=DEVICE)
IDX_MAPPING = torch.arange(NUM_REQS, dtype=torch.int32, device=DEVICE)
NUM_COMPUTED_TOKENS = torch.tensor([0, 2, 5, 1], dtype=torch.int32, device=DEVICE)
ALL_TOKEN_IDS = torch.randint(
    0, 1000, (MAX_NUM_REQS, MAX_LEN), dtype=torch.int32, device=DEVICE
)


def _log(status, stats, error_text, ts):
    os.makedirs(LOG_DIR, exist_ok=True)
    lines = [
        f"{ts}\n",
        f"Kernel: {KERNEL_NAME}\n",
        f"Kernel file: {KERNEL_FILE_PATH}\n",
        f"Device target: QAIC (device='{DEVICE}')\n",
        f"Status: {status}\n\n",
    ]
    if status == "SUCCESS":
        for k, v in stats.items():
            lines.append(f"- {k}: {v}\n")
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
        lines.append("Error:\n" + error_text + "\n")
    lines.append("\n------------------------------------\n\n")
    with open(LOG_FILE, "a") as f:
        f.write("".join(lines))


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


def pytorch_ref():
    qsl = QUERY_START_LOC.cpu()
    mapping = IDX_MAPPING.cpu()
    nct = NUM_COMPUTED_TOKENS.cpu()
    tokens = ALL_TOKEN_IDS.cpu()
    out = torch.zeros(NUM_TOKENS, dtype=torch.int64)
    for b in range(NUM_REQS):
        req = int(mapping[b].item())
        qs = int(qsl[b].item())
        qe = int(qsl[b + 1].item())
        base = int(nct[req].item()) + 1
        for j in range(qe - qs):
            out[qs + j] = int(tokens[req, base + j].item())
    return out


def kernel_impl():
    return get_prompt_logprobs_token_ids(
        NUM_TOKENS,
        QUERY_START_LOC,
        IDX_MAPPING,
        NUM_COMPUTED_TOKENS,
        ALL_TOKEN_IDS,
    )


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        out = kernel_impl()
        out_cpu = out.cpu().to(torch.int64)
        mismatch = int((out_cpu != ref).sum().item())
        assert mismatch == 0, f"token id mismatch count={mismatch}"
        stats = {
            "output_shape": tuple(out.shape),
            "output_dtype": str(out.dtype),
            "device": str(out.device),
            "query_lens": QUERY_LENS,
            "num_computed_tokens": NUM_COMPUTED_TOKENS.cpu().tolist(),
            "mismatch_count": mismatch,
            "max_abs_diff": 0,
            "comparison": "exact integer equality",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
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
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
