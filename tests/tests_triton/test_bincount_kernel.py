"""
Standalone QAIC validation for `_bincount_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/penalties.py
  - _bincount_kernel (build per-request prompt-token bitmask + output-token
    bincounts), via the `bincount` launcher.

For each request state:
  - prompt tokens  = all_token_ids[req, 0:prompt_len]
    -> set bit (token % 32) in word (token // 32) of prompt_bin_mask via atomic_or
  - output tokens  = all_token_ids[req, prompt_len:prefill_len]
    -> atomic_add 1 to output_bin_counts[req, token]

Integer kernel -> exact equality on the prompt bitmask and the output counts.
The `bincount` launcher first zeroes both tensors for the mapped requests.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.penalties import bincount

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_bincount_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/penalties.py"
KERNEL_NAME = "_bincount_kernel"

DEVICE = "qaic"
MAX_NUM_REQS = 4
NUM_TOKENS = 4  # one expanded token per request
VOCAB_SIZE = 256
PACKED = (VOCAB_SIZE + 31) // 32
MAX_LEN = 64

torch.manual_seed(42)
# Each expanded token maps to a distinct request state.
EXPANDED_IDX_MAPPING = torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
ALL_TOKEN_IDS = torch.randint(
    0, VOCAB_SIZE, (MAX_NUM_REQS, MAX_LEN), dtype=torch.int32, device=DEVICE
)
PROMPT_LEN = torch.tensor([8, 12, 5, 10], dtype=torch.int32, device=DEVICE)
PREFILL_LEN = torch.tensor([16, 20, 9, 24], dtype=torch.int32, device=DEVICE)
MAX_PREFILL_LEN = int(PREFILL_LEN.max().item())

PROMPT_BIN_MASK = torch.zeros(
    MAX_NUM_REQS, PACKED, dtype=torch.int32, device=DEVICE
)
OUTPUT_BIN_COUNTS = torch.zeros(
    MAX_NUM_REQS, VOCAB_SIZE, dtype=torch.int32, device=DEVICE
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
    tokens = ALL_TOKEN_IDS.cpu()
    mapping = EXPANDED_IDX_MAPPING.cpu()
    prompt_len = PROMPT_LEN.cpu()
    prefill_len = PREFILL_LEN.cpu()

    mask = torch.zeros(MAX_NUM_REQS, PACKED, dtype=torch.int32)
    counts = torch.zeros(MAX_NUM_REQS, VOCAB_SIZE, dtype=torch.int32)

    for t in range(NUM_TOKENS):
        req = int(mapping[t].item())
        pl = int(prompt_len[req].item())
        fl = int(prefill_len[req].item())
        # Prompt tokens -> bitmask.
        for i in range(pl):
            tok = int(tokens[req, i].item())
            word = tok // 32
            bit = tok % 32
            mask[req, word] = mask[req, word] | (1 << bit)
        # Output tokens -> bincounts.
        for i in range(pl, fl):
            tok = int(tokens[req, i].item())
            counts[req, tok] += 1
    return mask, counts


def kernel_impl():
    # bincount zeroes the mapped rows internally before accumulating.
    bincount(
        EXPANDED_IDX_MAPPING,
        ALL_TOKEN_IDS,
        PROMPT_LEN,
        PREFILL_LEN,
        PROMPT_BIN_MASK,
        OUTPUT_BIN_COUNTS,
        MAX_PREFILL_LEN,
    )
    return PROMPT_BIN_MASK, OUTPUT_BIN_COUNTS


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_mask, ref_counts = pytorch_ref()
        out_mask, out_counts = kernel_impl()
        # Compare unsigned bit patterns for the mask.
        m_kernel = out_mask.cpu().to(torch.int64) & 0xFFFFFFFF
        m_ref = ref_mask.to(torch.int64) & 0xFFFFFFFF
        c_kernel = out_counts.cpu().to(torch.int64)
        c_ref = ref_counts.to(torch.int64)

        mask_mismatch = int((m_kernel != m_ref).sum().item())
        count_mismatch = int((c_kernel != c_ref).sum().item())
        assert mask_mismatch == 0, f"prompt bitmask mismatch count={mask_mismatch}"
        assert count_mismatch == 0, f"output bincount mismatch count={count_mismatch}"

        stats = {
            "all_token_ids_shape": tuple(ALL_TOKEN_IDS.shape),
            "prompt_bin_mask_shape": tuple(out_mask.shape),
            "output_bin_counts_shape": tuple(out_counts.shape),
            "dtype": str(out_mask.dtype),
            "device": str(out_mask.device),
            "prompt_len": PROMPT_LEN.cpu().tolist(),
            "prefill_len": PREFILL_LEN.cpu().tolist(),
            "mask_mismatch_count": mask_mismatch,
            "count_mismatch_count": count_mismatch,
            "max_abs_diff": 0,
            "comparison": "exact integer equality (bitmask + counts)",
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
