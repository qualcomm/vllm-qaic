"""
Standalone QAIC validation for `_bad_words_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/bad_words.py
  - _bad_words_kernel (mask logits to -inf for tokens that would complete a
    user-specified bad-word token sequence), via the `apply_bad_words` launcher.

Grid is (num_tokens, max_num_bad_words). For each (token, bad_word):
  effective_len = (total_len - prompt_len) + pos
  prefix_len    = len(bad_word) - 1
  The kernel checks whether the last `prefix_len` generated tokens (drawn from
  the request's output history, and from the in-flight spec `input_ids` when
  actual_pos >= output_len) equal the bad-word prefix; if so it masks the
  bad word's final token to -inf.

We use pos == 0 for every token so the in-flight spec-input branch is never
taken (all history reads come from `all_token_ids`), keeping the reference
deterministic. Float compare on finite entries + exact -inf-mask equality.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.bad_words import apply_bad_words

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_bad_words_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/bad_words.py"
KERNEL_NAME = "_bad_words_kernel"

DEVICE = "qaic"
NUM_REQS = 2
NUM_TOKENS = NUM_REQS
VOCAB_SIZE = 128
MAX_LEN = 16
MAX_NUM_BAD_WORDS = 4
MAX_TOTAL_TOKENS = 32
MAX_NUM_BAD_WORDS_ACTIVE = 1  # grid dim 1

torch.manual_seed(42)
LOGITS = torch.randn(NUM_TOKENS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
EXPANDED_LOCAL_POS = torch.zeros(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
INPUT_IDS = torch.zeros(NUM_TOKENS, dtype=torch.int32, device=DEVICE)

PROMPT_LEN = torch.tensor([4, 4], dtype=torch.int32, device=DEVICE)
TOTAL_LEN = torch.tensor([8, 8], dtype=torch.int32, device=DEVICE)
NUM_BAD_WORDS = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)

# Bad words: req0 = [10, 20, 30] (prefix [10,20], last token 30);
#            req1 = [11, 21, 31].
BAD_WORD_TOKEN_IDS = torch.zeros(
    NUM_REQS, MAX_TOTAL_TOKENS, dtype=torch.int32, device=DEVICE
)
BAD_WORD_TOKEN_IDS[0, :3] = torch.tensor([10, 20, 30], dtype=torch.int32, device=DEVICE)
BAD_WORD_TOKEN_IDS[1, :3] = torch.tensor([11, 21, 31], dtype=torch.int32, device=DEVICE)
BAD_WORD_OFFSETS = torch.zeros(
    NUM_REQS, MAX_NUM_BAD_WORDS + 1, dtype=torch.int32, device=DEVICE
)
BAD_WORD_OFFSETS[0, :2] = torch.tensor([0, 3], dtype=torch.int32, device=DEVICE)
BAD_WORD_OFFSETS[1, :2] = torch.tensor([0, 3], dtype=torch.int32, device=DEVICE)

# all_token_ids: output tokens at [prompt_len:total_len].
# req0 last two output tokens = [10, 20] -> completes bad word -> mask token 30.
# req1 last two output tokens = [3, 4]   -> no match -> no masking.
ALL_TOKEN_IDS = torch.zeros(NUM_REQS, MAX_LEN, dtype=torch.int32, device=DEVICE)
ALL_TOKEN_IDS[0, 4:8] = torch.tensor([50, 60, 10, 20], dtype=torch.int32, device=DEVICE)
ALL_TOKEN_IDS[1, 4:8] = torch.tensor([1, 2, 3, 4], dtype=torch.int32, device=DEVICE)


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
    logits = LOGITS.cpu().clone().to(torch.float32)
    mapping = EXPANDED_IDX_MAPPING.cpu()
    pos_t = EXPANDED_LOCAL_POS.cpu()
    input_ids = INPUT_IDS.cpu()
    prompt_len = PROMPT_LEN.cpu()
    total_len = TOTAL_LEN.cpu()
    num_bw = NUM_BAD_WORDS.cpu()
    tokens = BAD_WORD_TOKEN_IDS.cpu()
    offsets = BAD_WORD_OFFSETS.cpu()
    all_tokens = ALL_TOKEN_IDS.cpu()

    for t in range(NUM_TOKENS):
        req = int(mapping[t].item())
        nbw = int(num_bw[req].item())
        pos = int(pos_t[t].item())
        cur_req_first_pos = t - pos
        pl = int(prompt_len[req].item())
        tl_len = int(total_len[req].item())
        output_len = tl_len - pl
        effective_len = output_len + pos

        for bw_idx in range(MAX_NUM_BAD_WORDS_ACTIVE):
            if bw_idx >= nbw:
                continue
            start = int(offsets[req, bw_idx].item())
            end = int(offsets[req, bw_idx + 1].item())
            prefix_len = (end - start) - 1
            if prefix_len > effective_len:
                continue
            last_token = int(tokens[req, end - 1].item())
            match = True
            for i in range(prefix_len):
                expected = int(tokens[req, start + i].item())
                actual_pos = effective_len - prefix_len + i
                if actual_pos >= output_len:
                    spec_offset = actual_pos - output_len
                    actual = int(input_ids[cur_req_first_pos + spec_offset].item())
                else:
                    actual = int(all_tokens[req, pl + actual_pos].item())
                match = match and (expected == actual)
            if match:
                logits[t, last_token] = float("-inf")
    return logits


def kernel_impl():
    logits = LOGITS.clone()
    apply_bad_words(
        logits,
        EXPANDED_IDX_MAPPING,
        BAD_WORD_TOKEN_IDS,
        BAD_WORD_OFFSETS,
        NUM_BAD_WORDS,
        ALL_TOKEN_IDS,
        PROMPT_LEN,
        TOTAL_LEN,
        INPUT_IDS,
        EXPANDED_LOCAL_POS,
        MAX_NUM_BAD_WORDS_ACTIVE,
    )
    return logits


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref()
        out = kernel_impl()
        ref_cpu = ref.cpu()
        out_cpu = out.cpu()

        same_inf = torch.isinf(ref_cpu) == torch.isinf(out_cpu)
        finite_mask = ~torch.isinf(ref_cpu)
        torch.testing.assert_close(
            out_cpu[finite_mask], ref_cpu[finite_mask], rtol=1e-3, atol=1e-3
        )
        assert bool(same_inf.all()), "-inf mask mismatch"
        num_masked = int(torch.isinf(out_cpu).sum().item())

        diff = (out_cpu[finite_mask] - ref_cpu[finite_mask]).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "num_masked_entries": num_masked,
            "inf_mask_match": bool(same_inf.all()),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "note": "pos=0 => spec-input branch disabled",
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
