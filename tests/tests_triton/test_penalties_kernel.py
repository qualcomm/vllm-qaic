"""
Standalone QAIC validation for `_penalties_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/penalties.py
  - _penalties_kernel (repetition / frequency / presence penalties), via the
    `apply_penalties` launcher.

Penalty math (per token row, per request state):
  output_bin_mask = output_bin_counts > 0
  # repetition: scale = rep_penalty if token in prompt-bitmask OR output-counts
  #             else 1.0 ; logits *= (1/scale if logits>0 else scale)
  # frequency:  logits -= freq_penalty * output_bin_counts
  # presence:   logits -= pres_penalty * output_bin_mask

Determinism note: we set `expanded_local_pos = 0` for every token, so the
kernel's in-flight draft-token accumulation loop (`for prev_pos in tl.range(pos)`)
does not execute. output_bin_counts is then exactly the provided base tensor,
which the pure-PyTorch reference reproduces. Float compare.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.penalties import apply_penalties

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_penalties_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/penalties.py"
KERNEL_NAME = "_penalties_kernel"

DEVICE = "qaic"
NUM_REQS = 4
NUM_TOKENS = NUM_REQS
VOCAB_SIZE = 256
PACKED = (VOCAB_SIZE + 31) // 32

torch.manual_seed(42)
LOGITS = torch.randn(NUM_TOKENS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
# pos == 0 => no in-flight draft-token accumulation loop.
EXPANDED_LOCAL_POS = torch.zeros(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
TOKEN_IDS = torch.zeros(NUM_TOKENS, dtype=torch.int64, device=DEVICE)

REP_PENALTY = torch.tensor([1.0, 1.2, 0.8, 1.5], dtype=torch.float32, device=DEVICE)
FREQ_PENALTY = torch.tensor([0.0, 0.5, 0.0, 0.3], dtype=torch.float32, device=DEVICE)
PRES_PENALTY = torch.tensor([0.0, 0.0, 0.7, 0.4], dtype=torch.float32, device=DEVICE)

# Random output bin counts (int32) and a random packed prompt bitmask.
OUTPUT_BIN_COUNTS = torch.randint(
    0, 3, (NUM_REQS, VOCAB_SIZE), dtype=torch.int32, device=DEVICE
)
PROMPT_BIN_MASK = torch.randint(
    -(2**31), 2**31, (NUM_REQS, PACKED), dtype=torch.int32, device=DEVICE
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


def _unpack_prompt_mask(packed_row):
    """Unpack a packed int32 bitmask row into a [VOCAB_SIZE] bool tensor,
    matching the kernel's `(packed >> arange(32)) & 1` unpacking."""
    packed = packed_row.to(torch.int64) & 0xFFFFFFFF  # treat as uint32 bits
    bits = torch.zeros(PACKED * 32, dtype=torch.bool)
    for w in range(PACKED):
        val = int(packed[w].item())
        for b in range(32):
            bits[w * 32 + b] = bool((val >> b) & 1)
    return bits[:VOCAB_SIZE]


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
    rep = REP_PENALTY.cpu()
    freq = FREQ_PENALTY.cpu()
    pres = PRES_PENALTY.cpu()
    counts = OUTPUT_BIN_COUNTS.cpu()
    packed = PROMPT_BIN_MASK.cpu()

    for t in range(NUM_TOKENS):
        req = int(mapping[t].item())
        rp = float(rep[req].item())
        fp = float(freq[req].item())
        pp = float(pres[req].item())
        use_rep = rp != 1.0
        use_freq = fp != 0.0
        use_pres = pp != 0.0
        if not (use_rep or use_freq or use_pres):
            continue

        row = logits[t]
        out_counts = counts[req].to(torch.float32)
        out_mask = out_counts > 0

        if use_rep:
            prompt_mask = _unpack_prompt_mask(packed[req])
            appears = prompt_mask | out_mask
            scale = torch.where(
                appears, torch.tensor(rp), torch.tensor(1.0)
            )
            row = row * torch.where(row > 0, 1.0 / scale, scale)

        row = row - fp * out_counts
        row = row - pp * out_mask.to(torch.float32)
        logits[t] = row
    return logits


def kernel_impl():
    logits = LOGITS.clone()
    apply_penalties(
        logits,
        EXPANDED_IDX_MAPPING,
        TOKEN_IDS,
        EXPANDED_LOCAL_POS,
        REP_PENALTY,
        FREQ_PENALTY,
        PRES_PENALTY,
        PROMPT_BIN_MASK,
        OUTPUT_BIN_COUNTS,
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
        ref_cpu = ref.cpu().to(torch.float32)
        out_cpu = out.cpu().to(torch.float32)
        torch.testing.assert_close(out_cpu, ref_cpu, rtol=1e-3, atol=1e-3)
        diff = (out_cpu - ref_cpu).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "rep_penalty": REP_PENALTY.cpu().tolist(),
            "freq_penalty": FREQ_PENALTY.cpu().tolist(),
            "pres_penalty": PRES_PENALTY.cpu().tolist(),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "note": "expanded_local_pos=0 => draft-token loop disabled",
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
