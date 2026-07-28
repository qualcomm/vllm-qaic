"""
Standalone QAIC validation for `_bias_kernel`.

Source under test:
vllm/v1/worker/gpu/sample/logit_bias.py
  - _bias_kernel (allowed-token restriction + additive logit bias +
    min-tokens stop-token masking), via the `apply_logit_bias` launcher.

Per token row (indexed to a request state):
  1. Allowed token ids: if num_allowed > 0, set ALL logits to -inf, then
     restore the original logits only at the allowed token ids.
  2. Logit bias: logits[bias_token_ids] += bias.
  3. Min tokens: if num_stop_token_ids > 0 and pos < min_len, set
     logits[stop_token_ids] = -inf.

Float compare on finite entries + exact -inf-mask equality.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.worker.gpu.sample.logit_bias import apply_logit_bias

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_bias_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/worker/gpu/sample/logit_bias.py"
KERNEL_NAME = "_bias_kernel"

DEVICE = "qaic"
NUM_REQS = 4
NUM_TOKENS = NUM_REQS
VOCAB_SIZE = 256
MAX_ALLOWED = 8
MAX_BIAS = 8
MAX_STOP = 8

torch.manual_seed(42)
LOGITS = torch.randn(NUM_TOKENS, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)
EXPANDED_IDX_MAPPING = torch.arange(NUM_TOKENS, dtype=torch.int32, device=DEVICE)
POS = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=DEVICE)

# Request 0: allowed-token restriction only.
# Request 1: additive logit bias only.
# Request 2: min-tokens stop masking (pos < min_len active).
# Request 3: no logit-bias features.
NUM_ALLOWED = torch.tensor([4, 0, 0, 0], dtype=torch.int32, device=DEVICE)
ALLOWED_TOKEN_IDS = torch.zeros(
    NUM_REQS, MAX_ALLOWED, dtype=torch.int32, device=DEVICE
)
ALLOWED_TOKEN_IDS[0, :4] = torch.tensor([3, 17, 42, 99], dtype=torch.int32, device=DEVICE)

NUM_BIAS = torch.tensor([0, 3, 0, 0], dtype=torch.int32, device=DEVICE)
BIAS_TOKEN_IDS = torch.zeros(NUM_REQS, MAX_BIAS, dtype=torch.int32, device=DEVICE)
BIAS_TOKEN_IDS[1, :3] = torch.tensor([5, 20, 200], dtype=torch.int32, device=DEVICE)
BIAS = torch.zeros(NUM_REQS, MAX_BIAS, dtype=torch.float32, device=DEVICE)
BIAS[1, :3] = torch.tensor([2.5, -1.0, 3.3], dtype=torch.float32, device=DEVICE)

MIN_LENS = torch.tensor([0, 0, 10, 0], dtype=torch.int32, device=DEVICE)
NUM_STOP = torch.tensor([0, 0, 3, 0], dtype=torch.int32, device=DEVICE)
STOP_TOKEN_IDS = torch.zeros(NUM_REQS, MAX_STOP, dtype=torch.int32, device=DEVICE)
STOP_TOKEN_IDS[2, :3] = torch.tensor([7, 88, 150], dtype=torch.int32, device=DEVICE)


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
    pos = POS.cpu()
    num_allowed = NUM_ALLOWED.cpu()
    allowed = ALLOWED_TOKEN_IDS.cpu()
    num_bias = NUM_BIAS.cpu()
    bias_ids = BIAS_TOKEN_IDS.cpu()
    bias = BIAS.cpu()
    min_lens = MIN_LENS.cpu()
    num_stop = NUM_STOP.cpu()
    stop_ids = STOP_TOKEN_IDS.cpu()

    for t in range(NUM_TOKENS):
        req = int(mapping[t].item())
        row = logits[t]

        na = int(num_allowed[req].item())
        if na > 0:
            ids = allowed[req, :na].to(torch.int64)
            saved = row[ids].clone()
            row[:] = float("-inf")
            row[ids] = saved

        nb = int(num_bias[req].item())
        if nb > 0:
            ids = bias_ids[req, :nb].to(torch.int64)
            row[ids] = row[ids] + bias[req, :nb]

        ns = int(num_stop[req].item())
        if ns > 0 and int(pos[t].item()) < int(min_lens[req].item()):
            ids = stop_ids[req, :ns].to(torch.int64)
            row[ids] = float("-inf")
        logits[t] = row
    return logits


def kernel_impl():
    logits = LOGITS.clone()
    apply_logit_bias(
        logits,
        EXPANDED_IDX_MAPPING,
        POS,
        NUM_ALLOWED,
        ALLOWED_TOKEN_IDS,
        NUM_BIAS,
        BIAS_TOKEN_IDS,
        BIAS,
        MIN_LENS,
        NUM_STOP,
        STOP_TOKEN_IDS,
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

        diff = (out_cpu[finite_mask] - ref_cpu[finite_mask]).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "inf_mask_match": bool(same_inf.all()),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "features": "allowed / bias / min-tokens per request",
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
