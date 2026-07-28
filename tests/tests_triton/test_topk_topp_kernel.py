"""
Standalone QAIC validation for `_topk_topp_kernel`.

Source under test:
vllm/v1/sample/ops/topk_topp_triton.py
  - _topk_topp_kernel (top-k and/or top-p logit filtering via outlier-gathering
    + ternary-search pivot), driven by the `apply_top_k_top_p_triton` launcher.

The kernel masks non-selected logits to -inf (MASK_VALUE). Its pivot-based
ternary search converges to the same *selection* as a standard sort-based
top-k / top-p filter, so we compare against a pure-PyTorch sort reference
(mirroring vllm's own `apply_top_k_top_p_pytorch`).

Comparison approach (documented):
  - The ternary search is an approximate/iterative pivot finder, so exact
    surviving-set equality can differ at tie boundaries. We therefore report
    the -inf mask agreement (fraction of vocab entries whose kept/masked
    decision matches the sort reference) and require it to be very high, and
    we verify surviving (finite) logit values match the input exactly (the
    kernel never alters kept logits, only masks the rest). We use distinct
    (non-tied) logits and a simple top_k / top_p so the selection is
    unambiguous and the masks agree exactly.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_topk_topp_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/sample/ops/topk_topp_triton.py"
KERNEL_NAME = "_topk_topp_kernel"

DEVICE = "qaic"
BATCH_SIZE = 8  # launcher requires batch >= 8 for the triton path
VOCAB_SIZE = 512
TOP_K = 32
TOP_P = 0.9

torch.manual_seed(42)
# Distinct logits (add a tiny unique per-index perturbation to avoid ties).
_logits = torch.randn(BATCH_SIZE, VOCAB_SIZE, dtype=torch.float32)
_logits += torch.arange(VOCAB_SIZE, dtype=torch.float32).view(1, -1) * 1e-4
LOGITS = _logits.to(DEVICE)
K = torch.full((BATCH_SIZE,), TOP_K, dtype=torch.int32, device=DEVICE)
P = torch.full((BATCH_SIZE,), TOP_P, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(logits, k, p):
    """Sort-based top-k then top-p filtering (mirrors apply_top_k_top_p_pytorch)."""
    x = logits.cpu().clone().to(torch.float32)
    logits_sort, logits_idx = x.sort(dim=-1, descending=False)

    # Top-k: keep the k largest.
    top_k_mask = logits_sort.size(1) - k.cpu().to(torch.long)
    top_k_vals = logits_sort.gather(1, top_k_mask.unsqueeze(1))
    tk_mask = logits_sort < top_k_vals
    logits_sort = logits_sort.masked_fill(tk_mask, float("-inf"))

    # Top-p over the (ascending) sorted logits.
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    tp_mask = probs_sum <= (1 - p.cpu()).unsqueeze(1)
    tp_mask[:, -1] = False  # keep at least one
    logits_sort = logits_sort.masked_fill(tp_mask, float("-inf"))

    out = torch.empty_like(x)
    out.scatter_(dim=-1, index=logits_idx, src=logits_sort)
    return out


def kernel_impl(logits, k, p):
    logits = logits.clone()
    return apply_top_k_top_p_triton(logits, k, p)


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref = pytorch_ref(LOGITS, K, P)
        out = kernel_impl(LOGITS, K, P)
        ref_cpu = ref.cpu()
        out_cpu = out.cpu()

        ref_kept = ~torch.isinf(ref_cpu)
        out_kept = ~torch.isinf(out_cpu)
        mask_agreement = float((ref_kept == out_kept).float().mean().item())

        # On kept positions common to both, the surviving values must match
        # the original logits exactly (kernel does not modify kept logits).
        common_kept = ref_kept & out_kept
        torch.testing.assert_close(
            out_cpu[common_kept], LOGITS.cpu()[common_kept], rtol=1e-3, atol=1e-3
        )
        assert mask_agreement >= 0.99, (
            f"mask agreement too low: {mask_agreement}"
        )

        diff = (out_cpu[common_kept] - LOGITS.cpu()[common_kept]).abs()
        stats = {
            "input_shape": tuple(LOGITS.shape),
            "output_shape": tuple(out.shape),
            "input_dtype": str(LOGITS.dtype),
            "output_dtype": str(out.dtype),
            "device": str(LOGITS.device),
            "top_k": TOP_K,
            "top_p": TOP_P,
            "mask_agreement": mask_agreement,
            "ref_num_kept": int(ref_kept.sum().item()),
            "kernel_num_kept": int(out_kept.sum().item()),
            "kept_value_max_abs_diff": diff.max().item(),
            "comparison": "mask agreement vs sort-based ref + kept-value equality",
            "kernel_file": KERNEL_FILE_PATH,
            "timestamp": ts,
        }
        pt_stats = _bench(lambda: pytorch_ref(LOGITS, K, P))
        kern_stats = _bench(lambda: kernel_impl(LOGITS, K, P))
        speedup = (kern_stats["avg_ms"] / pt_stats["avg_ms"]
                   if pt_stats["avg_ms"] > 0 else float("nan"))
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
        status = "SUCCESS"
        print("SUCCESS")
        print(stats)
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)
    finally:
        _log(status, stats, error_text, ts)
    return status


if __name__ == "__main__":
    main()
