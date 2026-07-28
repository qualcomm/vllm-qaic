"""
Standalone QAIC validation for `_gemma4_routing_kernel`.

Source under test:
vllm/model_executor/models/gemma4.py
  - _gemma4_routing_kernel   (@triton.jit)
  - launcher: gemma4_fused_routing_kernel_triton(gating_output, topk,
              per_expert_scale, num_warps=1)

Located inside the large gemma4.py by grepping for the kernel name: the
@triton.jit def is at line ~92 and its single launch site is the
`gemma4_fused_routing_kernel_triton` wrapper at line ~158 (grid=(T,),
BLOCK_E=next_pow2(E)). The file also ships a reference `gemma4_routing_function_torch`
(line ~183) which confirms the intended math.

The kernel computes Gemma4 MoE routing. Mechanism: it maps each fp32 logit to
an ascending-sortable int bijection, packs (key<<32 | expert_id), does a single
`tl.sort`, and reads the top-K sorted entries — i.e. the K largest logits. It
then computes softmax weights renormalized over the selected top-K experts and
folds in per_expert_scale:
    raw_exp   = exp2((logit - max_logit) * log2(e))      # unnormalized softmax
    weight    = raw_exp / sum_{top-K}(raw_exp) * per_expert_scale[id]

pytorch_ref (pure PyTorch, equivalent): probs=softmax over ALL experts, select
top-k, renormalize the top-k weights to sum to 1, multiply by per_expert_scale.
(softmax-over-all then renorm-over-topk == softmax renormalized over top-K.)

Comparison: selected expert ids EXACT and routing weights float rtol/atol=1e-3.
To be robust to any ordering difference between torch.topk and the kernel's
sort, each token's (id, weight) pairs are sorted by expert id before comparing
(ids compared exactly as an integer set-with-order, weights paired accordingly).

Small config: num_tokens=8, num_experts=8, top_k=2.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.models.gemma4 import gemma4_fused_routing_kernel_triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_gemma4_routing_kernel.txt")
KERNEL_FILE_PATH = "vllm/model_executor/models/gemma4.py"

DEVICE = "qaic"
torch.manual_seed(42)

NUM_TOKENS = 8
NUM_EXPERTS = 8
TOP_K = 2

GATING = torch.randn(NUM_TOKENS, NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
PER_EXPERT_SCALE = (
    torch.rand(NUM_EXPERTS, dtype=torch.float32, device=DEVICE) + 0.5
)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(gating, per_expert_scale):
    g = gating.cpu().to(torch.float32)
    scale = per_expert_scale.cpu().to(torch.float32)

    probs = torch.softmax(g, dim=-1)                 # softmax over ALL experts
    top_vals, top_ids = torch.topk(g, k=TOP_K, dim=-1)  # top-k by logit
    top_probs = probs.gather(1, top_ids)             # their softmax mass
    renorm = top_probs.sum(dim=-1, keepdim=True)
    renorm = torch.where(renorm > 0.0, renorm, torch.ones_like(renorm))
    weights = top_probs / renorm                     # renormalized over top-K
    weights = weights * scale[top_ids]               # fold per-expert scale
    return top_ids.to(torch.int32), weights.to(torch.float32)


def kernel_impl(gating, per_expert_scale):
    weights, ids = gemma4_fused_routing_kernel_triton(
        gating, TOP_K, per_expert_scale, num_warps=1
    )
    return ids, weights


def _sort_by_id(ids, weights):
    """Sort each row's (id, weight) pairs by expert id for order-robust compare."""
    order = torch.argsort(ids, dim=-1)
    return ids.gather(1, order), weights.gather(1, order)


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_ids, ref_w = pytorch_ref(GATING, PER_EXPERT_SCALE)
        k_ids, k_w = kernel_impl(GATING, PER_EXPERT_SCALE)
        k_ids = k_ids.cpu().to(torch.int32)
        k_w = k_w.cpu().to(torch.float32)

        ref_ids_s, ref_w_s = _sort_by_id(ref_ids, ref_w)
        k_ids_s, k_w_s = _sort_by_id(k_ids, k_w)

        assert torch.equal(k_ids_s, ref_ids_s), (
            f"expert id mismatch:\n{k_ids_s}\nvs\n{ref_ids_s}"
        )
        torch.testing.assert_close(k_w_s, ref_w_s, rtol=1e-3, atol=1e-3)

        diff = (k_w_s - ref_w_s).abs()
        stats = {
            "gating_shape": tuple(GATING.shape),
            "ids_shape": tuple(k_ids.shape),
            "weights_shape": tuple(k_w.shape),
            "in_dtype": str(GATING.dtype),
            "ids_dtype": str(k_ids.dtype),
            "device": DEVICE,
            "ids_exact": True,
            "w_max_abs_diff": diff.max().item(),
            "w_mean_abs_diff": diff.mean().item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(GATING, PER_EXPERT_SCALE))
        kern_stats = _bench(lambda: kernel_impl(GATING, PER_EXPERT_SCALE))
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
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
            "Kernel: _gemma4_routing_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- gating shape: {stats['gating_shape']} dtype {stats['in_dtype']}\n",
                f"- num_experts={NUM_EXPERTS}, top_k={TOP_K}, num_tokens={NUM_TOKENS}\n",
                f"- device: {stats['device']}\n\n",
                "Output (ids EXACT, weights rtol/atol=1e-3; per-row id-sorted):\n",
                f"- topk_ids shape: {stats['ids_shape']} dtype {stats['ids_dtype']} "
                f"exact: {stats['ids_exact']}\n",
                f"- topk_weights shape: {stats['weights_shape']}\n",
                f"- weights max_abs_diff: {stats['w_max_abs_diff']}\n",
                f"- weights mean_abs_diff: {stats['w_mean_abs_diff']}\n",
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
