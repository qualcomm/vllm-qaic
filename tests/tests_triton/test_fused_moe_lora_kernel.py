"""
Standalone QAIC validation for `_fused_moe_lora_kernel` (SHRINK phase).

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_op.py
  - _fused_moe_lora_kernel  (legacy two-phase MoE-LoRA GEMM dispatched per
    (lora, expert, slice, token-block) tile; one launch does EITHER shrink OR
    expand)

PHASE TESTED: SHRINK. We use the file's `_fused_moe_lora_shrink` launcher,
which invokes `_fused_moe_lora_kernel` with the hidden states as A and the
LoRA-A weights as B, writing the rank-dim result into an intermediate cache.
Shrink is the simpler phase (no add-into-output, no routed-weight scaling).

We drive the naive-block-assignment path (sorted_token_ids=None): each program
(pid_m) handles one flat (token, top_k) pair at lane 0; token = pid_m//top_k,
expert = expert_ids[pid_m], lora = token_lora_mapping[token]. The cache is
addressed as (num_slices, num_tokens*top_k, rank); row p holds
  cache[p] = x[token] @ A[lora, expert].T.

NOTE: the kernel casts both operands to bfloat16 before tl.dot (see the
`a.to(tl.bfloat16)` in the K-loop), so we use bf16 tensors and a relaxed
tolerance for the FLOAT compare.

Reference: pure PyTorch per-pair x @ A[lora, expert].T. FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fused_moe_lora_op import fused_moe_lora_shrink

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_moe_lora_kernel.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_op.py"
DEVICE = "qaic"
DTYPE = torch.bfloat16

torch.manual_seed(42)

# Global shared inputs
NUM_TOKENS = 4
TOP_K = 1
HIDDEN = 16          # K
RANK = 8             # N (shrink output dim)
NUM_EXPERTS = 2
MAX_LORAS = 2
NUM_SLICES = 1
FLAT_PAIRS = NUM_TOKENS * TOP_K

BLOCK_SIZE_M = 16
BLOCK_SIZE_N = 8
BLOCK_SIZE_K = 16
GROUP_SIZE_M = 1
NUM_WARPS = 4
NUM_STAGES = 2
SPLIT_K = 1
EM = FLAT_PAIRS * BLOCK_SIZE_M  # naive-path EM

X = (torch.randn(NUM_TOKENS, HIDDEN, device=DEVICE) * 0.5).to(DTYPE)
LORA_A = [(torch.randn(MAX_LORAS, NUM_EXPERTS, RANK, HIDDEN, device=DEVICE) * 0.5).to(DTYPE)]
TOPK_WEIGHTS = torch.rand(NUM_TOKENS, TOP_K, dtype=DTYPE, device=DEVICE)
EXPERT_IDS = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
TOKEN_LORA_MAPPING = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
LORA_IDS = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
ADAPTER_ENABLED = torch.ones(MAX_LORAS, dtype=torch.int32, device=DEVICE)
NUM_ACTIVE_LORAS = torch.tensor([MAX_LORAS], dtype=torch.int32, device="cpu")


def pytorch_ref():
    # cache viewed as (num_slices, flat_pairs, rank)
    out = torch.zeros(NUM_SLICES, FLAT_PAIRS, RANK, dtype=torch.float32, device=DEVICE)
    xf = X.to(torch.bfloat16)
    a = LORA_A[0].to(torch.bfloat16)
    for p in range(FLAT_PAIRS):
        token = p // TOP_K
        eid = int(EXPERT_IDS[p].item())
        lid = int(TOKEN_LORA_MAPPING[token].item())
        if lid < 0 or eid < 0:
            continue
        out[0, p] = (xf[token].float() @ a[lid, eid].float().t())
    return out


def kernel_impl():
    cache = torch.zeros(NUM_SLICES, NUM_TOKENS, TOP_K, RANK, dtype=DTYPE, device=DEVICE)
    fused_moe_lora_shrink(
        cache,
        X,
        LORA_A,
        TOPK_WEIGHTS,
        None,               # sorted_token_ids -> naive path
        EXPERT_IDS,
        None,               # num_tokens_post_padded
        TOKEN_LORA_MAPPING,
        TOP_K,
        LORA_IDS,
        ADAPTER_ENABLED,
        DEVICE,
        RANK,               # N
        NUM_TOKENS,         # M
        EM,
        HIDDEN,             # K
        FLAT_PAIRS,         # num_tokens (num_valid_tokens)
        NUM_EXPERTS,
        NUM_SLICES,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        GROUP_SIZE_M,
        NUM_WARPS,
        NUM_STAGES,
        SPLIT_K,
        NUM_ACTIVE_LORAS,
        mul_routed_weight=False,
        use_gdc=False,
        use_tma=False,
    )
    return cache.view(NUM_SLICES, FLAT_PAIRS, RANK)


def _log(text):
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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref()
        kernel_out = kernel_impl().float()
        ref_cpu = ref_out.cpu()
        ker_cpu = kernel_out.cpu()
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=5e-2, atol=5e-2)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "phase": "shrink",
            "input_shapes": [tuple(X.shape), tuple(LORA_A[0].shape)],
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(X.dtype),
            "device": str(X.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref())
        kern_stats = _bench(lambda: kernel_impl())
        speedup = kern_stats["avg_ms"] / pt_stats["avg_ms"] if pt_stats["avg_ms"] > 0 else float("nan")
        stats["pytorch_latency_ms"] = pt_stats
        stats["kernel_latency_ms"] = kern_stats
        stats["speedup_kernel_over_pytorch"] = speedup
        status = "SUCCESS"
        print("SUCCESS", stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")
    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE\n", error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _fused_moe_lora_kernel [SHRINK] (launcher fused_moe_lora_shrink)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n",
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
        _log("".join(lines))
    return status


if __name__ == "__main__":
    main()
