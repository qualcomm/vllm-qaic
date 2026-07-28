"""
Standalone QAIC validation for `_fused_moe_lora_one_shot_kernel`.

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_op.py
  - _fused_moe_lora_one_shot_kernel  (fully-fused MoE-LoRA: shrink + expand in
    one launch, rank-dim intermediate kept in fp32 registers)

Launcher: `_run_fused_moe_lora_one_shot` (the file's dedicated entry wrapper).
We drive the naive-block-assignment path (sorted_token_ids=None), which needs
no token-sort alignment metadata: each program handles one flat (token, top_k)
pair, resolving lora_id via token_lora_mapping[token] and expert_id via the
flat expert_ids table.

Per token the kernel computes:
  shrink   = x[token] @ A[lora, expert].T           # (rank,)
  out[token, slice] = shrink @ B[lora, expert].T     # (N_per_slice,)
with A shape (max_loras, num_experts, rank, K) and B shape
(max_loras, num_experts, N_per_slice, rank). mul_routed_weight=False and
add_inputs=False (kernel overwrites output with the LoRA delta).

Reference: pure PyTorch per-token shrink-then-expand. FLOAT compare.
"""

import datetime
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

from vllm.lora.ops.triton_ops.fused_moe_lora_op import _run_fused_moe_lora_one_shot

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_moe_lora_one_shot_kernel.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_op.py"
DEVICE = "qaic"
DTYPE = torch.float16

torch.manual_seed(42)

# Global shared inputs
NUM_TOKENS = 4
TOP_K = 1
HIDDEN = 16          # K
RANK = 8
N_PER_SLICE = 16     # N
NUM_EXPERTS = 2
MAX_LORAS = 2
NUM_SLICES = 1
BLOCK_SIZE_M = 16    # ignored by naive path but required by signature

X = torch.randn(NUM_TOKENS, HIDDEN, dtype=DTYPE, device=DEVICE)
LORA_A = [torch.randn(MAX_LORAS, NUM_EXPERTS, RANK, HIDDEN, dtype=DTYPE, device=DEVICE)]
LORA_B = [torch.randn(MAX_LORAS, NUM_EXPERTS, N_PER_SLICE, RANK, dtype=DTYPE, device=DEVICE)]
TOPK_WEIGHTS = torch.rand(NUM_TOKENS, TOP_K, dtype=DTYPE, device=DEVICE)
# flat expert ids per (token, top_k) pair
EXPERT_IDS = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
TOKEN_LORA_MAPPING = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
LORA_IDS = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
NUM_ACTIVE_LORAS = torch.tensor([MAX_LORAS], dtype=torch.int32, device="cpu")
ADAPTER_ENABLED = torch.ones(MAX_LORAS, dtype=torch.int32, device=DEVICE)


def pytorch_ref():
    out = torch.zeros(NUM_TOKENS, TOP_K, NUM_SLICES * N_PER_SLICE,
                      dtype=torch.float32, device=DEVICE)
    xf = X.float()
    for t in range(NUM_TOKENS):
        lid = int(TOKEN_LORA_MAPPING[t].item())
        eid = int(EXPERT_IDS[t].item())
        if lid < 0 or eid < 0:
            continue
        shrink = xf[t] @ LORA_A[0][lid, eid].float().t()   # (rank,)
        expand = shrink @ LORA_B[0][lid, eid].float().t()  # (N_per_slice,)
        out[t, 0, 0:N_PER_SLICE] = expand
    return out


def kernel_impl():
    out = torch.zeros(NUM_TOKENS, TOP_K, NUM_SLICES * N_PER_SLICE,
                      dtype=DTYPE, device=DEVICE)
    _run_fused_moe_lora_one_shot(
        out,
        X,
        LORA_A,
        LORA_B,
        TOPK_WEIGHTS,
        None,               # sorted_token_ids -> naive path
        EXPERT_IDS,
        None,               # num_tokens_post_padded
        TOKEN_LORA_MAPPING,
        RANK,               # max_lora_rank
        TOP_K,
        LORA_IDS,
        NUM_ACTIVE_LORAS,
        ADAPTER_ENABLED,
        False,              # mul_routed_weight
        BLOCK_SIZE_M,
        add_inputs=False,
    )
    return out


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
        # fp16 matmul accumulation -> relaxed tolerance.
        torch.testing.assert_close(ker_cpu, ref_cpu, rtol=2e-2, atol=2e-2)
        diff = (ker_cpu - ref_cpu).abs()
        stats = {
            "input_shapes": [tuple(X.shape), tuple(LORA_A[0].shape), tuple(LORA_B[0].shape)],
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
            "Kernel: _fused_moe_lora_one_shot_kernel (launcher _run_fused_moe_lora_one_shot)\n",
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
