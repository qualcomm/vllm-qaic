"""
Standalone QAIC validation for `_fused_moe_lora_kernel_fp8` (SHRINK phase).

Source under test:
vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py
  - _fused_moe_lora_kernel_fp8   (@triton.jit, the FP8 MoE-LoRA GEMM tile kernel)
  - launcher: _fused_moe_lora_shrink_fp8(...)  (invokes the shrink GEMM)

The FP8 MoE-LoRA kernel computes, per (lora, expert, slice, token-block) tile,
a grouped GEMM  C += A @ B  where A is the (fp8) activation and B is the (fp8)
LoRA weight, selected per token via the routing metadata (lora id / expert id /
token offsets resolved by the _get_lora_id / _get_expert_id / _get_token_offs
device helpers). The SHRINK phase projects hidden states (K) down to the LoRA
rank (N = max_lora_rank).

Routing metadata: we deliberately use the NAIVE block-assignment path
(sorted_token_ids=None). This is the source file's own simplest routing path
and bypasses the moe_align / sorted-token metadata helpers entirely, so no
hand-rolled sort/align is needed. In this path:
  - lora_id  = token_lora_mapping[pid_m // top_k_num]
  - expert_id = expert_ids[pid_m]
  - token row = pid_m // token_mapping_factor  (token_mapping_factor = top_k_num
    when mul_routed_weight=False), and each pid_m tile writes exactly its lane-0
    output row into a_intermediate_cache1 viewed as (num_slices, M*top_k, rank).

FP8 comparison choice: we use PER-TENSOR fp8 scales (the simplest quant mode;
per_channel / block-wise are the other constexpr modes). The shrink launcher's
GEMM body accumulates the fp8-operand product; we set the per-tensor scales to
1.0 (identity), so the DEQUANTIZED value equals the raw fp8 GEMM output. The
reference casts activations and LoRA-A weights to float8_e4m3fn (round-trip)
then performs the same per-token grouped matmul, and we compare dequantized
values with atol/rtol ~= 1e-2 (fp8 rounding tolerance).

Small config: num_experts=2, top_k=1, num_tokens=4, hidden(K)=16, rank(N)=8.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.lora.ops.triton_ops.fused_moe_lora_fp8_op import _fused_moe_lora_shrink_fp8

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_fused_moe_lora_kernel_fp8.txt")
KERNEL_FILE_PATH = "vllm/lora/ops/triton_ops/fused_moe_lora_fp8_op.py"

DEVICE = "qaic"
torch.manual_seed(42)

# Problem sizes
M = 4                 # tokens
TOP_K = 1
K = 16                # hidden size (shrink contraction dim)
RANK = 8              # max_lora_rank == N (shrink output dim)
NUM_EXPERTS = 2
MAX_LORAS = 2
NUM_SLICES = 1
NUM_TOKENS = M * TOP_K  # kernel `num_tokens`

# Kernel launch/tile config
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 16
GROUP_M = 1
NUM_WARPS = 4
NUM_STAGES = 2
SPLIT_K = 1
EM = NUM_TOKENS * BLOCK_M

FP8_DTYPE = torch.float8_e4m3fn

# Activations (fp8) and LoRA-A weights (fp8)
QHIDDEN = torch.randn(M, K, dtype=torch.float32, device=DEVICE).to(FP8_DTYPE)
LORA_A = (
    torch.randn(MAX_LORAS, NUM_EXPERTS, RANK, K, dtype=torch.float32, device=DEVICE)
    * 0.1
).to(FP8_DTYPE)

# Per-tensor fp8 scales (identity => dequant == raw fp8 product)
ACT_SCALE = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)
LORA_A_SCALE = [torch.tensor([1.0], dtype=torch.float32, device=DEVICE)]

# Routing metadata (NAIVE path)
TOKEN_LORA_MAPPING = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=DEVICE)
EXPERT_IDS = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device=DEVICE)  # per pid_m
LORA_IDS = torch.arange(MAX_LORAS, dtype=torch.int32, device=DEVICE)       # unused (naive)
ADAPTER_ENABLED = torch.ones(MAX_LORAS, dtype=torch.int32, device=DEVICE)
TOPK_WEIGHTS = torch.rand(M, TOP_K, dtype=torch.float32, device=DEVICE)


def _log(text: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(text)


def pytorch_ref(qhidden, lora_a, token_lora_mapping, expert_ids, act_scale, lora_a_scale):
    """Dequantized per-token MoE-LoRA shrink GEMM (naive block assignment).

    For each pid_m tile (== token row, since top_k=1):
      lora_id   = token_lora_mapping[pid_m // top_k]
      expert_id = expert_ids[pid_m]
      out[n]    = sum_k qhidden_fp8[row, k] * lora_a_fp8[lora_id, expert_id, n, k]
    Per-tensor scales (=1.0) are applied as dequant factors (identity here).
    Result laid out as (num_slices, M, top_k, rank).
    """
    a = qhidden.cpu().to(torch.float32)
    b = lora_a.cpu().to(torch.float32)
    tlm = token_lora_mapping.cpu()
    eids = expert_ids.cpu()
    a_s = float(act_scale.cpu()[0].item())
    b_s = float(lora_a_scale[0].cpu()[0].item())

    out = torch.zeros(NUM_SLICES, M, TOP_K, RANK, dtype=torch.float32)
    for pid_m in range(NUM_TOKENS):
        row = pid_m // TOP_K
        k_slot = pid_m % TOP_K
        lora_id = int(tlm[pid_m // TOP_K].item())
        expert_id = int(eids[pid_m].item())
        a_row = a[row]                       # (K,)
        b_mat = b[lora_id, expert_id]        # (rank, K)
        c = (b_mat * a_row[None, :]).sum(dim=1)   # (rank,)
        out[0, row, k_slot, :] = c * a_s * b_s
    return out


def kernel_impl():
    a_intermediate_cache1 = torch.zeros(
        NUM_SLICES, M, TOP_K, RANK, dtype=torch.float32, device=DEVICE
    )
    _fused_moe_lora_shrink_fp8(
        a_intermediate_cache1,
        QHIDDEN,
        [LORA_A],
        TOPK_WEIGHTS,
        None,                 # sorted_token_ids -> naive path
        EXPERT_IDS,
        None,                 # num_tokens_post_padded
        TOKEN_LORA_MAPPING,
        TOP_K,
        LORA_IDS,
        ADAPTER_ENABLED,
        torch.device(DEVICE),
        RANK,                 # N
        M,
        EM,
        K,
        NUM_TOKENS,
        NUM_EXPERTS,
        NUM_SLICES,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
        NUM_WARPS,
        NUM_STAGES,
        SPLIT_K,
        MAX_LORAS,            # num_active_loras
        LORA_A_SCALE,
        mul_routed_weight=False,
        use_gdc=False,
        act_scale=ACT_SCALE,
        use_fp8_w8a8=True,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        per_channel_quant=False,
        block_shape=None,
    )
    return a_intermediate_cache1


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
        ref = pytorch_ref(
            QHIDDEN, LORA_A, TOKEN_LORA_MAPPING, EXPERT_IDS, ACT_SCALE, LORA_A_SCALE
        )
        out = kernel_impl().cpu().to(torch.float32)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        diff = (out - ref).abs()
        stats = {
            "qhidden_shape": tuple(QHIDDEN.shape),
            "lora_a_shape": tuple(LORA_A.shape),
            "output_shape": tuple(out.shape),
            "fp8_dtype": str(FP8_DTYPE),
            "out_dtype": str(out.dtype),
            "device": DEVICE,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "act_scale": float(ACT_SCALE[0].item()),
            "lora_a_scale": float(LORA_A_SCALE[0][0].item()),
        }
        pt_stats = _bench(lambda: pytorch_ref(
            QHIDDEN, LORA_A, TOKEN_LORA_MAPPING, EXPERT_IDS, ACT_SCALE, LORA_A_SCALE
        ))
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
        print("FAILURE\n" + error_text)
    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _fused_moe_lora_kernel_fp8 (SHRINK phase)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- qhidden(fp8) shape: {stats['qhidden_shape']} dtype {stats['fp8_dtype']}\n",
                f"- lora_a(fp8) shape: {stats['lora_a_shape']}\n",
                f"- num_experts={NUM_EXPERTS}, top_k={TOP_K}, num_tokens={M}, "
                f"K={K}, rank={RANK}, naive_block_assignment=True\n",
                f"- per-tensor scales: act={stats['act_scale']}, "
                f"lora_a={stats['lora_a_scale']}\n",
                f"- device: {stats['device']}\n\n",
                "Output (dequantized-value comparison, rtol/atol=1e-2):\n",
                f"- a_intermediate_cache1 shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
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
