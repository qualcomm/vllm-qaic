"""
Standalone QAIC validation for `_fwd_kernel_ep_gather`.

Source under test:
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
  - _fwd_kernel_ep_gather  (@triton.jit)

The inverse of the ep-scatter: after the per-expert grouped GEMM, this kernel
gathers each expert's output rows back into their original token order and
combines the top-k contributions with a weighted sum. For each original token
and each of its top-k experts it reads the source row (via `input_index`, the
inverse-permutation produced by the scatter), multiplies by the routing weight
`recv_topk_weight`, and accumulates into `output_tensor[token]`.

Launcher: replicated from the repo's `ep_gather`
(grid = (cdiv(hidden, BLOCK_D), min(num_tokens, 1024)),
BLOCK_D = min(hidden, 1024)). HAS_EXPERT_MAP is False.

Reference: pure PyTorch weighted top-k gather-sum:
    out[t] = sum_k  input_tensor[input_index[t, k]] * recv_topk_weight[t, k]
(summing only over valid experts, expert_id >= 0).
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    _fwd_kernel_ep_gather,
)
from vllm.triton_utils import triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_deep_gemm_fwd_kernel_ep_gather.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py"

DEVICE = "qaic"
NUM_EXPERTS = 4
TOPK_NUM = 2
NUM_TOKENS = 8
HIDDEN_SIZE = 16
SRC_ROWS = NUM_TOKENS * TOPK_NUM  # packed source rows

torch.manual_seed(42)
# Per-expert GEMM output rows (packed / expert-sorted layout).
INPUT_TENSOR = torch.randn(SRC_ROWS, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
# Each token picks TOPK_NUM distinct experts (all valid).
RECV_TOPK_IDS = torch.stack(
    [torch.randperm(NUM_EXPERTS, device=DEVICE)[:TOPK_NUM] for _ in range(NUM_TOKENS)]
).to(torch.int32)
RECV_TOPK_WEIGHT = torch.rand(
    NUM_TOKENS, TOPK_NUM, dtype=torch.float32, device=DEVICE
)
# input_index: source row for each (token, topk). Use a permutation of all
# packed rows so every source row is consumed exactly once.
INPUT_INDEX = torch.randperm(SRC_ROWS, device=DEVICE).reshape(
    NUM_TOKENS, TOPK_NUM
).to(torch.int32)


def _log(text: str) -> None:
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


def pytorch_ref(input_tensor, recv_topk_ids, recv_topk_weight, input_index):
    """Pure PyTorch weighted top-k gather-sum."""
    input_tensor = input_tensor.cpu()
    recv_topk_ids = recv_topk_ids.cpu()
    recv_topk_weight = recv_topk_weight.cpu()
    input_index = input_index.cpu()

    out = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32)
    for t in range(NUM_TOKENS):
        acc = torch.zeros(HIDDEN_SIZE, dtype=torch.float32)
        for k in range(TOPK_NUM):
            e = int(recv_topk_ids[t, k].item())
            if e >= 0:
                src = int(input_index[t, k].item())
                w = float(recv_topk_weight[t, k].item())
                acc += input_tensor[src].to(torch.float32) * w
        out[t] = acc
    return out


def kernel_impl(input_tensor, recv_topk_ids, recv_topk_weight, input_index):
    output_tensor = torch.zeros(
        NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32, device=input_tensor.device
    )
    BLOCK_D = min(HIDDEN_SIZE, 1024)
    grid = (triton.cdiv(HIDDEN_SIZE, BLOCK_D), min(NUM_TOKENS, 1024))
    _fwd_kernel_ep_gather[grid](
        NUM_TOKENS,
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=TOPK_NUM,
        expert_map=None,
        HAS_EXPERT_MAP=False,
        BLOCK_D=BLOCK_D,
        num_warps=2,
    )
    return output_tensor


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(
            INPUT_TENSOR, RECV_TOPK_IDS, RECV_TOPK_WEIGHT, INPUT_INDEX
        )
        kernel_out = kernel_impl(
            INPUT_TENSOR, RECV_TOPK_IDS, RECV_TOPK_WEIGHT, INPUT_INDEX
        )

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        denom = ref_cpu.abs().max().clamp_min(1e-12)
        stats = {
            "input_shape": tuple(INPUT_TENSOR.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(INPUT_TENSOR.dtype),
            "device": str(INPUT_TENSOR.device),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_error": (diff.max() / denom).item(),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                INPUT_TENSOR, RECV_TOPK_IDS, RECV_TOPK_WEIGHT, INPUT_INDEX
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                INPUT_TENSOR, RECV_TOPK_IDS, RECV_TOPK_WEIGHT, INPUT_INDEX
            )
        )
        speedup = (
            kern_stats["avg_ms"] / pt_stats["avg_ms"]
            if pt_stats["avg_ms"] > 0
            else float("nan")
        )
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
            "Kernel: _fwd_kernel_ep_gather\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- input_tensor shape: {stats['input_shape']}\n",
                f"- num_tokens={NUM_TOKENS}, num_experts={NUM_EXPERTS}, "
                f"topk={TOPK_NUM}, hidden={HIDDEN_SIZE}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output_tensor shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- rel_error: {stats['rel_error']}\n",
            ]
            if "pytorch_latency_ms" in stats:
                lines.append("Timing:\n")
                lines.append(
                    f"- PyTorch latency (ms): avg={stats['pytorch_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['pytorch_latency_ms']['min_ms']:.4f} "
                    f"max={stats['pytorch_latency_ms']['max_ms']:.4f} "
                    f"median={stats['pytorch_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Kernel latency (ms): avg={stats['kernel_latency_ms']['avg_ms']:.4f} "
                    f"min={stats['kernel_latency_ms']['min_ms']:.4f} "
                    f"max={stats['kernel_latency_ms']['max_ms']:.4f} "
                    f"median={stats['kernel_latency_ms']['median_ms']:.4f}\n"
                )
                lines.append(
                    f"- Speedup (Kernel/PyTorch): {stats['speedup_kernel_over_pytorch']:.4f}x\n"
                )
        else:
            lines += ["Error:\n", error_text + "\n"]
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))
    return status


if __name__ == "__main__":
    sys.exit(0 if main() == "SUCCESS" else 1)
