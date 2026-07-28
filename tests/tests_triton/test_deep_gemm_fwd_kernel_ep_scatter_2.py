"""
Standalone QAIC validation for `_fwd_kernel_ep_scatter_2`.

Source under test:
vllm/model_executor/layers/fused_moe/deep_gemm_utils.py
  - _fwd_kernel_ep_scatter_2  (@triton.jit)

Stage-2 of the DeepGEMM expert-parallel scatter. For every received token and
every one of its top-k experts, it:
  * reserves a destination slot inside that expert's contiguous region by
    atomically incrementing `expert_start_loc[expert_id]`,
  * copies the token's activation row (`recv_x`) and its quant-scale row
    (`recv_x_scale`) into `output_tensor` / `output_tensor_scale` at that slot,
  * records the assigned slot into `output_index[token, topk]` (used later by
    the gather kernel to undo the permutation).

Determinism note: the source uses `tl.atomic_add` for slot assignment, whose
ordering across parallel programs is nondeterministic. To make an exact
comparison possible we launch with **grid = (1,)** so a single program walks the
tokens in order (start_token_id=0, grid_num=1), giving a deterministic,
sequential slot assignment that the PyTorch reference reproduces exactly.

Dtype note: token/scale payloads are copied verbatim by the kernel (it is
dtype-agnostic), so we use a float32 proxy for both `recv_x` and its scale for
simplicity. HAS_EXPERT_MAP is False (no EP remap).

Reference: pure PyTorch deterministic scatter by running per-expert counters
seeded from `expert_start_loc`.
"""

import datetime
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch

from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    _fwd_kernel_ep_scatter_2,
)
from vllm.triton_utils import triton

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_deep_gemm_fwd_kernel_ep_scatter_2.txt")
KERNEL_FILE_PATH = "vllm/model_executor/layers/fused_moe/deep_gemm_utils.py"

DEVICE = "qaic"
NUM_EXPERTS = 4
TOPK_NUM = 2
NUM_TOKENS = 8
HIDDEN_SIZE = 16
BLOCK_D = 8  # quant block size -> scale hidden = HIDDEN_SIZE // BLOCK_D
SCALE_HIDDEN_SIZE = HIDDEN_SIZE // BLOCK_D  # = 2
BLOCK_E = 128  # per-expert alignment (matches ep_scatter)

torch.manual_seed(42)
RECV_X = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)
RECV_X_SCALE = torch.randn(
    NUM_TOKENS, SCALE_HIDDEN_SIZE, dtype=torch.float32, device=DEVICE
)
# Each token picks TOPK_NUM distinct experts (all valid, no -1).
RECV_TOPK = torch.stack(
    [torch.randperm(NUM_EXPERTS, device=DEVICE)[:TOPK_NUM] for _ in range(NUM_TOKENS)]
).to(torch.int32)

# Per-expert received counts (how many (token,topk) pairs hit each expert).
_counts = torch.bincount(RECV_TOPK.reshape(-1), minlength=NUM_EXPERTS)
_aligned = ((_counts.to(torch.int64) + (BLOCK_E - 1)) // BLOCK_E) * BLOCK_E
EXPERT_START_LOC = (torch.cumsum(_aligned, 0) - _aligned).to(torch.int32).to(DEVICE)
M_SUM = int(_aligned.sum().item())


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


def pytorch_ref(recv_x, recv_x_scale, recv_topk, expert_start_loc):
    """Pure PyTorch deterministic sequential scatter."""
    recv_x = recv_x.cpu()
    recv_x_scale = recv_x_scale.cpu()
    recv_topk = recv_topk.cpu()
    counter = expert_start_loc.cpu().clone().to(torch.int64)

    out = torch.zeros(M_SUM, HIDDEN_SIZE, dtype=torch.float32)
    out_scale = torch.zeros(M_SUM, SCALE_HIDDEN_SIZE, dtype=torch.float32)
    out_index = torch.full((NUM_TOKENS, TOPK_NUM), -1, dtype=torch.int32)

    for token in range(NUM_TOKENS):
        for topk in range(TOPK_NUM):
            e = int(recv_topk[token, topk].item())
            if e >= 0:
                dest = int(counter[e].item())
                counter[e] += 1
                out[dest] = recv_x[token]
                out_scale[dest] = recv_x_scale[token]
                out_index[token, topk] = dest
    return out, out_scale, out_index


def kernel_impl(recv_x, recv_x_scale, recv_topk, expert_start_loc):
    esl = expert_start_loc.clone()  # kernel mutates via atomic_add
    output_tensor = torch.zeros(
        M_SUM, HIDDEN_SIZE, dtype=torch.float32, device=recv_x.device
    )
    output_scale = torch.zeros(
        M_SUM, SCALE_HIDDEN_SIZE, dtype=torch.float32, device=recv_x.device
    )
    output_index = torch.full(
        (NUM_TOKENS, TOPK_NUM), -1, dtype=torch.int32, device=recv_x.device
    )
    _fwd_kernel_ep_scatter_2[(1,)](
        NUM_TOKENS,
        esl,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_scale,
        output_scale.stride(0),
        output_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        topk_num=TOPK_NUM,
        expert_map=None,
        HAS_EXPERT_MAP=False,
        HIDDEN_SIZE=HIDDEN_SIZE,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(HIDDEN_SIZE),
        SCALE_HIDDEN_SIZE=SCALE_HIDDEN_SIZE,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(SCALE_HIDDEN_SIZE),
        num_warps=8,
    )
    return output_tensor, output_scale, output_index


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_x, ref_s, ref_i = pytorch_ref(
            RECV_X, RECV_X_SCALE, RECV_TOPK, EXPERT_START_LOC
        )
        k_x, k_s, k_i = kernel_impl(RECV_X, RECV_X_SCALE, RECV_TOPK, EXPERT_START_LOC)

        k_x_c, k_s_c, k_i_c = k_x.cpu(), k_s.cpu(), k_i.cpu()

        idx_ok = bool(torch.equal(k_i_c, ref_i))
        assert idx_ok, "output_index mismatch"
        torch.testing.assert_close(k_x_c, ref_x, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(k_s_c, ref_s, rtol=1e-3, atol=1e-3)

        diff = (k_x_c - ref_x).abs()
        stats = {
            "recv_x_shape": tuple(RECV_X.shape),
            "out_shape": tuple(k_x.shape),
            "out_index_shape": tuple(k_i.shape),
            "dtype": str(RECV_X.dtype),
            "device": str(RECV_X.device),
            "m_sum": M_SUM,
            "index_exact_match": idx_ok,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
        }

        pt_stats = _bench(
            lambda: pytorch_ref(RECV_X, RECV_X_SCALE, RECV_TOPK, EXPERT_START_LOC)
        )
        kern_stats = _bench(
            lambda: kernel_impl(RECV_X, RECV_X_SCALE, RECV_TOPK, EXPERT_START_LOC)
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
            "Kernel: _fwd_kernel_ep_scatter_2\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- recv_x shape: {stats['recv_x_shape']}\n",
                f"- num_tokens={NUM_TOKENS}, num_experts={NUM_EXPERTS}, "
                f"topk={TOPK_NUM}, hidden={HIDDEN_SIZE}\n",
                f"- m_sum (packed rows): {stats['m_sum']}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- output_tensor shape: {stats['out_shape']}\n",
                f"- output_index shape: {stats['out_index_shape']}\n",
                f"- index_exact_match: {stats['index_exact_match']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
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
