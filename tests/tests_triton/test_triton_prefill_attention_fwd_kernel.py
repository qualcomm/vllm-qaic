"""
Standalone QAIC validation for `_fwd_kernel` (Triton context prefill attention).

Source under test:
vllm/v1/attention/ops/triton_prefill_attention.py
  - _fwd_kernel  (Memory-efficient flash-attention-style causal prefill
    attention over packed [total_tokens, head, head_dim] q/k/v, with
    optional GQA head-repeat and bidirectional sliding-window masking.)

For this single-batch, no-GQA, no-sliding-window configuration, standard
scaled-dot-product causal attention over the packed q/k/v is mathematically
equivalent to the kernel's tiled online-softmax computation, so we use
`torch.nn.functional.scaled_dot_product_attention` (a standard PyTorch op,
not a custom/triton/vllm kernel) as the pure-PyTorch reference.

Reference: pure PyTorch scaled_dot_product_attention with is_causal=True.
"""

import datetime
import math
import os
import sys
import traceback

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_triton_prefill_attention_fwd_kernel.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_prefill_attention.py"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from vllm.v1.attention.ops.triton_prefill_attention import (  # noqa: E402
    context_attention_fwd,
)

# ---------------------------------------------------------------------------
# Global inputs
# ---------------------------------------------------------------------------
DEVICE = "qaic"
SEQ_LEN = 32
NUM_HEADS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 32
IS_CAUSAL = True
SLIDING_WINDOW_Q = None
SLIDING_WINDOW_K = None
SOFTMAX_SCALE = 1.0 / math.sqrt(HEAD_DIM)
MAX_INPUT_LEN = SEQ_LEN

torch.manual_seed(42)

Q = torch.randn(SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
K = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
V = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
B_START_LOC = torch.tensor([0], dtype=torch.int32, device=DEVICE)
B_SEQ_LEN = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)


def pytorch_ref(q, k, v, b_start_loc, b_seq_len, softmax_scale, is_causal):
    """Pure PyTorch reference implementation.

    Requirements (Claude.md):
      - Pure PyTorch only.
      - No custom kernel calls.
      - No Triton kernel calls.
      - No vLLM kernel calls.
      - No QAIC custom operator calls.
    """
    q = q.float()
    k = k.float()
    v = v.float()
    b_start_loc = b_start_loc.cpu()
    b_seq_len = b_seq_len.cpu()

    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_heads // num_kv_heads

    out = torch.zeros_like(q)
    for bidx in range(b_start_loc.shape[0]):
        start = int(b_start_loc[bidx].item())
        length = int(b_seq_len[bidx].item())
        q_b = q[start : start + length]  # [S, H, D]
        k_b = k[start : start + length]  # [S, Hk, D]
        v_b = v[start : start + length]  # [S, Hk, D]

        if kv_group_num > 1:
            k_b = k_b.repeat_interleave(kv_group_num, dim=1)
            v_b = v_b.repeat_interleave(kv_group_num, dim=1)

        # [S, H, D] -> [H, S, D] for SDPA.
        q_t = q_b.transpose(0, 1)
        k_t = k_b.transpose(0, 1)
        v_t = v_b.transpose(0, 1)

        o_t = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=is_causal, scale=softmax_scale
        )
        out[start : start + length] = o_t.transpose(0, 1)

    return out


def kernel_impl(
    q,
    k,
    v,
    b_start_loc,
    b_seq_len,
    max_input_len,
    is_causal,
    softmax_scale,
    sliding_window_q,
    sliding_window_k,
):
    """Kernel wrapper: launch only.

    Requirements (Claude.md):
      - Kernel launch only.
      - Minimal setup logic.
      - No reference implementation logic.
      - No correctness-check logic.
      - No validation logic.
    """
    o = torch.empty_like(q)
    context_attention_fwd(
        q,
        k,
        v,
        o,
        b_start_loc,
        b_seq_len,
        max_input_len,
        is_causal=is_causal,
        softmax_scale=softmax_scale,
        sliding_window_q=sliding_window_q,
        sliding_window_k=sliding_window_k,
    )
    return o


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


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(
            Q, K, V, B_START_LOC, B_SEQ_LEN, SOFTMAX_SCALE, IS_CAUSAL
        )
        kernel_out = kernel_impl(
            Q,
            K,
            V,
            B_START_LOC,
            B_SEQ_LEN,
            MAX_INPUT_LEN,
            IS_CAUSAL,
            SOFTMAX_SCALE,
            SLIDING_WINDOW_Q,
            SLIDING_WINDOW_K,
        )

        ref_out_cpu = ref_out.cpu()
        kernel_out_cpu = kernel_out.cpu()

        torch.testing.assert_close(kernel_out_cpu, ref_out_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_out_cpu - ref_out_cpu).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        rel_err = (diff.max() / (ref_out_cpu.abs().max() + 1e-8)).item()

        stats = {
            "q_shape": tuple(Q.shape),
            "k_shape": tuple(K.shape),
            "v_shape": tuple(V.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(Q.dtype),
            "device": str(Q.device),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "rel_err": rel_err,
            "is_causal": IS_CAUSAL,
            "softmax_scale": SOFTMAX_SCALE,
        }

        pt_stats = _bench(
            lambda: pytorch_ref(
                Q, K, V, B_START_LOC, B_SEQ_LEN, SOFTMAX_SCALE, IS_CAUSAL
            )
        )
        kern_stats = _bench(
            lambda: kernel_impl(
                Q,
                K,
                V,
                B_START_LOC,
                B_SEQ_LEN,
                MAX_INPUT_LEN,
                IS_CAUSAL,
                SOFTMAX_SCALE,
                SLIDING_WINDOW_Q,
                SLIDING_WINDOW_K,
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
        print("SUCCESS")
        print(stats)
        print(f"Speedup (Kernel/PyTorch): {speedup:.4f}x")

    except Exception as e:
        error_text = str(e) + "\n" + traceback.format_exc()
        print("FAILURE")
        print(error_text)

    finally:
        lines = [
            f"{timestamp}\n",
            "Kernel: _fwd_kernel\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines.append("Inputs:\n")
            lines.append(f"- q shape: {stats['q_shape']}\n")
            lines.append(f"- k shape: {stats['k_shape']}\n")
            lines.append(f"- v shape: {stats['v_shape']}\n")
            lines.append(f"- dtype: {stats['dtype']}\n")
            lines.append(f"- device: {stats['device']}\n")
            lines.append(f"- is_causal: {stats['is_causal']}\n")
            lines.append(f"- softmax_scale: {stats['softmax_scale']}\n\n")
            lines.append("Outputs:\n")
            lines.append(f"- output shape: {stats['output_shape']}\n")
            lines.append(f"- max_abs_diff: {stats['max_abs_diff']}\n")
            lines.append(f"- mean_abs_diff: {stats['mean_abs_diff']}\n")
            lines.append(f"- rel_err: {stats['rel_err']}\n")
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
            lines.append("Error:\n")
            lines.append(error_text + "\n")
        lines.append("\n------------------------------------\n\n")
        _log("".join(lines))

    return status


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result == "SUCCESS" else 1)
