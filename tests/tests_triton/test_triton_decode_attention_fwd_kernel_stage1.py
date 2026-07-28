"""
Standalone QAIC validation for `_fwd_kernel_stage1`.

Source under test:
vllm/v1/attention/ops/triton_decode_attention.py
  - _fwd_kernel_stage1  (split-KV flash-decoding stage 1: per-split paged
    attention scores + online-softmax value accumulation)
  - _decode_att_m_fwd   (Python launcher used here)

Stage 1 partitions the KV sequence into ``NUM_KV_SPLITS`` contiguous splits.
For each (batch, head, split) it computes, over that split's KV tokens:
    qk       = (q . k_i) * sm_scale                 (logit_cap tanh optional)
    m        = max_i qk_i
    p_i      = exp(qk_i - m)
    e_sum    = sum_i p_i
and stores into ``att_out[b, h, split, :]``:
    [:Lv] = (sum_i p_i * v_i) / e_sum   (per-split softmax-normalized output)
    [Lv]  = m + log(e_sum)              (per-split log-sum-exp)
Empty splits (split_kv_end <= split_kv_start) are left untouched.

Config: single request, single MHA head (kv_group_num == 1), page_size == 1,
NUM_KV_SPLITS chosen so every split is non-empty; non-FP8 K/V cache. Paged
access uses an identity Req_to_tokens mapping (token i -> slot i).

Reference: pure PyTorch per-split normalized attention + per-split LSE.
"""

import datetime
import math
import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vllm"))

import torch  # noqa: E402

from vllm.v1.attention.ops.triton_decode_attention import (  # noqa: E402
    _decode_att_m_fwd,
)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs_v23")
LOG_FILE = os.path.join(LOG_DIR, "log_triton_decode_attention_fwd_kernel_stage1.txt")
KERNEL_FILE_PATH = "vllm/v1/attention/ops/triton_decode_attention.py"

DEVICE = "qaic"
BATCH = 1
HEAD = 1
KV_HEAD = 1  # kv_group_num == 1 (MHA)
HEAD_DIM = 32
SEQ_LEN = 16
NUM_KV_SPLITS = 4  # 16 / 4 = 4 tokens per split -> all non-empty
PAGE_SIZE = 1
LOGIT_CAP = 0.0
SM_SCALE = 1.0 / math.sqrt(HEAD_DIM)

torch.manual_seed(42)

Q = torch.randn(BATCH, HEAD, HEAD_DIM, dtype=torch.float32, device=DEVICE)
# K/V buffers: [num_tokens, num_kv_heads, head_dim]
K_BUFFER = torch.randn(SEQ_LEN, KV_HEAD, HEAD_DIM, dtype=torch.float32, device=DEVICE)
V_BUFFER = torch.randn(SEQ_LEN, KV_HEAD, HEAD_DIM, dtype=torch.float32, device=DEVICE)
# Identity paging: request 0's token i lives at slot i.
REQ_TO_TOKENS = torch.arange(SEQ_LEN, dtype=torch.int32, device=DEVICE).reshape(1, SEQ_LEN)
B_SEQLEN = torch.tensor([SEQ_LEN], dtype=torch.int32, device=DEVICE)
K_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
V_SCALE = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)


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


def pytorch_ref(q, k_buffer, v_buffer, sm_scale):
    """Pure PyTorch per-split normalized attention + per-split LSE.

    Returns att_out of shape [BATCH, HEAD, NUM_KV_SPLITS, HEAD_DIM + 1].
    """
    q = q.float().cpu()
    k = k_buffer.float().cpu()
    v = v_buffer.float().cpu()
    out = torch.zeros(BATCH, HEAD, NUM_KV_SPLITS, HEAD_DIM + 1, dtype=torch.float32)

    kv_len_per_split = math.ceil(SEQ_LEN / NUM_KV_SPLITS)
    for b in range(BATCH):
        for h in range(HEAD):
            kv_h = h  # kv_group_num == 1
            qh = q[b, h]  # [D]
            for s in range(NUM_KV_SPLITS):
                start = kv_len_per_split * s
                end = min(start + kv_len_per_split, SEQ_LEN)
                if end <= start:
                    continue
                k_s = k[start:end, kv_h]  # [n, D]
                v_s = v[start:end, kv_h]  # [n, D]
                scores = (k_s @ qh) * sm_scale  # [n]
                m = scores.max()
                p = torch.exp(scores - m)
                e_sum = p.sum()
                out[b, h, s, :HEAD_DIM] = (p.unsqueeze(-1) * v_s).sum(0) / e_sum
                out[b, h, s, HEAD_DIM] = m + torch.log(e_sum)
    return out


def kernel_impl(q, k_buffer, v_buffer, req_to_tokens, b_seqlen, sm_scale):
    """Kernel wrapper: launch only."""
    att_out = torch.zeros(
        BATCH, HEAD, NUM_KV_SPLITS, HEAD_DIM + 1, dtype=torch.float32, device=DEVICE
    )
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        att_out,
        req_to_tokens,
        b_seqlen,
        NUM_KV_SPLITS,
        sm_scale,
        PAGE_SIZE,
        LOGIT_CAP,
        K_SCALE,
        V_SCALE,
    )
    return att_out


def main():
    status = "FAILURE"
    error_text = ""
    stats = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        ref_out = pytorch_ref(Q, K_BUFFER, V_BUFFER, SM_SCALE)
        kernel_out = kernel_impl(Q, K_BUFFER, V_BUFFER, REQ_TO_TOKENS, B_SEQLEN, SM_SCALE)

        ref_cpu = ref_out.cpu()
        kernel_cpu = kernel_out.cpu()
        torch.testing.assert_close(kernel_cpu, ref_cpu, rtol=1e-3, atol=1e-3)

        diff = (kernel_cpu - ref_cpu).abs()
        stats = {
            "q_shape": tuple(Q.shape),
            "k_shape": tuple(K_BUFFER.shape),
            "v_shape": tuple(V_BUFFER.shape),
            "output_shape": tuple(kernel_out.shape),
            "dtype": str(Q.dtype),
            "device": str(Q.device),
            "num_kv_splits": NUM_KV_SPLITS,
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "rel_err": (diff.max() / (ref_cpu.abs().max() + 1e-8)).item(),
        }
        pt_stats = _bench(lambda: pytorch_ref(Q, K_BUFFER, V_BUFFER, SM_SCALE))
        kern_stats = _bench(
            lambda: kernel_impl(Q, K_BUFFER, V_BUFFER, REQ_TO_TOKENS, B_SEQLEN, SM_SCALE)
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
            "Kernel: _fwd_kernel_stage1 (split-KV decode stage 1)\n",
            f"Kernel file: {KERNEL_FILE_PATH}\n",
            f"Device target: QAIC (device='{DEVICE}')\n",
            f"Status: {status}\n\n",
        ]
        if status == "SUCCESS":
            lines += [
                "Inputs:\n",
                f"- q shape: {stats['q_shape']}\n",
                f"- k shape: {stats['k_shape']}\n",
                f"- v shape: {stats['v_shape']}\n",
                f"- num_kv_splits: {stats['num_kv_splits']}\n",
                f"- dtype: {stats['dtype']}\n",
                f"- device: {stats['device']}\n\n",
                "Output:\n",
                f"- att_out shape: {stats['output_shape']}\n",
                f"- max_abs_diff: {stats['max_abs_diff']}\n",
                f"- mean_abs_diff: {stats['mean_abs_diff']}\n",
                f"- rel_err: {stats['rel_err']}\n",
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
